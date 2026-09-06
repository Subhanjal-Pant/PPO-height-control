import os
import time
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from pneumatic_env import PneumaticAirSpringEnv as PneumaticEnv
from train_lagrangian import LagrangianSolver

torch.set_num_threads(1)

SOLVER = LagrangianSolver(
    cost_limits={
        "cost_overshoot_up": 0.35,
        "cost_overshoot_down": 0.35,
        "cost_duty_rate_inflate": 0.3,
        "cost_duty_rate_exhaust": 0.5,
    },
    learning_rates={
        "cost_overshoot_up": 0.01,
        "cost_overshoot_down": 0.01,
        "cost_duty_rate_inflate": 0.01,
        "cost_duty_rate_exhaust": 0.01,
    },
    max_lambda=3.5,
)


class LagrangianWrapper(gym.Wrapper):
    """Worker process IPC wrapper for dynamic cost penalization."""

    def __init__(self, env):
        super().__init__(env)
        self.lambdas = {
            "cost_overshoot_up": 0.0,
            "cost_overshoot_down": 0.0,
            "cost_duty_rate_inflate": 0.0,
            "cost_duty_rate_exhaust": 0.0,
        }

    def set_lambdas(self, lambdas_dict: dict):
        self.lambdas = lambdas_dict.copy()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        costs = info.get("costs", {})

        penalized_reward = reward
        for key, lambda_val in self.lambdas.items():
            cost_val = costs.get(key, 0.0)
            penalized_reward -= lambda_val * cost_val

        return obs, float(penalized_reward), terminated, truncated, info


class LagrangianCallback(BaseCallback):
    """Collects per-step direction-split costs across the rollout and
    updates the Lagrangian dual multipliers at rollout end.

    FIX (this version): each cost key's rolling buffer now only receives a
    sample on steps where `info["cost_active"][key]` is True. Previously
    every key received a sample on every step -- including the steps where
    that direction was force-zeroed as a placeholder (e.g.
    cost_duty_rate_inflate = 0.0 on every step duty_cmd was negative). Since
    duty flips sign roughly every other step under a chattering policy,
    that inflated each buffer with ~50% irrelevant zeros, roughly halving
    the reported mean cost relative to the TRUE mean cost conditional on
    that direction being active. That made the dual solver under-react:
    lambda_duty_rate_* only started climbing once the true per-active-step
    duty-rate was already ~2x its intended 0.3 limit, so the penalty never
    got strong enough to stop the chatter -- reproducing the original
    "never settles" failure mode via a different mechanism. Filtering by
    cost_active restores each mean to "average cost given this direction
    was active," which is what the limits (0.35 / 0.3) were actually tuned
    against.
    """

    def __init__(self, warmup_steps: int = 30_000, verbose: int = 0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.rollout_costs = {k: [] for k in SOLVER.lambdas}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "costs" in info:
                active = info.get("cost_active", {})
                for key in self.rollout_costs:
                    # Only record this step's value for a key if that
                    # direction was actually active this step -- skip the
                    # force-zeroed inactive entries rather than averaging
                    # them in.
                    if active.get(key, True):
                        self.rollout_costs[key].append(info["costs"].get(key, 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps < self.warmup_steps:
            for v in self.rollout_costs.values():
                v.clear()
            return

        # Guard against any single direction having zero active samples in
        # a rollout (e.g. very early in training, or a short/degenerate
        # rollout) -- fall back to the previous lambda for that key rather
        # than corrupting its mean with an empty-list np.mean warning/NaN.
        mean_costs = {}
        have_any = False
        for key, values in self.rollout_costs.items():
            if values:
                mean_costs[key] = float(np.mean(values))
                have_any = True
            else:
                mean_costs[key] = SOLVER.limits[key]  # neutral: no update pressure this key

        if have_any:
            SOLVER.update_multipliers(mean_costs)
            self.training_env.env_method("set_lambdas", SOLVER.lambdas)

            for key, lam in SOLVER.lambdas.items():
                self.logger.record(f"lagrangian/lambda_{key}", lam)
            for key, val in mean_costs.items():
                self.logger.record(f"lagrangian/mean_{key}", val)
            # Short aliases for the diagnostic active-sample counts -- the
            # full "lagrangian/active_samples_cost_duty_rate_inflate" /
            # "..._exhaust" names truncate to an identical prefix under
            # SB3's console/CSV logger's max key length, which raises a
            # ValueError on the second write. Kept short and still unique.
            short_names = {
                "cost_overshoot_up": "n_overshoot_up",
                "cost_overshoot_down": "n_overshoot_down",
                "cost_duty_rate_inflate": "n_duty_inflate",
                "cost_duty_rate_exhaust": "n_duty_exhaust",
            }
            for key, values in self.rollout_costs.items():
                alias = short_names.get(key, key)
                self.logger.record(f"lagrangian/{alias}", len(values))

        for v in self.rollout_costs.values():
            v.clear()


def make_env(rank: int, seed: int = 0):
    def _init():
        env = PneumaticEnv()
        env = LagrangianWrapper(env)
        env.reset(seed=seed + rank)
        return env

    return _init


def main():
    print("=" * 60)
    print("      STEP 3.1: HIGH-THROUGHPUT CPU DRL TRAINING PIPELINE       ")
    print("      CONTINUOUS PWM DUTY-CYCLE ACTION SPACE                   ")
    print("=" * 60)

    TOTAL_TIMESTEPS = 4_000_000

    log_dir = "./logs/"
    models_dir = "./models/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    num_cpu = max(1, (os.cpu_count() or 4) - 1)
    print(f"\n[1/4] Launching {num_cpu} Parallel Physics Environments...")

    vec_env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(log_dir, "monitor.csv"))

    print("[2/4] Initializing Neural Network [64, 64] (Gaussian policy for continuous action)...")
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        device="cpu",
        verbose=1,
        tensorboard_log=log_dir,
    )

    warmup_steps = min(30_000, int(TOTAL_TIMESTEPS * 0.20))
    lagrangian_callback = LagrangianCallback(warmup_steps=warmup_steps)

    save_freq = max(1000, (TOTAL_TIMESTEPS // 5) // num_cpu)
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=models_dir,
        name_prefix="ppo_pneumatic_pwm",
        save_replay_buffer=False,
    )

    print(f"\n[3/4] Starting Optimized Training for {TOTAL_TIMESTEPS:,} steps...")
    print(f"      -> Warmup Steps: {warmup_steps:,}")
    print(f"      -> Checkpoint Frequency: Every {save_freq * num_cpu:,} steps")

    start_time = time.time()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, lagrangian_callback],
        progress_bar=True,
    )

    elapsed_time = time.time() - start_time
    print(f"\nTraining finished in {elapsed_time / 60:.2f} minutes! ({TOTAL_TIMESTEPS / elapsed_time:.0f} steps/sec)")

    final_model_path = os.path.join(models_dir, "ppo_pneumatic_pwm_final")
    model.save(final_model_path)
    print(f"[4/4] Saved final model to: {final_model_path}.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()