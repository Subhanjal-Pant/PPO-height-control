import matplotlib.pyplot as plt
import numpy as np
from pneumatic_env import PneumaticAirSpringEnv
from stable_baselines3 import PPO

DEADBAND = 0.004
DEADBAND_MM = DEADBAND * 1000.0  # heights_arr/targets_arr are in mm, DEADBAND is defined in meters


def run_evaluation(inject_disturbance: bool = False, disturbance_step: int = 150,
                    disturbance_delta_kg: float = 8.0):
    model = PPO.load("./models/ppo_pneumatic_pwm_final.zip")

    env = PneumaticAirSpringEnv(eval_mode=True)

    total_steps = 250
    dt = env.dt
    time_axis = np.arange(total_steps) * dt

    step_references = np.full(total_steps, 0.185)
    step_references[50:125] = 0.195
    step_references[125:200] = 0.165

    heights, pressures, s1_states, s2_states, targets, masses, duty_cmds = [], [], [], [], [], [], []

    obs, _ = env.reset()

    for i in range(total_steps):
        env.target_height = step_references[i]

        if inject_disturbance and i == disturbance_step:
            new_mass = env.sim.m_payload + disturbance_delta_kg
            env.apply_mass_disturbance(new_mass)
            print(f"[t={i*dt:.2f}s] Injected disturbance: mass -> {new_mass:.1f} kg")

        obs = env._get_obs()

        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)

        heights.append(env._last_height * 1000.0)
        pressures.append(env._last_pressure / 1000.0)
        s1_states.append(env.current_valve_state[0])
        s2_states.append(env.current_valve_state[1])
        targets.append(step_references[i] * 1000.0)
        masses.append(env.sim.m_payload)
        duty_cmds.append(info["duty_cmd"])

    s1_arr = np.array(s1_states)
    s2_arr = np.array(s2_states)
    heights_arr = np.array(heights)
    targets_arr = np.array(targets)

    mode_switches = int(np.sum(np.abs(np.diff(s1_arr)) + np.abs(np.diff(s2_arr))))
    simultaneous_fires = int(np.sum((s1_arr == 1.0) & (s2_arr == 1.0)))
    within_deadband = float(np.mean(np.abs(heights_arr - targets_arr) <= DEADBAND_MM))
    mean_abs_duty = float(np.mean(np.abs(duty_cmds)))

    print(f"Total valve open/close transitions over 10s: {mode_switches}")
    print(f"Simultaneous S1+S2 fires (should be 0): {simultaneous_fires}")
    print(f"Fraction of steps within +/-{DEADBAND_MM:.1f}mm deadband: {within_deadband:.2%}")
    print(f"Mean |duty command|: {mean_abs_duty:.3f}")

    if inject_disturbance:
        post = heights_arr[disturbance_step:] - targets_arr[disturbance_step:]
        peak_dev_mm = float(np.max(np.abs(post)))
        settled_idx = None
        for k in range(len(post)):
            if np.all(np.abs(post[k:]) <= DEADBAND_MM):
                settled_idx = k
                break
        settle_time_s = settled_idx * dt if settled_idx is not None else None
        print(f"Post-disturbance peak deviation: {peak_dev_mm:.2f} mm")
        print(
            f"Post-disturbance settle time to +/-{DEADBAND_MM:.1f}mm: "
            f"{settle_time_s:.2f}s" if settle_time_s is not None else "did not settle within episode"
        )

    n_rows = 5 if inject_disturbance else 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 12 if inject_disturbance else 10), sharex=True)
    ax1, ax2, ax3, ax4 = axes[0], axes[1], axes[2], axes[3]
    fig.suptitle(
        "PPO Closed-Loop Pneumatic Height Control Evaluation (PWM Duty Control)",
        fontsize=14,
        fontweight="bold",
    )

    ax1.plot(time_axis, targets, "r--", label="Target Ref h_ref(t)")
    ax1.plot(time_axis, heights, "b-", label="PPO Response h(t)")
    if inject_disturbance:
        ax1.axvline(disturbance_step * dt, color="orange", linestyle=":", label="Disturbance")
    ax1.set_ylabel("Height [mm]")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(time_axis, duty_cmds, "purple", label="Commanded Duty (signed)")
    ax2.axhline(0.0, color="gray", linewidth=0.8)
    ax2.set_ylabel("Duty Cmd [-1, 1]")
    ax2.set_ylim([-1.1, 1.1])
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    ax3.step(time_axis, s1_states, "g-", label="Inlet S1 (resultant)", where="post")
    ax3.step(time_axis, s2_states, "m-", label="Exhaust S2 (resultant)", where="post")
    ax3.set_ylabel("Binary State {0, 1}")
    ax3.set_ylim([-0.1, 1.1])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right")

    ax4.plot(time_axis, pressures, "k-", label="Chamber Pressure P(t)")
    ax4.set_ylabel("Pressure [kPa]")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right")

    if inject_disturbance:
        ax5 = axes[4]
        ax5.plot(time_axis, masses, "c-", label="Payload Mass [kg]")
        ax5.set_ylabel("Mass [kg]")
        ax5.set_xlabel("Time [s]")
        ax5.grid(True, linestyle=":", alpha=0.6)
        ax5.legend(loc="upper right")
    else:
        ax4.set_xlabel("Time [s]")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_evaluation(inject_disturbance=False)