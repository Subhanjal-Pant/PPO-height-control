import gymnasium as gym
from gymnasium import spaces
import numpy as np

from pneumatics import PneumaticAirSpringPhysics


def gauge_bar_to_abs_pa(gauge_bar: float, p_atm: float = 101325.0) -> float:
    """Converts a gauge pressure in bar (what a real compressor/tank gauge
    reads) into absolute Pascals (what the physics simulator's P_supply and
    P_atm-relative math expects)."""
    return gauge_bar * 1e5 + p_atm


class PneumaticAirSpringEnv(gym.Env):
    """Custom Gymnasium Environment for Pneumatic Air Spring Height Control,
    using PWM duty-cycle valve actuation matching the real ESP32 firmware:

      - Control loop: 25 Hz (dt = 40ms), matching LOOP_INTERVAL.
      - PWM base period: 143ms (~7 Hz), matching CYCLE_TIME / solenoid rating.
      - The valve-open decision is sampled ONCE per 40ms control step, based
        on where that instant falls within the recurring 143ms PWM cycle --
        exactly mirroring runBinaryPulsing()'s single evaluation per loop().
        It is NOT re-evaluated continuously within the 40ms window, so the
        real achievable duty resolution is coarser than the continuous
        action suggests (~3-4 effective phase samples per 143ms cycle).

    Action space is continuous: a single value in [-1, 1] where sign selects
    direction (positive = Inflate/S1, negative = Deflate/S2) and magnitude
    selects duty cycle. A minimum duty of 0.20 (20%) applies once a
    direction is engaged, matching the firmware's
    `dutyCycle = constrain(abs(rawOutput), 20, 100)` -- below the deadzone
    threshold the valve holds fully closed for that step.

    Setpoint scheduling (training only): rather than re-randomizing the
    target at a fixed 60-step interval, the interval to the NEXT jump is
    resampled after every jump. With probability CHAIN_JUMP_PROB, a SHORT
    interval is chosen (CHAIN_INTERVAL_RANGE), deliberately producing a
    second large setpoint change before the plant has settled from the
    previous one.

    Direction-decomposed costs: cost_duty_rate and cost_overshoot are each
    split into an inflate/exhaust (or up/down) pair, reported every step
    via `info["costs"]`. Because inflate and exhaust never fire on the same
    step, one member of each pair is always force-zeroed on any given step
    -- that zero is a real "not this step" placeholder, not a "measured
    zero jerk/overshoot". `info["cost_active"]` reports, per cost key,
    whether THIS step was actually in that key's regime, so the training
    side can average each key only over the steps where it applied instead
    of diluting the mean with irrelevant zeros. See train.py's
    LagrangianCallback for the consumer of this flag.
    """

    metadata = {"render_modes": []}

    # --- PHYSICAL NORMALIZATION BOUNDS (Bryson's Rule Scale Constants) ---
    E_MAX = 0.050      # 50 mm maximum tracking error bound (m)
    V_MAX = 0.500      # 0.5 m/s maximum physical velocity limit (m/s)
    # 4mm matches measured real-hardware PID deadband at 2 bar (see project
    # notes); replaces the old 1.5mm aspirational placeholder.
    DEADBAND = 0.004  # 4 mm hysteresis tolerance (m)

    # --- PWM CONFIGURATION (matches firmware CYCLE_TIME / solenoid rating) ---
    PWM_CYCLE_TIME = 0.143     # s, ~7 Hz base period
    MIN_DUTY = 0.20            # minimum effective duty once direction is engaged
    ACTION_DEADZONE = 0.05     # |action| below this => Hold, valve fully closed

    # --- SETPOINT JUMP SCHEDULING (training only) ---
    NORMAL_INTERVAL_RANGE = (50, 70)   # steps between jumps in the "normal" case (~2-2.8s)
    CHAIN_INTERVAL_RANGE = (15, 35)    # steps between jumps in the "chained" case (~0.6-1.4s)
    CHAIN_JUMP_PROB = 0.35             # probability the NEXT interval is a short/chained one

    def __init__(self, render_mode=None, eval_mode=False):
        super().__init__()

        self.eval_mode = eval_mode

        # --- 1. Timing Specifications ---
        self.dt = 0.040  # 25 Hz control loop (ESP32 step time)

        # --- 2. Action Space Definition (continuous PWM duty command) ---
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # --- 3. Observation Space Definition ---
        low_obs = np.array(
            [0.140, -0.5, 101325.0, -1.0, 0.0, 0.0, -1.0, 200000.0],
            dtype=np.float32,
        )
        high_obs = np.array(
            [0.230, 0.5, 600000.0, 1.0, 1.0, 1.0, 1.0, 500000.0],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        # --- 4. Physics Engine Initialization ---
        self.nominal_mass = 30.0  # kg
        self.nominal_p_supply = gauge_bar_to_abs_pa(2.0)  # fixed 2 bar gauge operating point
        self.nominal_tau_v = 0.012  # s (12 ms valve lag)

        self.sim = PneumaticAirSpringPhysics(
            dt=self.dt,
            mass=self.nominal_mass,
            p_supply=self.nominal_p_supply,
            tau_v=self.nominal_tau_v,
        )

        self.target_height = 0.185  # 18.5 cm default target

        # --- 5. State Tracking Variables ---
        self.current_valve_state = np.array([0.0, 0.0])  # [S1, S2] resultant binary state this step
        self.current_step = 0
        self.max_steps = 250  # 10 seconds per episode
        self.last_duty_cmd = 0.0  # signed duty command from the previous step, for smoothness cost

        # --- 5b. Setpoint Jump Scheduling State (training only) ---
        self.next_jump_step = -1  # step index at which the next target re-randomization fires

        # --- 6. Mid-Episode Disturbance Configuration ---
        self.disturbance_enabled = True
        self.disturbance_prob = 0.5
        self.disturbance_step = -1
        self.disturbance_mass_range = (5.0, 10.0)

        # Cache state
        self._last_height = 0.185
        self._last_velocity = 0.0
        self._last_pressure = 101325.0
        self._current_p_supply = self.nominal_p_supply

    def _sample_next_jump_interval(self) -> int:
        """Chooses how many steps until the NEXT setpoint jump. Most of the
        time this is a 'normal' interval, giving the plant time to settle.
        With CHAIN_JUMP_PROB, a short interval is chosen instead, so a
        second big jump lands while the plant is still recovering from the
        previous one -- deliberately exposing the policy to the compounding-
        transient pattern that isolated-jump training under-represents."""
        if self.np_random.random() < self.CHAIN_JUMP_PROB:
            lo, hi = self.CHAIN_INTERVAL_RANGE
        else:
            lo, hi = self.NORMAL_INTERVAL_RANGE
        return int(self.np_random.integers(lo, hi + 1))

    def _get_obs(self):
        """Constructs 8-dimensional normalized observation vector."""
        if self.eval_mode:
            noisy_height = self._last_height
            noisy_velocity = self._last_velocity
            noisy_pressure = self._last_pressure
            noisy_p_supply = self._current_p_supply
        else:
            noisy_height = self._last_height + float(self.np_random.normal(0, 0.0005))
            noisy_velocity = self._last_velocity + float(self.np_random.normal(0, 0.005))
            noisy_pressure = self._last_pressure + float(self.np_random.normal(0, 250.0))
            noisy_p_supply = self._current_p_supply + float(self.np_random.normal(0, 500.0))

        error = self.target_height - noisy_height

        norm_h = (noisy_height - 0.140) / (0.230 - 0.140)
        norm_v = noisy_velocity / self.V_MAX
        norm_p = (noisy_pressure - 101325.0) / (600000.0 - 101325.0)
        norm_err = error / self.E_MAX
        v0 = float(self.current_valve_state[0])
        v1 = float(self.current_valve_state[1])
        norm_last_duty = float(np.clip(self.last_duty_cmd, -1.0, 1.0))
        norm_p_supply = (noisy_p_supply - 200000.0) / (500000.0 - 200000.0)

        obs = np.array(
            [
                np.clip(norm_h, 0.0, 1.0),
                np.clip(norm_v, -1.0, 1.0),
                np.clip(norm_p, 0.0, 1.0),
                np.clip(norm_err, -1.0, 1.0),
                v0,
                v1,
                norm_last_duty,
                np.clip(norm_p_supply, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.last_duty_cmd = 0.0

        sim_seed = int(self.np_random.integers(0, 2**31 - 1))
        if hasattr(self.sim, "seed"):
            self.sim.seed(sim_seed)

        if not self.eval_mode:
            rand_mass = float(self.np_random.uniform(27.0, 33.0))
            rand_p_supply = float(
                self.np_random.uniform(gauge_bar_to_abs_pa(1.8), gauge_bar_to_abs_pa(2.2))
            )
            rand_tau_v = float(self.np_random.uniform(0.010, 0.015))
            rand_viscous = float(self.np_random.uniform(35.0, 55.0))
            rand_f_coulomb = float(self.np_random.uniform(6.0, 10.0))
            rand_cd_inlet = float(self.np_random.uniform(0.40, 0.50))
            rand_cd_exhaust = float(self.np_random.uniform(0.53, 0.63))

            self.target_height = float(self.np_random.uniform(0.155, 0.205))
            start_height = float(self.np_random.uniform(0.160, 0.175))

            # Schedule the first setpoint jump of this episode.
            self.next_jump_step = self._sample_next_jump_interval()

            if self.disturbance_enabled and self.np_random.random() < self.disturbance_prob:
                self.disturbance_step = int(self.np_random.integers(30, self.max_steps - 20))
            else:
                self.disturbance_step = -1
        else:
            rand_mass = self.nominal_mass
            rand_p_supply = self.nominal_p_supply
            rand_tau_v = self.nominal_tau_v
            rand_viscous = 45.0
            rand_f_coulomb = 8.0
            rand_cd_inlet = 0.45
            rand_cd_exhaust = 0.58

            start_height = 0.1675
            self.target_height = 0.185
            self.disturbance_step = -1
            self.next_jump_step = -1  # eval sequences set target_height externally

        if hasattr(self.sim, "set_randomized_parameters"):
            self.sim.set_randomized_parameters(
                mass=rand_mass,
                p_supply=rand_p_supply,
                tau_v=rand_tau_v,
                viscous=rand_viscous,
                f_coulomb=rand_f_coulomb,
                Cd_inlet=rand_cd_inlet,
                Cd_exhaust=rand_cd_exhaust,
            )

        sim_obs = self.sim.reset(current_height=start_height)

        self.current_valve_state = np.array([0.0, 0.0])
        self.current_step = 0

        self._current_p_supply = rand_p_supply
        self._last_height = sim_obs["height"]
        self._last_velocity = sim_obs["velocity"]
        self._last_pressure = sim_obs["P_spring_abs"]

        obs = self._get_obs()
        info = {}

        return obs, info

    def apply_mass_disturbance(self, new_mass: float) -> None:
        """Manually injects a payload mass change mid-episode (eval/testing)."""
        self.sim.m_payload = float(new_mass)

    def _resolve_pwm_valve_state(self, duty_cmd: float) -> np.ndarray:
        """Given a signed duty command in [-1, 1], reproduces the firmware's
        runBinaryPulsing() logic: determine which valve direction is
        requested, apply the minimum-duty floor, then sample whether the
        valve is open at THIS instant based on phase within the recurring
        143ms PWM cycle. Returns the resultant binary [S1, S2] state for
        this control step (held constant through the physics substeps,
        exactly as the real firmware holds its digitalWrite() output for
        the full 40ms until the next loop() tick)."""

        if abs(duty_cmd) < self.ACTION_DEADZONE:
            return np.array([0.0, 0.0])

        # Apply minimum duty floor once a direction is engaged (firmware:
        # dutyCycle = constrain(abs(rawOutput), 20, 100))
        duty_fraction = np.clip(abs(duty_cmd), self.MIN_DUTY, 1.0)

        # Phase within the recurring PWM cycle, using absolute elapsed sim
        # time (matches firmware's `millis() % CYCLE_TIME`)
        elapsed_time = self.current_step * self.dt
        phase = elapsed_time % self.PWM_CYCLE_TIME
        on_time = duty_fraction * self.PWM_CYCLE_TIME
        valve_open = phase < on_time

        if not valve_open:
            return np.array([0.0, 0.0])

        if duty_cmd > 0.0:
            return np.array([1.0, 0.0])  # Inflate (S1)
        else:
            return np.array([0.0, 1.0])  # Deflate (S2)

    def step(self, action):
        self.current_step += 1

        # 1. Extract scalar duty command from action
        duty_cmd = float(np.clip(action[0] if hasattr(action, "__len__") else action, -1.0, 1.0))

        # 2. Setpoint jump scheduling (training only).
        if not self.eval_mode and self.current_step == self.next_jump_step:
            self.target_height = float(self.np_random.uniform(0.155, 0.205))
            self.next_jump_step = self.current_step + self._sample_next_jump_interval()

        # 2b. Fire scheduled mid-episode mass disturbance (training only)
        if not self.eval_mode and self.current_step == self.disturbance_step:
            delta = float(self.np_random.uniform(*self.disturbance_mass_range))
            sign = 1.0 if self.np_random.random() < 0.5 else -1.0
            self.sim.m_payload = max(5.0, self.sim.m_payload + sign * delta)

        prev_duty_cmd = self.last_duty_cmd
        self.last_duty_cmd = duty_cmd

        # 3. Resolve PWM phase into this step's actual binary valve state,
        # exactly matching how the firmware samples runBinaryPulsing() once
        # per 40ms loop tick.
        self.current_valve_state = self._resolve_pwm_valve_state(duty_cmd)

        # 4. Step Physics Simulation (binary state held constant across substeps)
        sim_obs = self.sim.step(
            int(self.current_valve_state[0]), int(self.current_valve_state[1])
        )

        raw_height = sim_obs["height"]
        raw_velocity = sim_obs["velocity"]

        self._last_height = raw_height
        self._last_velocity = raw_velocity
        self._last_pressure = sim_obs["P_spring_abs"]

        obs = self._get_obs()

        # 5. Reward
        raw_error = self.target_height - raw_height
        abs_error = abs(raw_error)

        norm_e = abs_error / self.E_MAX
        norm_v = raw_velocity / self.V_MAX

        r_track = -12.0 * (norm_e**2) - 0.1 * (norm_v**2)

        if abs_error <= self.DEADBAND:
            r_track += 1.0

        reward = float(r_track)

        # 6. Physical Boundary Penalties
        terminated = False
        min_limit = getattr(self.sim, "min_height", 0.145) - 0.005
        max_limit = getattr(self.sim, "max_height", 0.225) + 0.005

        if raw_height <= min_limit or raw_height >= max_limit:
            reward -= 100.0
            terminated = True

        truncated = self.current_step >= self.max_steps

        # 7. Cost Metrics for Lagrangian Dual Updates
        # Split by direction so each gets its own Lagrange multiplier --
        # inflate and exhaust operate in different flow regimes (choked vs
        # subcritical), so pooling their duty-rate cost into one signal
        # masks that asymmetry from the dual solver. Same idea for
        # overshoot: settling behavior differs depending on whether the
        # transition needs height to increase or decrease.
        #
        # IMPORTANT: on any given step, duty_cmd has ONE sign (or is zero),
        # so exactly one member of each pair below is a real measurement
        # and the other is force-zeroed as a placeholder -- it does NOT
        # mean "zero jerk" or "zero overshoot" for that direction on this
        # step, it means "this direction wasn't in play this step". See
        # cost_active below, which flags which member is the real one.
        duty_rate = abs(duty_cmd - prev_duty_cmd)
        overshoot_occurred = 1.0 if abs(raw_height - self.target_height) > self.DEADBAND else 0.0

        if duty_cmd > 0.0:
            cost_duty_rate_inflate = duty_rate
            cost_duty_rate_exhaust = 0.0
        elif duty_cmd < 0.0:
            cost_duty_rate_inflate = 0.0
            cost_duty_rate_exhaust = duty_rate
        else:
            cost_duty_rate_inflate = 0.0
            cost_duty_rate_exhaust = 0.0

        if raw_error > 0.0:      # below target -> this transition needs upward correction
            cost_overshoot_up = overshoot_occurred
            cost_overshoot_down = 0.0
        else:                    # at/above target -> this transition needs downward correction
            cost_overshoot_up = 0.0
            cost_overshoot_down = overshoot_occurred

        costs = {
            "cost_overshoot_up": cost_overshoot_up,
            "cost_overshoot_down": cost_overshoot_down,
            "cost_duty_rate_inflate": cost_duty_rate_inflate,
            "cost_duty_rate_exhaust": cost_duty_rate_exhaust,
        }

        # Per-key flag: was THIS step actually in that cost key's regime?
        # duty_cmd == 0.0 (hold) counts as inactive for BOTH duty_rate keys
        # -- there's no meaningful inflate-vs-exhaust duty-rate measurement
        # to attribute on a hold step, so it's correctly excluded from both
        # rather than force-included in either.
        cost_active = {
            "cost_overshoot_up": raw_error > 0.0,
            "cost_overshoot_down": raw_error <= 0.0,
            "cost_duty_rate_inflate": duty_cmd > 0.0,
            "cost_duty_rate_exhaust": duty_cmd < 0.0,
        }

        info = {
            "duty_cmd": duty_cmd,
            "applied_valves": self.current_valve_state.tolist(),
            "disturbance_fired": self.current_step == self.disturbance_step,
            "current_mass": self.sim.m_payload,
            "costs": costs,
            "cost_active": cost_active,
        }

        return obs, reward, terminated, truncated, info