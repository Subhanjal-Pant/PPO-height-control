# Pneumatic Air Spring Height Control via Constrained RL

Closed-loop height control of a dual-plate pneumatic air spring test rig using
Proximal Policy Optimization (PPO) with Lagrangian-constrained costs, trained
against a physics-based simulator and matched bit-for-bit to the real ESP32
firmware's PWM valve-actuation logic.

The long-term goal is a direct comparison between the learned PPO policy and
classical controllers (PID, adaptive PID, ADRC) on tracking accuracy,
disturbance rejection, and up/down control asymmetry, targeting a Q1 journal
submission.

## Problem

The rig uses two binary ON/OFF solenoid valves — an inlet (S1, inflate) and
an exhaust (S2, deflate) — to regulate the height of a mass sitting on a
rolling-lobe air spring. There is no proportional valve or electronic
pressure regulator, so all control authority comes from timing when each
binary valve is open.

Two behaviors make this harder than a standard bang-bang problem:

- **PWM-quantized actuation.** The real firmware doesn't switch valves
  continuously — it samples a duty-cycle decision once per 40 ms control
  tick against a recurring 143 ms (~7 Hz) PWM cycle, matching the solenoids'
  rated switching frequency. The simulator reproduces this exactly, so a
  policy trained in sim sees the same coarse, phase-quantized actuation
  resolution it will see on hardware.
- **Persistent up/down asymmetry.** Upward transitions fight gravity while
  downward transitions are assisted by it, and the inflate path chokes
  across most of the operating range while the exhaust path is usually
  subcritical — giving inflate roughly 3x the control authority of exhaust.
  This shows up consistently in both hardware PID data and simulation, and
  is treated as a real physical effect to characterize and mitigate, not
  just a caveat to report.

## Repository contents

| File | Role |
|---|---|
| `pneumatics.py` | `PneumaticAirSpringPhysics` — the physics engine |
| `pneumatic_env.py` | `PneumaticAirSpringEnv` — Gymnasium wrapper around the physics engine |
| `train.py` | PPO training entry point with Lagrangian cost shaping |
| `train_lagrangian.py` | `LagrangianSolver` — dual gradient ascent for the cost multipliers |

## Physics engine (`pneumatics.py`)

A 7-state RK4-integrated non-isothermal thermodynamic and kinematic model of
the rig, not a simplified isothermal approximation:

- **State vector:** `[h, v, P_spring, T_spring, m_air, u_inlet_act, u_exhaust_act]`
- **Flow model:** compressible Saint-Venant orifice equations for both
  choked and subcritical regimes, applied independently to the inlet and
  exhaust paths (identical orifice areas — the up/down asymmetry is not a
  valve-sizing artifact).
- **Geometry:** effective piston area and chamber volume are looked up from
  a height/pressure lookup table via 2D cubic `RectBivariateSpline`
  interpolation (`LUT_effective_area.csv` is currently placeholder data
  pending real bench calibration).
- **Contact model:** a Hunt–Crossley non-linear viscoelastic model handles
  the hard stroke end-stops.
- **Actuator dynamics:** first-order valve lag (`tau_v`) between the
  commanded and actual open fraction of each valve.
- **Sensor delay:** a small `deque`-based buffer simulates the ~40 ms I2C /
  ToF sensor read latency observed on the real hardware, returning the
  oldest buffered observation rather than the true instantaneous state.
- **Sub-stepping:** the 40 ms control step is integrated internally at 1 ms
  resolution (`n_substeps=40`) for numerical stability of the fast pressure
  dynamics.

## Gym environment (`pneumatic_env.py`)

- **Action space:** a single continuous value in `[-1, 1]`. Sign selects
  direction (inflate / exhaust), magnitude selects duty cycle. Actuation is
  resolved through `_resolve_pwm_valve_state`, which reproduces the
  firmware's `runBinaryPulsing()` exactly: an action-deadzone of 0.05 (below
  which the valve holds closed), a minimum-duty floor of 0.20 once a
  direction is engaged, and a single phase-sampled open/closed decision per
  40 ms step against the recurring 143 ms PWM cycle.
- **Observation space (8-dim):** normalized height, velocity, pressure,
  tracking error, current binary valve states (S1, S2), the last commanded
  duty cycle, and supply pressure.
- **Episodes:** 250 steps (10 s) at 25 Hz.
- **Setpoint scheduling (training only):** rather than re-randomizing the
  target height on a fixed interval, the gap to the *next* jump is resampled
  after every jump — 65% of the time a "normal" 50–70 step gap that lets the
  plant settle, 35% of the time a short 15–35 step "chained" gap that forces
  a second large setpoint change before the previous one has settled. This
  deliberately exposes the policy to compounding transients that
  isolated-jump training under-represents.
- **Domain randomization (training only):** payload mass (27–33 kg, nominal
  30 kg), supply pressure (1.8–2.2 bar gauge, nominal 2.0 bar), valve time
  constant (10–15 ms), viscous and Coulomb friction, and inlet/exhaust
  discharge coefficients — plus a mid-episode mass disturbance (50%
  probability, 5–10 kg) to test disturbance rejection.
- **Reward:** `-12 * (normalized_error)^2 - 0.1 * (normalized_velocity)^2`,
  plus a +1.0 bonus while within the ±4 mm deadband (matched to the measured
  real-hardware PID deadband at 2 bar supply pressure).
- **Direction-split costs:** `cost_overshoot_up/down` and
  `cost_duty_rate_inflate/exhaust` are tracked separately rather than
  pooled, since inflate and exhaust operate in different flow regimes and
  pooling would mask that asymmetry from the dual solver. Each step also
  reports `cost_active`, flagging which member of each pair was actually
  "in play" that step (since only one direction can be active on a
  hold-free step) so downstream code can average conditionally instead of
  diluting the mean with placeholder zeros.

## Constrained RL training (`train.py`, `train_lagrangian.py`)

PPO (`stable-baselines3`, `MlpPolicy`, `[64, 64]` actor and critic — sized to
fit the target ESP32 deployment budget alongside sensor reads and Kalman
filtering) is trained with **Lagrangian-constrained cost shaping** rather
than a single hand-tuned penalty term:

- `LagrangianSolver` maintains one dual multiplier (λ) per cost key, updated
  via dual gradient ascent: λ increases when the rollout's mean cost exceeds
  its limit, decreases otherwise, clipped to `[0, max_lambda]`.
- Cost limits: `cost_overshoot_up = 0.35`, `cost_overshoot_down = 0.35`,
  `cost_duty_rate_inflate = 0.3`, `cost_duty_rate_exhaust = 0.5` — the
  higher exhaust limit reflecting its inherently weaker control authority.
- `LagrangianWrapper` (a per-worker `gym.Wrapper`) subtracts
  `Σ λ_key * cost_key` from the reward before PPO ever sees it.
- `LagrangianCallback` collects per-step costs across each rollout, filters
  each cost key's buffer to only the steps where `cost_active[key]` was
  true, and updates the multipliers at rollout end. Averaging over *all*
  steps (including force-zeroed inactive placeholders) was an earlier bug
  that roughly halved the reported mean cost and let policy chatter persist
  well past the intended duty-rate limit before the multiplier reacted
  strongly enough to suppress it.
- Training runs `SubprocVecEnv`-parallelized (one process per core minus
  one) for 4M timesteps, checkpointing periodically and logging to
  TensorBoard.

## Status

The environment and training pipeline are past two diagnosed and fixed
failure modes:

1. **Chattering policy** — caused by the duty-rate cost's learning rate
   being too low relative to the overshoot cost's, compounded by an
   entropy coefficient that kept policy variance plateaued. Fixed by
   raising `cost_duty_rate` learning rates, lowering `ent_coef`, and
   raising `max_lambda`.
2. **Non-settling chained transitions** — caused by the original fixed
   60-step setpoint re-randomization interval. Fixed by the resampled
   normal/chained interval scheme described above.

Current focus is running training to completion on the corrected pipeline
and evaluating tracking performance, before finalizing the sim-first paper
(simulator fidelity, domain-randomized robustness, disturbance rejection,
actuator-authority / PWM-quantization analysis, the compounding-transition
finding, and the up/down asymmetry finding). Real bench data collection is
not currently possible (no hardware/lab access), so the paper is designed to
stand on simulation alone, with the effective-area LUT still using
placeholder values pending future bench calibration.

## Requirements

```
gymnasium
numpy
pandas
scipy
torch
stable-baselines3
```

## Usage

```bash
python train.py
```

Trains PPO on `PneumaticAirSpringEnv` with Lagrangian cost shaping,
checkpointing to `./models/` and logging to `./logs/` (TensorBoard-readable).
The final policy is saved to `./models/ppo_pneumatic_pwm_final.zip`.
