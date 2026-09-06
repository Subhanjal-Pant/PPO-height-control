from collections import deque
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RectBivariateSpline


class PneumaticAirSpringPhysics:
    """Peer-Review Grade Thermodynamic & Kinetic Simulator for a

    Rolling-Lobe Air Spring Rig with Binary Solenoid Valve Control.
    """

    def __init__(
        self,
        dt: float = 0.040,  # RL Agent Action Interval (25 Hz)
        mass: float = 30.0,
        p_supply: float = 300000.0,
        tau_v: float = 0.012,
        viscous: float = 45.0,
        f_coulomb: float = 8.0,
        Cd_inlet: float = 0.65,
        Cd_exhaust: float = 0.58,
        T_amb: float = 293.15,
        n_substeps: int = 40,  # Internal physics sub-stepping (dt_physics = 1 ms)
        csv_filepath: str = "LUT_effective_area.csv", # Dummy data for now
        enable_randomization: bool = False,
        delay_steps: int = 1,  # Assumed Sensor & I2C hardware observation delay (1 step = 40ms)
    ):
        self.dt_control = dt
        self.n_substeps = max(1, n_substeps)
        self.dt_physics = self.dt_control / self.n_substeps
        self.enable_randomization = enable_randomization

        # ADDED: Latency / Hardware Delay Queue
        self.delay_steps = delay_steps
        self.obs_buffer = deque(maxlen=self.delay_steps + 1)

        # --- Base Physical & Gas Constants ---
        self.R = 287.05  # Gas constant for dry air [J/(kg*K)]
        self.gamma = 1.4  # Specific heat ratio
        self.c_v = self.R / (self.gamma - 1.0)
        self.c_p = self.gamma * self.c_v
        self.g = 9.81  # Acceleration due to gravity [m/s^2]

        # --- Nominal Atmospheric & Thermal Parameters ---
        self.T_env_nom = 293.15  # Nominal ambient temperature [K]
        self.P_atm_nom = 101325.0  # Nominal atmospheric pressure [Pa abs]

        self.h_c_nom = 25.0  # Convective heat transfer coefficient [W/(m^2*K)]
        self.A_surf_est = 0.08  # Effective heat transfer area [m^2]

        # --- Kinematic Limits ---
        self.min_height = 0.145  # Minimum stroke bound [m] (145 mm)
        self.max_height = 0.225  # Maximum stroke bound [m] (225 mm)
        self.height_nom = (self.min_height + self.max_height) / 2.0
        self.V_dead = 0.0002  # Dead volume [m^3]

        # --- Valve Flow & Geometry ---
        self.A_valve_inlet = 1.5e-5  # [m^2]
        self.A_valve_exhaust = 1.5e-5  # [m^2]
        self.b_crit = 0.528  # Choked flow critical pressure ratio

        # --- Friction & Contact Model Parameters ---
        self.v_ref_friction = 1e-3  # Velocity scaling for tanh friction [m/s]
        self.k_stop = 2.5e6  # Contact stiffness [N/m]
        self.p_stop = 1.5  # Non-linear contact exponent
        self.b_stop = 1.0e2  # Viscous dissipation coefficient [s/m]

        # --- Load LUT Surface Interpolators ---
        self.load_LUT_splines(csv_filepath)

        # --- Assign Active Parameter State ---
        self.m_payload = mass
        self.P_supply = p_supply
        self.tau_valve = tau_v  # FIXED: Correctly assign valve response time
        self.P_atm = self.P_atm_nom
        self.T_env = T_amb
        self.h_c = self.h_c_nom
        self.C_d_inlet = Cd_inlet
        self.C_d_exhaust = Cd_exhaust
        self.viscous = viscous
        self.F_coulomb = f_coulomb

        # Continuous State Vector y = [h, v, P_spring, T_spring, m_air, u_inlet_act, u_exhaust_act]
        self.y = np.zeros(7, dtype=np.float64)
        self.reset()

    def set_randomized_parameters(
        self,
        mass: float,
        p_supply: float,
        tau_v: float,
        viscous: float,
        f_coulomb: float,
        Cd_inlet: float,
        Cd_exhaust: float,
    ) -> None:
        """Dynamically update physical parameters from Gym Environment during reset."""
        self.m_payload = mass
        self.P_supply = p_supply
        self.tau_valve = tau_v
        self.viscous = viscous
        self.F_coulomb = f_coulomb
        self.C_d_inlet = Cd_inlet
        self.C_d_exhaust = Cd_exhaust

    def load_LUT_splines(self, filepath: str) -> None:
        """Loads CSV effective area lookup data and generates 2D cubic splines."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"LUT file missing: {filepath}")

        df = pd.read_csv(filepath, index_col=0)
        df.columns = df.columns.astype(float)

        self.lut_heights = df.index.to_numpy(dtype=float)
        self.lut_pressures = df.columns.to_numpy(dtype=float)
        A_eff_matrix = df.to_numpy(dtype=float)

        V_integrated = cumulative_trapezoid(
            A_eff_matrix, x=self.lut_heights, axis=0, initial=0.0
        )
        V_matrix = self.V_dead + V_integrated

        self.spline_A_eff = RectBivariateSpline(
            self.lut_heights, self.lut_pressures, A_eff_matrix, kx=3, ky=3
        )
        self.spline_V = RectBivariateSpline(
            self.lut_heights, self.lut_pressures, V_matrix, kx=3, ky=3
        )

    def get_geometry(self, h: float, P_abs: float) -> Tuple[float, float, float, float]:
        """Queries 2D splines with boundary clamping."""
        P_gauge = max(0.0, P_abs - self.P_atm)

        h_clamp = float(np.clip(h, self.lut_heights.min(), self.lut_heights.max()))
        P_clamp = float(np.clip(P_gauge, self.lut_pressures.min(), self.lut_pressures.max()))

        A_eff = float(self.spline_A_eff(h_clamp, P_clamp, grid=False))
        V_chamber = float(self.spline_V(h_clamp, P_clamp, grid=False))
        dA_dh = float(self.spline_A_eff(h_clamp, P_clamp, dx=1, grid=False))
        dV_dP = float(self.spline_V(h_clamp, P_clamp, dy=1, grid=False))

        dV_dP = max(0.0, dV_dP)
        return max(1e-5, A_eff), max(self.V_dead, V_chamber), dA_dh, dV_dP

    def compute_mass_flow(
        self, P_high: float, P_low: float, T_high: float, C_d: float, A_valve: float
    ) -> float:
        """Compressible Saint-Venant orifice mass flow rate calculation."""
        if P_high <= P_low or P_high <= 0.0 or T_high <= 0.0:
            return 0.0

        pr = P_low / P_high

        if pr <= self.b_crit:
            term = (2.0 / (self.gamma + 1.0)) ** (
                (self.gamma + 1.0) / (self.gamma - 1.0)
            )
            return (
                C_d
                * A_valve
                * P_high
                * np.sqrt((self.gamma / (self.R * T_high)) * term)
            )
        else:
            factor = (2.0 * self.gamma) / (self.R * T_high * (self.gamma - 1.0))
            bracket = (pr ** (2.0 / self.gamma)) - (
                pr ** ((self.gamma + 1.0) / self.gamma)
            )
            return (
                C_d
                * A_valve
                * P_high
                * np.sqrt(max(0.0, factor * bracket))
            )

    def compute_endstop_force(self, h: float, v: float) -> float:
        """Hunt-Crossley non-linear viscoelastic contact model for hard stroke limits."""
        F_stop = 0.0

        if h < self.min_height:
            x_pen = self.min_height - h
            v_pen = -v
            damping_factor = max(0.0, 1.0 + self.b_stop * v_pen)
            F_stop = (self.k_stop * (x_pen**self.p_stop)) * damping_factor
        elif h > self.max_height:
            x_pen = h - self.max_height
            v_pen = v
            damping_factor = max(0.0, 1.0 + self.b_stop * v_pen)
            F_stop = -(self.k_stop * (x_pen**self.p_stop)) * damping_factor

        return F_stop

    def _ode_rhs(
        self, y: np.ndarray, u_cmd_in: float, u_cmd_out: float
    ) -> np.ndarray:
        """System of Non-linear Differential Equations governing rig dynamics."""
        h, v, P, T, m, u_act_in, u_act_out = y

        P = max(1e3, P)
        T = max(100.0, T)
        m = max(1e-6, m)

        A_eff, V_chamber, _, dV_dP = self.get_geometry(h, P)

        mdot_in = (
            self.compute_mass_flow(
                self.P_supply, P, self.T_env, self.C_d_inlet, self.A_valve_inlet
            )
            * u_act_in
        )
        mdot_out = (
            self.compute_mass_flow(
                P, self.P_atm, T, self.C_d_exhaust, self.A_valve_exhaust
            )
            * u_act_out
        )
        mdot_net = mdot_in - mdot_out

        Q_dot = self.h_c * self.A_surf_est * (self.T_env - T)

        rhs_no_V = (
            self.R * mdot_in * self.T_env
            - self.R * mdot_out * T
            + (self.gamma - 1.0) * Q_dot / self.gamma
        )

        denom = 1.0 + (self.gamma * P / V_chamber) * dV_dP
        denom = max(1e-8, denom)

        dP_dt = (
            (self.gamma / V_chamber) * (rhs_no_V - P * A_eff * v)
        ) / denom

        dV_dt = A_eff * v + dV_dP * dP_dt
        dT_dt = (T / P) * dP_dt + (T / V_chamber) * dV_dt - (T / m) * mdot_net

        F_friction = self.viscous * v + self.F_coulomb * np.tanh(v / self.v_ref_friction)
        F_pneumatic = (P - self.P_atm) * A_eff
        F_gravity = self.m_payload * self.g
        F_endstop = self.compute_endstop_force(h, v)

        F_net = F_pneumatic - F_gravity - F_friction + F_endstop
        dv_dt = F_net / self.m_payload

        tau_v = max(1e-5, self.tau_valve)
        du_in_dt = (u_cmd_in - u_act_in) / tau_v
        du_out_dt = (u_cmd_out - u_act_out) / tau_v

        return np.array(
            [v, dv_dt, dP_dt, dT_dt, mdot_net, du_in_dt, du_out_dt],
            dtype=np.float64,
        )

    def rk4_step(self, u_cmd_in: float, u_cmd_out: float) -> None:
        """4th-Order Runge-Kutta integration step."""
        dt = self.dt_physics
        y = self.y

        k1 = self._ode_rhs(y, u_cmd_in, u_cmd_out)
        k2 = self._ode_rhs(y + 0.5 * dt * k1, u_cmd_in, u_cmd_out)
        k3 = self._ode_rhs(y + 0.5 * dt * k2, u_cmd_in, u_cmd_out)
        k4 = self._ode_rhs(y + dt * k3, u_cmd_in, u_cmd_out)

        self.y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def step(self, action_s1: int, action_s2: int) -> Dict[str, float]:
        """Steps the environment forward by dt_control through sub-sampled physics."""
        u_cmd_in = float(np.clip(action_s1, 0, 1))
        u_cmd_out = float(np.clip(action_s2, 0, 1))

        for _ in range(self.n_substeps):
            self.rk4_step(u_cmd_in, u_cmd_out)

        h, v, P, T, m, u_act_in, u_act_out = self.y
        A_eff, V_chamber, _, _ = self.get_geometry(h, P)
        
        # --- ADDED: Push fresh physical state into delay queue ---
        current_obs = {
            "height": float(h),
            "velocity": float(v),
            "P_spring_abs": float(P),
            "P_spring_gauge": float(P - self.P_atm),
            "T_spring": float(T),
            "m_air": float(m),
            "A_eff": float(A_eff),
            "V_chamber": float(V_chamber),
            "u_act_inlet": float(u_act_in),
            "u_act_exhaust": float(u_act_out),
            "mass_constraint_residual": float(m - (P * V_chamber) / (self.R * T)),
        }
        self.obs_buffer.append(current_obs)

        # --- ADDED: Return oldest observation in queue to simulate sensor lag ---
        return self.obs_buffer[0]

    def reset(
        self,
        current_height: Optional[float] = None,
        payload_mass: Optional[float] = None,
        equilibrium_tol: float = 1e-3,
        equilibrium_max_iter: int = 50,
    ) -> Dict[str, float]:
        """Resets simulator state with current parameters."""

        self.P_atm = self.P_atm_nom
        self.T_env = self.T_env_nom

        h_init = current_height if current_height is not None else self.height_nom
        h_init = float(np.clip(h_init, self.min_height, self.max_height))

        # Dynamically sample initial area at zero gauge pressure for safe fixed-point iteration seed
        A_eff_init, _, _, _ = self.get_geometry(h_init, self.P_atm)
        P_gauge_approx = (self.m_payload * self.g) / A_eff_init

        for _ in range(equilibrium_max_iter):
            P_abs_temp = self.P_atm + P_gauge_approx
            A_eff_temp, _, _, _ = self.get_geometry(h_init, P_abs_temp)
            P_gauge_new = (self.m_payload * self.g) / A_eff_temp
            if abs(P_gauge_new - P_gauge_approx) < equilibrium_tol:
                P_gauge_approx = P_gauge_new
                break
            P_gauge_approx = P_gauge_new

        P_init = self.P_atm + P_gauge_approx
        T_init = self.T_env
        _, V_init, _, _ = self.get_geometry(h_init, P_init)
        m_init = (P_init * V_init) / (self.R * T_init)

        self.y = np.array(
            [h_init, 0.0, P_init, T_init, m_init, 0.0, 0.0], dtype=np.float64
        )

      
        initial_obs = {
            "height": float(h_init),
            "velocity": 0.0,
            "P_spring_abs": float(P_init),
            "P_spring_gauge": float(P_gauge_approx),
            "T_spring": float(T_init),
            "m_air": float(m_init),
            "A_eff": float(A_eff_init),
            "V_chamber": float(V_init),
            "u_act_inlet": 0.0,
            "u_act_exhaust": 0.0,
            "mass_constraint_residual": 0.0,
        }

        self.obs_buffer.clear()
        for _ in range(self.delay_steps + 1):
            self.obs_buffer.append(initial_obs)

        return self.obs_buffer[0]