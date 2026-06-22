"""
CPG controller for bipedal stair climbing.

Modes:
  1. Ramp-up + flat walking  (t < ~7 s)
  2. Stair climbing CPG      (robot on stairs)
  3. Platform walk           (flat-ground CPG + velocity feedback to target position)
  4. Stand upright           (knees straight, hips neutral) at platform_target_x
"""
import numpy as np


class CPGController:
    def __init__(self):
        self.frequency          = 0.62
        self.hip_amplitude      = 0.52
        self.hip_stair_inc      = 0.14
        self.base_knee_bend     = 0.18
        self.knee_clearance     = 0.65
        self.cl_scale           = 0.07
        self.knee_phase_shift   = 0.30
        self.ankle_flat         = 0.12
        self.ankle_stair        = 0.22
        self.ankle_stair_inc    = 0.07
        self.ramp_time          = 3.5
        self.stair_start_x      = 0.80
        self.stair_step_depth   = 0.42
        self.stair_count        = 5
        self.hip_fade_dist      = 0.80

        # Platform walk: target is 2.8 m past the landing edge (~middle of platform)
        _landing_x = self.stair_start_x + self.stair_count * self.stair_step_depth
        self.platform_target_x  = -(_landing_x + 2.8)   # approx -5.7
        self.platform_walk_vel  = -0.45   # desired speed on platform (m/s, negative = forward)
        self.platform_slow_dist = 0.50   # start slowing this many metres before target
        self.stand_knee_bend    = 0.06   # slight bend to avoid hyperextension in stand mode

        self._x_pos       = 0.0
        self._x_vel       = 0.0
        self._torso_pitch = 0.0
        self._t0          = None
        self._phase       = 0.0
        self._t_prev      = None
        self._on_platform = False

    # ------------------------------------------------------------------
    def _stair_factor(self) -> float:
        s = (-self._x_pos - self.stair_start_x) / self.stair_step_depth
        return float(np.clip(s, 0.0, float(self.stair_count)))

    def _platform_dist(self) -> float:
        landing_x = self.stair_start_x + self.stair_count * self.stair_step_depth
        return float(max(0.0, -self._x_pos - landing_x))

    # ------------------------------------------------------------------
    def update(self, current_time: float, observation: dict) -> np.ndarray:
        if "qpos" in observation:
            self._x_pos       = float(observation["qpos"][0])
            self._torso_pitch = float(observation["qpos"][2])
        if "qvel" in observation:
            self._x_vel = float(observation["qvel"][0])

        if self._t0 is None:
            self._t0     = current_time
            self._t_prev = current_time

        sf    = self._stair_factor()
        sf1   = float(np.clip(sf, 0.0, 1.0))
        sf3   = float(np.clip(sf, 0.0, 3.0))
        sf5   = float(np.clip(sf, 0.0, 5.0))
        on_plat = self._platform_dist()

        # S-curve amplitude ramp (avoids shuffle at start)
        raw_ramp = float(np.clip(current_time / self.ramp_time, 0.0, 1.0))
        ramp     = float(np.sin(np.pi * raw_ramp / 2.0) ** 2)

        freq = self.frequency - 0.14 * sf1
        dt   = current_time - self._t_prev
        self._t_prev = current_time

        pitch_corr = float(np.clip(0.5 * self._torso_pitch, -0.12, 0.12))
        balance    = float(np.clip(-0.45 * self._x_vel, -0.20, 0.20))

        self._on_platform = (on_plat > 0.0)

        # ================================================================
        # PLATFORM MODE
        # ================================================================
        if on_plat > 0.0:
            dist_to_target = self._x_pos - self.platform_target_x  # >0 while approaching

            # ---- STAND UPRIGHT: reached target ----
            if dist_to_target <= 0.0:
                hip_cmd = float(np.clip(pitch_corr + balance, -0.20, 0.20))
                angles  = np.array([hip_cmd, -self.stand_knee_bend, 0.0,
                                    hip_cmd, -self.stand_knee_bend, 0.0])
                return np.clip(angles, -1.57, 1.57)

            # ---- WALK TO TARGET: flat-ground CPG + velocity feedback ----
            # Ramp desired velocity to 0 as we approach the target
            speed_frac = float(np.clip(dist_to_target / self.platform_slow_dist, 0.0, 1.0))
            v_des      = self.platform_walk_vel * speed_frac
            v_error    = self._x_vel - v_des
            hip_drive  = float(np.clip(-0.80 * v_error, -0.35, 0.35))

            # Advance phase at flat-ground frequency
            flat_freq = 0.68
            self._phase += 2.0 * np.pi * flat_freq * dt
            phase = self._phase

            p_hip_amp = 0.38
            p_cl      = 0.48
            p_ank_amp = 0.16

            left_hip   =  p_hip_amp * np.sin(phase) + pitch_corr + hip_drive
            right_hip  = -p_hip_amp * np.sin(phase) + pitch_corr + hip_drive
            left_knee  = -(self.base_knee_bend + p_cl * max(0.0, np.cos(phase)) ** 1.2)
            right_knee = -(self.base_knee_bend + p_cl * max(0.0, -np.cos(phase)) ** 1.2)
            left_ankle  =  p_ank_amp * np.sin(phase + 0.4)
            right_ankle = -p_ank_amp * np.sin(phase + 0.4)

            angles = np.array([left_hip, left_knee, left_ankle,
                               right_hip, right_knee, right_ankle])
            return np.clip(angles, -1.57, 1.57)

        # ================================================================
        # STAIR CLIMBING / FLAT WALKING CPG
        # ================================================================
        self._phase += 2.0 * np.pi * freq * dt
        phase = self._phase

        # Fade hip amplitude over last hip_fade_dist before platform edge
        landing_x    = self.stair_start_x + self.stair_count * self.stair_step_depth
        dist_to_edge = landing_x - (-self._x_pos)
        hip_fade     = float(np.clip(dist_to_edge / self.hip_fade_dist, 0.0, 1.0))

        # ---- Hip -------------------------------------------------------
        hip_amp   = (self.hip_amplitude + self.hip_stair_inc * sf5 / 5.0) * ramp * hip_fade
        left_hip  =  hip_amp * np.sin(phase) + pitch_corr + balance
        right_hip = -hip_amp * np.sin(phase) + pitch_corr + balance

        # ---- Knee -------------------------------------------------------
        kshift = self.knee_phase_shift * sf1
        cl_eff = self.knee_clearance + self.cl_scale * sf3
        left_knee  = -(self.base_knee_bend
                       + cl_eff * max(0.0, np.cos(phase - kshift)) ** 1.2)
        right_knee = -(self.base_knee_bend
                       + cl_eff * max(0.0, -np.cos(phase - kshift)) ** 1.2)

        # ---- Ankle -------------------------------------------------------
        ank_amp = (
            self.ankle_flat * (1.0 - sf1)
            + (self.ankle_stair + self.ankle_stair_inc * sf1) * sf1
        ) * ramp * hip_fade
        left_ankle  =  ank_amp * np.sin(phase + 0.4)
        right_ankle = -ank_amp * np.sin(phase + 0.4)

        angles = np.array([left_hip, left_knee, left_ankle,
                           right_hip, right_knee, right_ankle])
        return np.clip(angles, -1.57, 1.57)

    # ------------------------------------------------------------------
    def reset(self):
        self._t0          = None
        self._t_prev      = None
        self._phase       = 0.0
        self._x_pos       = 0.0
        self._x_vel       = 0.0
        self._torso_pitch = 0.0
        self._on_platform = False

    def get_state_name(self) -> str:
        if self._t0 is None:
            return "idle"
        d = self._platform_dist()
        if d > 0.0:
            if self._x_pos <= self.platform_target_x:
                return "standing_upright"
            return "walking_platform"
        sf = self._stair_factor()
        return f"climbing_stair_{int(sf) + 1}" if sf > 0.1 else "walking"
