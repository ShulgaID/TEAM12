from enum import Enum
from dataclasses import dataclass
import numpy as np


class GaitState(Enum):
    IDLE = "idle"
    LEFT_STANCE = "left_stance"
    LEFT_SWING = "left_swing"
    RIGHT_STANCE = "right_stance"
    RIGHT_SWING = "right_swing"
    TRANSITION = "transition"
    ERROR = "error"


@dataclass
class ControllerState:
    current_state: GaitState
    phase: float
    step_count: int
    time_in_state: float
    target_angles: np.ndarray
    actual_angles: np.ndarray
    contact_flags: np.ndarray


class StairControllerFSM:

    def __init__(self, step_height: float = 0.3, step_depth: float = 0.4):
        self.step_height = step_height
        self.step_depth = step_depth
        self._x_vel: float = 0.0
        self._x_pos: float = 0.0
        self.stair_start_x: float = 0.8
        self.stair_step_depth: float = 0.42   # matches bipedal_robot.xml (tread=0.42m)
        self.stair_step_height: float = 0.13  # matches bipedal_robot.xml (riser=0.13m)
        self.stair_count: int = 5

        self.hip_to_knee = 0.6
        self.knee_to_ankle = 0.6
        self.ankle_to_foot = 0.15

        self.current_state = GaitState.IDLE
        self.previous_state = GaitState.IDLE
        self.state_start_time = 0.0

        self.left_stance_duration = 0.38
        self.left_swing_duration = 0.38
        self.right_stance_duration = 0.38
        self.right_swing_duration = 0.38

        self.kp = 120.0
        self.ki = 5.0
        self.kd = 15.0
        self.integral_limit = 10.0
        self._prev_error: np.ndarray = np.zeros(6)
        self._integral: np.ndarray = np.zeros(6)
        self._prev_time: float = 0.0

        self.state_history = []
        self.max_history_size = 1000

        self.controller_state = ControllerState(
            current_state=GaitState.IDLE,
            phase=0.0,
            step_count=0,
            time_in_state=0.0,
            target_angles=np.zeros(6),
            actual_angles=np.zeros(6),
            contact_flags=np.array([True, True])
        )

    def update(self, current_time: float, observation: dict) -> np.ndarray:
        self._extract_observation(observation)
        self._update_fsm(current_time)
        self._compute_target_trajectory()

        balance = float(np.clip(-0.25 * self._x_vel, -0.15, 0.15))
        angles = self.controller_state.target_angles.copy()
        angles[0] += balance
        angles[3] += balance
        control_output = np.clip(angles, -1.57, 1.57)

        self._record_state_history()
        return control_output

    def _extract_observation(self, observation: dict) -> None:
        if "joint_angles" in observation:
            self.controller_state.actual_angles = observation["joint_angles"].copy()
        elif "qpos" in observation:
            qpos = observation["qpos"]
            if len(qpos) >= 8:
                self.controller_state.actual_angles = qpos[2:8].copy()

        if "contact_flags" in observation:
            self.controller_state.contact_flags = observation["contact_flags"].copy()

        if "qvel" in observation:
            qvel = observation["qvel"]
            self._x_vel = float(qvel[0]) if len(qvel) > 0 else 0.0
        else:
            self._x_vel = 0.0
        if "qpos" in observation:
            qpos = observation["qpos"]
            self._x_pos = float(qpos[0]) if len(qpos) > 0 else 0.0
        else:
            self._x_pos = 0.0

    def _phase_durations(self):
        sf = min(self._stair_factor(), 1.0)
        stance = self.left_stance_duration + 0.17 * sf
        swing  = self.left_swing_duration  + 0.22 * sf
        return stance, swing

    def _update_fsm(self, current_time: float) -> None:
        time_in_state = current_time - self.state_start_time
        self.controller_state.time_in_state = time_in_state
        stance_dur, swing_dur = self._phase_durations()

        if self.current_state == GaitState.IDLE:
            self._handle_idle_state()

        elif self.current_state == GaitState.LEFT_STANCE:
            self.controller_state.phase = min(time_in_state / stance_dur, 1.0)
            if time_in_state >= stance_dur:
                self._transition_to(GaitState.LEFT_SWING, current_time)

        elif self.current_state == GaitState.LEFT_SWING:
            self.controller_state.phase = min(time_in_state / swing_dur, 1.0)
            if time_in_state >= swing_dur:
                self._transition_to(GaitState.RIGHT_STANCE, current_time)

        elif self.current_state == GaitState.RIGHT_STANCE:
            self.controller_state.phase = min(time_in_state / stance_dur, 1.0)
            if time_in_state >= stance_dur:
                self._transition_to(GaitState.RIGHT_SWING, current_time)

        elif self.current_state == GaitState.RIGHT_SWING:
            self.controller_state.phase = min(time_in_state / swing_dur, 1.0)
            if time_in_state >= swing_dur:
                self._transition_to(GaitState.LEFT_STANCE, current_time)
                self.controller_state.step_count += 1

        elif self.current_state == GaitState.ERROR:
            pass

        self.controller_state.current_state = self.current_state

    def _handle_idle_state(self) -> None:
        self._transition_to(GaitState.LEFT_STANCE, 0.0)

    def _transition_to(self, new_state: GaitState, time: float) -> None:
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time

    def _compute_target_trajectory(self) -> None:
        phase = self.controller_state.phase

        if self.current_state == GaitState.LEFT_STANCE:
            left_angles = self._stance_trajectory(phase)
            right_angles = self._prepare_swing_trajectory(phase)

        elif self.current_state == GaitState.LEFT_SWING:
            left_angles = self._swing_trajectory(phase)
            right_angles = self._stance_trajectory(phase)

        elif self.current_state == GaitState.RIGHT_STANCE:
            left_angles = self._prepare_swing_trajectory(phase)
            right_angles = self._stance_trajectory(phase)

        elif self.current_state == GaitState.RIGHT_SWING:
            left_angles = self._stance_trajectory(phase)
            right_angles = self._swing_trajectory(phase)

        else:
            left_angles = np.zeros(3)
            right_angles = np.zeros(3)

        self.controller_state.target_angles = np.concatenate([left_angles, right_angles])

    def _stair_factor(self) -> float:
        x = self._x_pos
        steps_in = (-x - self.stair_start_x) / self.stair_step_depth
        return float(np.clip(steps_in, 0.0, float(self.stair_count)))

    def _lift_profile(self, p: float) -> float:
        rise = min(1.0, p / 0.25)
        fall = max(0.0, 1.0 - max(0.0, p - 0.70) / 0.30)
        return min(rise, fall)

    def _stance_trajectory(self, progress: float) -> np.ndarray:
        sf = min(self._stair_factor(), 1.0)
        # On stairs, use SMALLER hip range (tread=0.42m < normal step length).
        # Increasing range with sf caused the split — now it decreases.
        hip_start  =  0.38 - 0.04 * sf
        hip_end    = -0.38 + 0.04 * sf
        hip_angle  = hip_start + (hip_end - hip_start) * progress
        knee_angle = -0.22 - 0.13 * sf
        ankle_end   = -0.28 - 0.08 * sf
        ankle_angle = +0.22 + (ankle_end - 0.22) * progress
        return np.array([hip_angle, knee_angle, ankle_angle])

    def _swing_trajectory(self, progress: float) -> np.ndarray:
        sf = self._stair_factor()
        # Match stance: smaller symmetric hip range to avoid split.
        # Was min(sf, 3.0) for hip_end — allowed 0.58 rad vs -0.46 for hip_start.
        hip_start = -0.38 + 0.04 * min(sf, 1.0)
        hip_end   =  0.38 - 0.04 * min(sf, 1.0)
        hip_angle = hip_start + (hip_end - hip_start) * progress
        x = self._x_pos
        dist_to_stair = -x - (self.stair_start_x - 0.5)
        anticipate = float(np.clip(dist_to_stair / self.stair_step_depth, 0.0, 1.0))
        extra_lift = 0.12 * anticipate + 0.30 * sf
        knee_lift  = (1.0 + extra_lift) * self._lift_profile(progress)
        knee_angle = -knee_lift
        ankle_angle = +0.18 * progress
        return np.array([hip_angle, knee_angle, ankle_angle])

    def _prepare_swing_trajectory(self, progress: float) -> np.ndarray:
        sf = min(self._stair_factor(), 1.0)
        # Must match reduced stance hip range to avoid discontinuity at transition.
        hip_angle   =  0.38 - 0.04*sf - (0.76 - 0.08*sf) * progress
        knee_angle  = -0.22 - 0.13 * sf
        ankle_angle = +0.22 - 0.50 * progress
        return np.array([hip_angle, knee_angle, ankle_angle])

    def _compute_control_output(self, dt: float) -> np.ndarray:
        error = self.controller_state.target_angles - self.controller_state.actual_angles
        self._integral += error * dt
        self._integral = np.clip(self._integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error.copy()
        control = self.kp * error + self.ki * self._integral + self.kd * derivative
        return np.clip(control, -20, 20)

    def _record_state_history(self) -> None:
        state_record = {
            "time": self.controller_state.time_in_state,
            "state": self.current_state.value,
            "phase": self.controller_state.phase,
            "step_count": self.controller_state.step_count,
            "target_angles": self.controller_state.target_angles.copy(),
            "actual_angles": self.controller_state.actual_angles.copy(),
        }
        self.state_history.append(state_record)
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)

    def get_state(self) -> ControllerState:
        return self.controller_state

    def get_state_name(self) -> str:
        return self.current_state.value

    def reset(self) -> None:
        self.current_state = GaitState.IDLE
        self.previous_state = GaitState.IDLE
        self.state_start_time = 0.0
        self.controller_state.step_count = 0
        self.state_history.clear()
        self._prev_error = np.zeros(6)
        self._integral = np.zeros(6)
        self._prev_time = 0.0
