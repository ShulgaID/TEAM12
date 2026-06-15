from enum import Enum
from dataclasses import dataclass
import numpy as np

#Gait states for the finite state machine
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
    #State information for the controller
    current_state: GaitState
    phase: float  # 0.0 to 1.0 within current state
    step_count: int
    time_in_state: float
    target_angles: np.ndarray  # [hip, knee, ankle] * 2 legs
    actual_angles: np.ndarray  # Current joint angles
    contact_flags: np.ndarray  # Contact state [left_foot, right_foot]

#Finite state machine-based controller for stair climbing
class StairControllerFSM:

    #Initialize FSM controller. Args: step_height: Height of each stair step (meters); step_depth: Depth of each stair step (meters)
    def __init__(self, step_height: float = 0.3, step_depth: float = 0.4):
        self.step_height = step_height
        self.step_depth = step_depth

        # Robot segment lengths (meters)
        self.hip_to_knee = 0.6
        self.knee_to_ankle = 0.6
        self.ankle_to_foot = 0.15

        # State machine configuration
        self.current_state = GaitState.IDLE
        self.previous_state = GaitState.IDLE
        self.state_start_time = 0.0

        # Phase timing (in seconds)
        self.left_stance_duration = 0.3
        self.left_swing_duration = 0.2
        self.right_stance_duration = 0.3
        self.right_swing_duration = 0.2

        # Control gains
        self.kp = 50.0  # Proportional gain
        self.kd = 5.0   # Derivative gain

        # State history for debugging
        self.state_history = []
        self.max_history_size = 1000

        # Initialize controller state
        self.controller_state = ControllerState(
            current_state=GaitState.IDLE,
            phase=0.0,
            step_count=0,
            time_in_state=0.0,
            target_angles=np.zeros(6),
            actual_angles=np.zeros(6),
            contact_flags=np.array([True, True])
        )

    #Execute one control loop step
    #Args: current_time: Current simulation time (seconds); observation: Current state observation from simulator
    #Returns: Control commands for 6 actuators
    def update(self, current_time: float, observation: dict) -> np.ndarray:
        # Extract observation data
        self._extract_observation(observation)

        # Update state machine
        self._update_fsm(current_time)

        # Generate target trajectory based on current state
        self._compute_target_trajectory()

        # Compute control output
        control_output = self._compute_control_output()

        # Record state for debugging
        self._record_state_history()

        return control_output

    def _extract_observation(self, observation: dict) -> None:
        """Extract relevant data from observation."""
        if "joint_angles" in observation:
            self.controller_state.actual_angles = observation["joint_angles"].copy()
        elif "qpos" in observation:
            qpos = observation["qpos"]
            if len(qpos) >= 8:
                self.controller_state.actual_angles = qpos[2:8].copy()

        if "contact_flags" in observation:
            self.controller_state.contact_flags = observation["contact_flags"].copy()

    #Update finite state machine based on transitions.
    #Args: current_time: Current simulation time
    def _update_fsm(self, current_time: float) -> None:
        time_in_state = current_time - self.state_start_time
        self.controller_state.time_in_state = time_in_state

        # Calculate phase within current state (0.0 to 1.0)
        if self.current_state == GaitState.IDLE:
            self._handle_idle_state()

        elif self.current_state == GaitState.LEFT_STANCE:
            self.controller_state.phase = min(
                time_in_state / self.left_stance_duration, 1.0
            )
            if time_in_state >= self.left_stance_duration:
                self._transition_to(GaitState.LEFT_SWING, current_time)

        elif self.current_state == GaitState.LEFT_SWING:
            self.controller_state.phase = min(
                time_in_state / self.left_swing_duration, 1.0
            )
            if time_in_state >= self.left_swing_duration:
                self._transition_to(GaitState.RIGHT_STANCE, current_time)

        elif self.current_state == GaitState.RIGHT_STANCE:
            self.controller_state.phase = min(
                time_in_state / self.right_stance_duration, 1.0
            )
            if time_in_state >= self.right_stance_duration:
                self._transition_to(GaitState.RIGHT_SWING, current_time)

        elif self.current_state == GaitState.RIGHT_SWING:
            self.controller_state.phase = min(
                time_in_state / self.right_swing_duration, 1.0
            )
            if time_in_state >= self.right_swing_duration:
                self._transition_to(GaitState.LEFT_STANCE, current_time)
                self.controller_state.step_count += 1

        elif self.current_state == GaitState.ERROR:
            # Error state - try to recover
            pass

        self.controller_state.current_state = self.current_state

    #Handle initial idle state
    def _handle_idle_state(self) -> None:
        # Start with left leg stance
        self._transition_to(GaitState.LEFT_STANCE, 0.0)

    #Perform state transition
    #Args: new_state: Target state; time: Current time for state start marker
    def _transition_to(self, new_state: GaitState, time: float) -> None:
        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time

    def _compute_target_trajectory(self) -> None:
        """Compute target joint angles based on current FSM state."""
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

    #Generate target angles during stance phase
    #Args: progress: Progress through stance phase [0, 1]
    #Returns: Target angles [hip, knee, ankle]
    def _stance_trajectory(self, progress: float) -> np.ndarray:
        hip_angle = 0.3 * np.sin(progress * np.pi)
        knee_angle = 0.2 * np.cos(progress * np.pi)
        ankle_angle = 0.1 * np.sin(progress * np.pi)

        return np.array([hip_angle, knee_angle, ankle_angle])

    #Generate target angles during swing phase
    #Args: progress: Progress through swing phase [0, 1]
    #Returns: Target angles [hip, knee, ankle]
    def _swing_trajectory(self, progress: float) -> np.ndarray:
        hip_angle = 0.8 * progress - 0.4
        knee_angle = 1.2 * np.sin(progress * np.pi)
        ankle_angle = -0.2 * np.cos(progress * np.pi)

        return np.array([hip_angle, knee_angle, ankle_angle])

    #Generate target angles preparing for swing.
    #Args: progress: Progress [0, 1]
    #Returns: Target angles [hip, knee, ankle]
    def _prepare_swing_trajectory(self, progress: float) -> np.ndarray:
        # Slight motion during preparation phase
        hip_angle = 0.1 * np.cos(progress * np.pi)
        knee_angle = 0.05 * np.sin(progress * np.pi)
        ankle_angle = 0.05 * np.cos(progress * np.pi)

        return np.array([hip_angle, knee_angle, ankle_angle])

    #Compute motor control outputs using PD control
    #Returns: Control commands for 6 actuators
    def _compute_control_output(self) -> np.ndarray:
        error = self.controller_state.target_angles - self.controller_state.actual_angles
        control = self.kp * error

        # Clip to motor limits
        control_clipped = np.clip(control, -20, 20)

        return control_clipped

    #Record current state for debugging
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

        # Keep history bounded
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)

    #Get current controller state
    #Returns: Current ControllerState object
    def get_state(self) -> ControllerState:
        return self.controller_state

    #Get current state name
    #Returns: State name as string
    def get_state_name(self) -> str:
        return self.current_state.value

    #Reset controller to initial state
    def reset(self) -> None:
        self.current_state = GaitState.IDLE
        self.previous_state = GaitState.IDLE
        self.state_start_time = 0.0
        self.controller_state.step_count = 0
        self.state_history.clear()
