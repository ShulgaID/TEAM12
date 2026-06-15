import os
from typing import Any
import mujoco
import numpy as np
from examples.bipedal_stairs.communication.layer import CommunicationLayer

#MuJoCo environment for bipedal robot climbing stairs with communication layer
class BipedalStairsEnv:

    #Initialize the environment. Args: communication_layer = Communication layer instance; render = Whether to render the simulation
    def __init__(self, communication_layer: CommunicationLayer, render: bool = False):

        self.communication_layer = communication_layer
        self.render = render

        # Load MuJoCo models
        model_dir = os.path.dirname(__file__)
        robot_path = os.path.join(model_dir, "bipedal_robot.xml")

        # Load robot model
        self.model = mujoco.MjModel.from_xml_file(robot_path)
        self.data = mujoco.MjData(self.model)

        # Setup stairs in the scene
        self._setup_stairs()

        # Renderer
        self.viewer = None
        if self.render:
            self.viewer = mujoco.Viewer(self.model, self.data)

        # Simulation parameters
        self.timestep = self.model.opt.timestep
        self.frame_skip = 1
        self.current_time = 0.0

        # Last control command and state
        self.last_control = np.zeros(6)
        self.last_controller_state = "idle"

    #Setup stairs in the simulation
    def _setup_stairs(self) -> None:
        # Stairs are already in the robot model's world body
        pass

    #Reset the environment and notify controller
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)

        # Set initial robot position
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.data.xpos[torso_id] = np.array([0.0, 0.0, 1.0])

        mujoco.mj_forward(self.model, self.data)

        self.current_time = 0.0
        self.last_control = np.zeros(6)

        # Send reset signal through communication layer
        self.communication_layer.send_reset(self.current_time)

    #Execute one step of the simulation
    #   1. Get observation from current state
    #   2. Send observation through communication layer
    #   3. Receive control command from controller
    #   4. Apply control to simulation
    #   5. Step physics
    #   6. Return new observation
    #
    # Returns: Tuple of (observation, is_fallen)
    def step(self) -> tuple[dict, bool]:
        # Step 1: Get current observation
        observation = self._get_observation()
        # Step 2: Send observation to controller via communication layer
        self.communication_layer.send_observation(self.current_time, observation)
        # Step 3: Receive control command from controller via communication layer
        control_data = self.communication_layer.receive_control_command()
        if control_data is not None:
            control, controller_state = control_data
            self.last_control = np.array(control)
            self.last_controller_state = controller_state
        else:
            # No control received, use last known command
            control = self.last_control
        # Step 4: Apply control to simulation
        self.data.ctrl[:] = control
        # Step 5: Step physics simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.current_time += self.timestep * self.frame_skip
        # Step 6: Get new observation
        new_observation = self._get_observation()

        # Check if robot has fallen
        is_fallen = self._is_fallen()

        # Render if enabled
        if self.render and self.viewer:
            self.viewer.sync()

        return new_observation, is_fallen

    #Get current observation from simulation. Returns: Dictionary containing observation data
    def _get_observation(self) -> dict:
        obs = {}

        # Get torso position and orientation
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        obs["torso_position"] = self.data.xpos[torso_id].copy()
        obs["torso_quaternion"] = self.data.xquat[torso_id].copy()
        obs["torso_velocity"] = self.data.cvel[torso_id * 6 : torso_id * 6 + 6].copy()

        # Get foot positions
        left_ankle_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle"
        )
        right_ankle_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle"
        )
        obs["left_foot_position"] = self.data.xpos[left_ankle_id].copy()
        obs["right_foot_position"] = self.data.xpos[right_ankle_id].copy()

        # Get joint positions (qpos) and velocities (qvel)
        obs["qpos"] = self.data.qpos.copy()
        obs["qvel"] = self.data.qvel.copy()

        # Extract joint angles for the 6 controlled joints
        if len(obs["qpos"]) >= 8:
            obs["joint_angles"] = obs["qpos"][2:8].copy()
        else:
            obs["joint_angles"] = np.zeros(6)

        # Get contact information
        obs["contact_flags"] = self._get_contact_flags()

        # Simulation metadata
        obs["time"] = self.current_time
        obs["last_control"] = self.last_control.copy()
        obs["controller_state"] = self.last_controller_state

        return obs

    # Get contact flags for both feet. Returns: array [left_foot_contact, right_foot_contact]
    def _get_contact_flags(self) -> np.ndarray:
        contacts = np.array([False, False])

        # Check for contacts with ground
        for contact in range(self.data.ncon):
            # Contact geometry IDs
            geom1 = self.data.contact[contact].geom1
            geom2 = self.data.contact[contact].geom2

            # Get geom names
            try:
                name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)

                if "left_foot" in (name1 or "") or "left_foot" in (name2 or ""):
                    contacts[0] = True
                if "right_foot" in (name1 or "") or "right_foot" in (name2 or ""):
                    contacts[1] = True
            except:
                pass

        return contacts

    #Check if robot has fallen. Returns: True if robot has fallen, False otherwise
    def _is_fallen(self) -> bool:
        # Get torso height
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        torso_z = self.data.xpos[torso_id][2]

        # Check if torso is too low (fallen)
        if torso_z < 0.5:
            return True

        # Check if torso is tilted too much (fallen to the side)
        torso_quat = self.data.xquat[torso_id]
        # For 2D control, Y-axis rotation (roll) should stay near 0
        roll = 2 * np.arctan2(torso_quat[2], torso_quat[0])
        if abs(roll) > np.pi / 4:  # 45 degrees
            return True

        return False

    #Get current simulation time
    def get_time(self) -> float:
        return self.current_time

    #Get current environment state
    def get_state(self) -> dict:
        return {
            "time": self.current_time,
            "last_control": self.last_control.copy(),
            "last_controller_state": self.last_controller_state,
        }

    #Close the environment
    def close(self) -> None:
        if self.viewer:
            self.viewer.close()
