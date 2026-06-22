"""
Diagnostic: robot standing with neutral pose (hip=0, knee=0).
With all joints at 0, no horizontal forces from actuators.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = str(Path(__file__).parent / "examples/bipedal_stairs/sim/bipedal_robot.xml")

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Neutral pose: legs straight down (hip=0, knee=0, ankle=0)
# slide_z = leg_length + foot_half - torso_body_z = 0.6+0.6+0.15+0.04 - 1.3 = 0.09
data.qpos[0] = 0.0   # slide_x
data.qpos[1] = 0.09  # slide_z: straight legs land exactly on ground
data.qpos[2] = 0.0   # hinge_y
data.qpos[3] = 0.0   # left_hip_pitch
data.qpos[4] = 0.0   # left_knee_pitch  (straight)
data.qpos[5] = 0.0   # left_ankle_pitch
data.qpos[6] = 0.0   # right_hip_pitch
data.qpos[7] = 0.0   # right_knee_pitch
data.qpos[8] = 0.0   # right_ankle_pitch

# Position actuators: target = 0 (neutral), no horizontal forces
TARGET = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
data.ctrl[:] = TARGET

mujoco.mj_forward(model, data)
mujoco.mj_kinematics(model, data)
print(f"Initial torso z: {data.qpos[1]:.3f}  (slide_z)")
print(f"Initial torso body z: {data.xpos[1, 2]:.3f}")

for step in range(2000):
    # Balance feedback: adjust hips to lean into drift direction
    x_vel = data.qvel[0]   # horizontal velocity
    x_pos = data.qpos[0]   # horizontal position (drift)
    balance = np.clip(-1.0 * x_vel - 0.3 * x_pos, -0.3, 0.3)

    ctrl = np.array([balance, -0.05, 0.0,
                     balance, -0.05, 0.0])
    data.ctrl[:] = ctrl
    mujoco.mj_step(model, data)

    if step % 200 == 0:
        z = data.xpos[1, 2]
        x = data.qpos[0]
        pitch = data.qpos[2]
        print(f"  step={step:4d}  torso_z={z:.3f}  torso_x={x:.3f}  pitch={pitch:.3f} rad")

print(f"\nFinal torso z: {data.xpos[1, 2]:.3f}")
print(f"Final torso x: {data.qpos[0]:.3f}")
print("PASS: robot standing!" if data.xpos[1, 2] > 0.8 else "FAIL: robot fell")
