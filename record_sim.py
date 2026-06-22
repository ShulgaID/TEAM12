"""Record MuJoCo simulation to MP4 -- direct physics, no comm layer delay.

Recording starts at RECORD_AFTER_T=2.0s -- the robot is visibly walking at
~60% amplitude by then, so there is no shuffling, and ~1.5s of flat-ground
approach is captured before the stairs.
"""
import sys
from pathlib import Path

root = Path(__file__).parent
sys.path.insert(0, str(root))

import mujoco
import numpy as np
from examples.bipedal_stairs.control.cpg_controller import CPGController

STEPS           = 12000    # 24 s simulation (extra time for platform steps)
RECORD_AFTER_T  = 2.0      # skip near-zero-amplitude phase
SAMPLE          = 20       # 25 frames/s -> real-time video
W, H            = 1280, 720
FPS             = 25
OUT             = "simulation_mujoco.mp4"

model_path = root / "examples/bipedal_stairs/sim/bipedal_robot.xml"
model = mujoco.MjModel.from_xml_path(str(model_path))
data  = mujoco.MjData(model)

controller = CPGController()
renderer   = mujoco.Renderer(model, height=H, width=W)

mujoco.mj_resetData(model, data)
data.qpos[0] = 0.0
data.qpos[1] = 0.09
data.qpos[2] = 0.0
data.ctrl[:] = 0.0
mujoco.mj_forward(model, data)
controller.reset()

cam = mujoco.MjvCamera()
cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
cam.distance  = 6.5
cam.azimuth   = 85.0
cam.elevation = -12.0

frames = []
print("Simulating %d steps; recording after t=%.1fs ..." % (STEPS, RECORD_AFTER_T))

for step in range(STEPS):
    t = step * model.opt.timestep
    obs = {"qpos": data.qpos.copy(), "qvel": data.qvel.copy()}
    ctrl_out = controller.update(t, obs)
    data.ctrl[:] = ctrl_out
    mujoco.mj_step(model, data)

    if t >= RECORD_AFTER_T and step % SAMPLE == 0:
        x = float(data.qpos[0])
        z = float(data.xpos[1, 2])
        cam.lookat[0] = x - 1.2
        cam.lookat[1] = 0.0
        cam.lookat[2] = max(0.5, z - 0.1)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    if step % 1000 == 0:
        x = data.qpos[0]
        z = data.xpos[1, 2]
        sf = max(0, (-x - 0.80) / 0.42)
        state = controller.get_state_name()
        print("  t=%.1fs  x=%.2f  z=%.3f  stair=%.1f  %s  frames=%d" % (
              t, x, z, sf, state, len(frames)))

print("Saving %d frames -> %s ..." % (len(frames), OUT))
import imageio
imageio.mimwrite(
    OUT, frames, fps=FPS, format="FFMPEG", codec="libx264",
    output_params=["-crf", "18", "-pix_fmt", "yuv420p"]
)
print("Done! -> " + OUT)
