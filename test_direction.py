"""Quick check: does robot walk in +x direction?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))
import numpy as np

XML_PATH = str(Path(__file__).parent / "examples/bipedal_stairs/sim/bipedal_robot.xml")

# Manually test hip angle convention
# If positive hip = foot backward, stance -0.35->+0.35 should move robot forward
import math
def foot_x_offset(hip, knee):
    """foot x relative to hip (positive = forward in world)"""
    # Rotation Ry(hip): thigh direction (0,0,-1) -> (-sin(hip)*L, 0, -cos(hip)*L)
    # With MuJoCo Ry: x-component = -L*sin(hip)  (positive hip = foot behind)
    knee_x = -0.6 * math.sin(hip)
    ankle_abs = hip + knee
    foot_x = knee_x - 0.6 * math.sin(ankle_abs)
    return foot_x

print("Hip convention check:")
print(f"  hip=-0.35 (start stance): foot_x = {foot_x_offset(-0.35,-0.25):+.3f}  (should be POSITIVE = ahead)")
print(f"  hip=+0.35 (end stance):   foot_x = {foot_x_offset(+0.35,-0.35):+.3f}  (should be NEGATIVE = behind)")
print()
print("If start=positive and end=negative: body moves FORWARD over the foot ✓")
print("If start=negative and end=positive: body moves BACKWARD ✗ -- flip hip signs")
