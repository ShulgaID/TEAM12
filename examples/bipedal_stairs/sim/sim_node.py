"""
Simulator node — runs MuJoCo physics and communicates via ZeroMQ.

Can be started standalone:
    python -m examples.bipedal_stairs.sim.sim_node [--steps N]
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.bipedal_stairs.communication.zmq_transport import SimulatorTransport

_XML = Path(__file__).parent / "bipedal_robot.xml"
_DT  = 0.002   # model timestep (s)


class SimulatorNode:
    """Runs MuJoCo physics and publishes observations / consumes commands."""

    def __init__(
        self,
        stop_event: threading.Event | None = None,
        max_steps: int = 10_000,
    ) -> None:
        self._stop   = stop_event or threading.Event()
        self._steps  = max_steps

        self._model  = mujoco.MjModel.from_xml_path(str(_XML))
        self._data   = mujoco.MjData(self._model)
        self._transport = SimulatorTransport()

        self._reset()

    # ------------------------------------------------------------------

    def _reset(self) -> None:
        mujoco.mj_resetData(self._model, self._data)
        self._data.qpos[0] = 0.0    # slide_x
        self._data.qpos[1] = 0.09   # slide_z (feet at floor)
        self._data.qpos[2] = 0.0    # hinge_y (torso upright)
        self._data.ctrl[:] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def run(self) -> None:
        last_ctrl: np.ndarray = np.zeros(self._model.nu)

        print(f"[sim] starting — {self._steps} steps @ {1/_DT:.0f} Hz")

        for step in range(self._steps):
            if self._stop.is_set():
                break

            t   = step * _DT
            obs = {
                "qpos": self._data.qpos.copy(),
                "qvel": self._data.qvel.copy(),
            }

            # 1. Push observation to controller
            self._transport.send_observation(t, obs)

            # 2. Pull latest command (non-blocking, keep last if none arrives)
            cmd = self._transport.recv_command()
            if cmd is not None:
                last_ctrl = cmd

            # 3. Apply command and step physics
            self._data.ctrl[:] = last_ctrl
            mujoco.mj_step(self._model, self._data)

            # Progress log every 2 s
            if step % 1000 == 0:
                x  = float(self._data.qpos[0])
                z  = float(self._data.xpos[1, 2])
                vx = float(self._data.qvel[0])
                print(f"[sim] t={t:.1f}s  x={x:.2f}  z_torso={z:.3f}  vx={vx:+.2f}")

        self._stop.set()
        self._transport.close()
        print("[sim] finished")


# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10_000)
    args = parser.parse_args()
    SimulatorNode(max_steps=args.steps).run()


if __name__ == "__main__":
    main()
