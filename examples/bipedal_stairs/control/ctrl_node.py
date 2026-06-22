"""
Controller node — receives observations via ZeroMQ and sends CPG commands.

Can be started standalone:
    python -m examples.bipedal_stairs.control.ctrl_node
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.bipedal_stairs.communication.zmq_transport import ControllerTransport
from examples.bipedal_stairs.control.cpg_controller import CPGController


class ControllerNode:
    """Wraps CPGController with ZeroMQ I/O."""

    def __init__(self, stop_event: threading.Event | None = None) -> None:
        self._stop       = stop_event or threading.Event()
        self._transport  = ControllerTransport()
        self._controller = CPGController()
        self._controller.reset()

    def run(self) -> None:
        print("[ctrl] ready — waiting for observations")

        while not self._stop.is_set():
            result = self._transport.recv_observation()
            if result is None:
                continue

            t, obs = result
            ctrl   = self._controller.update(t, obs)
            self._transport.send_command(ctrl)

        self._transport.close()
        print("[ctrl] finished")


# ------------------------------------------------------------------
if __name__ == "__main__":
    ControllerNode().run()
