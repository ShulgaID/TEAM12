"""
Bipedal robot stair-climbing simulation.

Simulator and controller run in **separate threads** and exchange messages
over ZeroMQ (PUSH/PULL on tcp://127.0.0.1:5555-5556).

┌─────────────────────┐   ZeroMQ   ┌──────────────────────┐
│   SimulatorNode     │ ──obs:5555─► │   ControllerNode     │
│   (MuJoCo physics)  │ ◄─cmd:5556── │   (CPG gait)         │
└─────────────────────┘            └──────────────────────┘

Usage
-----
Run both nodes in one command (default):

    python -m examples.bipedal_stairs.run

Or launch each node in a separate terminal:

    # Terminal 1
    python -m examples.bipedal_stairs.sim.sim_node

    # Terminal 2
    python -m examples.bipedal_stairs.control.ctrl_node
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from examples.bipedal_stairs.sim.sim_node import SimulatorNode
from examples.bipedal_stairs.control.ctrl_node import ControllerNode


def main(max_steps: int = 10_000) -> None:
    stop_event = threading.Event()

    sim  = SimulatorNode(stop_event=stop_event, max_steps=max_steps)
    ctrl = ControllerNode(stop_event=stop_event)

    print("=" * 60)
    print("Bipedal Stairs — ZeroMQ transport")
    print(f"  obs  tcp://127.0.0.1:5555  (sim → ctrl)")
    print(f"  cmd  tcp://127.0.0.1:5556  (ctrl → sim)")
    print("=" * 60)

    t_ctrl = threading.Thread(target=ctrl.run, name="controller", daemon=True)
    t_sim  = threading.Thread(target=sim.run,  name="simulator",  daemon=False)

    # Controller must be ready before simulator starts sending
    t_ctrl.start()
    time.sleep(0.05)   # give ZMQ sockets time to bind/connect
    t_sim.start()

    try:
        t_sim.join()
    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        stop_event.set()
        t_ctrl.join(timeout=2.0)

    print("[main] done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bipedal stairs simulation")
    parser.add_argument("--steps", type=int, default=10_000,
                        help="Maximum physics steps (default 10 000 = 20 s)")
    args = parser.parse_args()
    main(max_steps=args.steps)
