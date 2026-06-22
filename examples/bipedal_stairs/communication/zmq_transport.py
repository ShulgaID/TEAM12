"""
ZeroMQ transport layer for simulator ↔ controller communication.

Architecture (PUSH/PULL pipeline pattern):

    ┌─────────────────────┐          ┌──────────────────────┐
    │   SimulatorNode     │          │   ControllerNode     │
    │                     │          │                      │
    │  obs_sock (PUSH) ───┼──:5555──►│ obs_sock (PULL)      │
    │                     │          │                      │
    │  cmd_sock (PULL) ◄──┼──:5556──┤ cmd_sock (PUSH)      │
    └─────────────────────┘          └──────────────────────┘

Both nodes can run in the same process (threads) or separate processes.
Change OBS_ADDR / CMD_ADDR to connect across machines.
"""
from __future__ import annotations

import numpy as np
import zmq

# Bind on simulator side, connect on controller side
OBS_ADDR = "tcp://127.0.0.1:5555"  # sim → ctrl  (observations)
CMD_ADDR = "tcp://127.0.0.1:5556"  # ctrl → sim  (commands)

# Simulator polls for commands with a very short timeout so it never
# blocks the physics loop.  Controller waits longer for observations.
_SIM_CMD_TIMEOUT_MS  = 2    # non-blocking-ish on sim side
_CTRL_OBS_TIMEOUT_MS = 20   # controller polls at ~50 Hz when idle


class SimulatorTransport:
    """Simulator-side ZeroMQ sockets.

    Binds both ports so the controller can connect from any process.
    """

    def __init__(
        self,
        obs_addr: str = OBS_ADDR,
        cmd_addr: str = CMD_ADDR,
        recv_timeout_ms: int = _SIM_CMD_TIMEOUT_MS,
    ) -> None:
        self._ctx = zmq.Context.instance()

        self._obs_sock = self._ctx.socket(zmq.PUSH)
        self._obs_sock.setsockopt(zmq.SNDHWM, 2)
        self._obs_sock.bind(obs_addr)

        self._cmd_sock = self._ctx.socket(zmq.PULL)
        self._cmd_sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._cmd_sock.setsockopt(zmq.RCVHWM, 2)
        self._cmd_sock.bind(cmd_addr)

    # ------------------------------------------------------------------

    def send_observation(self, t: float, observation: dict) -> None:
        """Push a timestamped observation to the controller (non-blocking)."""
        payload = {
            "t": t,
            "qpos": observation["qpos"].tolist(),
            "qvel": observation["qvel"].tolist(),
        }
        try:
            self._obs_sock.send_json(payload, zmq.NOBLOCK)
        except zmq.Again:
            pass  # drop if controller is lagging

    def recv_command(self) -> np.ndarray | None:
        """Pull the latest control command (returns None quickly on timeout)."""
        try:
            msg = self._cmd_sock.recv_json()
            return np.array(msg["ctrl"], dtype=np.float64)
        except zmq.Again:
            return None

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._obs_sock.close(linger=0)
        self._cmd_sock.close(linger=0)


class ControllerTransport:
    """Controller-side ZeroMQ sockets.

    Connects to addresses bound by SimulatorTransport.
    """

    def __init__(
        self,
        obs_addr: str = OBS_ADDR,
        cmd_addr: str = CMD_ADDR,
        recv_timeout_ms: int = _CTRL_OBS_TIMEOUT_MS,
    ) -> None:
        self._ctx = zmq.Context.instance()

        self._obs_sock = self._ctx.socket(zmq.PULL)
        self._obs_sock.setsockopt(zmq.RCVTIMEO, recv_timeout_ms)
        self._obs_sock.setsockopt(zmq.RCVHWM, 2)
        self._obs_sock.connect(obs_addr)

        self._cmd_sock = self._ctx.socket(zmq.PUSH)
        self._cmd_sock.setsockopt(zmq.SNDHWM, 2)
        self._cmd_sock.connect(cmd_addr)

    # ------------------------------------------------------------------

    def recv_observation(self) -> tuple[float, dict] | None:
        """Pull the latest observation from the simulator.

        Returns ``(t, observation_dict)`` or ``None`` on timeout.
        """
        try:
            msg = self._obs_sock.recv_json()
            obs = {
                "qpos": np.array(msg["qpos"], dtype=np.float64),
                "qvel": np.array(msg["qvel"], dtype=np.float64),
            }
            return float(msg["t"]), obs
        except zmq.Again:
            return None

    def send_command(self, ctrl: np.ndarray) -> None:
        """Push a control command to the simulator (non-blocking)."""
        try:
            self._cmd_sock.send_json({"ctrl": ctrl.tolist()}, zmq.NOBLOCK)
        except zmq.Again:
            pass

    # ------------------------------------------------------------------

    def close(self) -> None:
        self._obs_sock.close(linger=0)
        self._cmd_sock.close(linger=0)
