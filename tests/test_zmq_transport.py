"""
Integration tests for the ZeroMQ transport layer.

Spins up SimulatorTransport and ControllerTransport in separate threads
and verifies observation + command round-trips.
"""
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from examples.bipedal_stairs.communication.zmq_transport import (
    ControllerTransport,
    SimulatorTransport,
)

# Use non-default ports to avoid conflicts with a running simulation
_OBS = "tcp://127.0.0.1:15555"
_CMD = "tcp://127.0.0.1:15556"


def _make_obs(t: float):
    return {
        "qpos": np.zeros(9),
        "qvel": np.zeros(9),
    }


class TestRoundTrip:
    """Verify that observations flow sim→ctrl and commands flow ctrl→sim."""

    def test_single_message(self):
        """Send one observation, receive one command back."""
        received_obs: list = []
        received_cmd: list = []
        ready = threading.Event()
        done  = threading.Event()

        def controller_thread():
            transport = ControllerTransport(obs_addr=_OBS, cmd_addr=_CMD)
            ready.set()
            result = transport.recv_observation()
            if result is not None:
                t, obs = result
                received_obs.append((t, obs))
                ctrl = np.ones(6) * 0.1
                transport.send_command(ctrl)
            transport.close()
            done.set()

        sim = SimulatorTransport(obs_addr=_OBS, cmd_addr=_CMD)
        t_ctrl = threading.Thread(target=controller_thread, daemon=True)
        t_ctrl.start()

        ready.wait(timeout=2.0)
        time.sleep(0.01)   # give ZMQ sockets time to connect

        sim.send_observation(1.23, _make_obs(1.23))
        done.wait(timeout=2.0)

        cmd = sim.recv_command()

        sim.close()
        t_ctrl.join(timeout=2.0)

        assert len(received_obs) == 1
        assert received_obs[0][0] == pytest.approx(1.23)
        assert cmd is not None
        assert cmd.shape == (6,)
        assert np.allclose(cmd, 0.1)

    def test_observation_fields(self):
        """Controller receives correct qpos/qvel arrays."""
        received: list = []
        ready = threading.Event()
        done  = threading.Event()

        def controller_thread():
            transport = ControllerTransport(obs_addr=_OBS, cmd_addr=_CMD, recv_timeout_ms=100)
            ready.set()
            result = transport.recv_observation()
            if result is not None:
                received.append(result)
            transport.close()
            done.set()

        sim = SimulatorTransport(obs_addr=_OBS, cmd_addr=_CMD)
        t_ctrl = threading.Thread(target=controller_thread, daemon=True)
        t_ctrl.start()
        ready.wait(timeout=2.0)
        time.sleep(0.01)

        qpos = np.arange(9, dtype=float)
        qvel = np.ones(9, dtype=float) * 2.0
        sim.send_observation(0.5, {"qpos": qpos, "qvel": qvel})
        done.wait(timeout=2.0)
        sim.close()
        t_ctrl.join(timeout=2.0)

        assert len(received) == 1
        t_out, obs_out = received[0]
        assert t_out == pytest.approx(0.5)
        assert np.allclose(obs_out["qpos"], qpos)
        assert np.allclose(obs_out["qvel"], qvel)

    def test_none_on_timeout(self):
        """recv_command returns None if no command arrives."""
        sim = SimulatorTransport(obs_addr=_OBS, cmd_addr=_CMD, recv_timeout_ms=20)
        # Don't start a controller — should time out
        result = sim.recv_command()
        sim.close()
        assert result is None
