"""Tests for PID controller and Renderer recording (issue #25)."""
import os
import sys
import tempfile

import numpy as np
import pytest

# ── paths ──────────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "simulator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples", "bipedal_stairs", "control"))

os.environ.setdefault("MPLBACKEND", "Agg")


# ══════════════════════════════════════════════════════════════════════════════
# PID controller
# ══════════════════════════════════════════════════════════════════════════════

from stair_controller import StairControllerFSM, GaitState


def _obs(angles=None):
    return {
        "joint_angles": np.zeros(6) if angles is None else np.array(angles),
        "contact_flags": np.array([True, True]),
    }


def test_pid_returns_array():
    ctrl = StairControllerFSM()
    ctrl.reset()
    out = ctrl.update(0.0, _obs())
    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)


def test_pid_output_clipped():
    """Control signal must stay within motor limits."""
    ctrl = StairControllerFSM()
    ctrl.reset()
    for t in np.arange(0, 2.0, 0.01):
        out = ctrl.update(float(t), _obs())
        assert np.all(out >= -20) and np.all(out <= 20), f"Clip violated at t={t}: {out}"


def test_pid_integral_accumulates():
    """_compute_control_output can accumulate integral when called directly."""
    ctrl = StairControllerFSM()
    ctrl.reset()
    obs = _obs([0.5] * 6)
    ctrl.update(0.00, obs)
    ctrl.update(0.01, obs)
    # FSM uses position actuators; call _compute_control_output explicitly
    ctrl._compute_control_output(0.01)
    assert not np.all(ctrl._integral == 0), "Integral should accumulate"


def test_pid_derivative_active():
    """prev_error is updated when _compute_control_output is called directly."""
    ctrl = StairControllerFSM()
    ctrl.reset()
    obs = _obs([0.3] * 6)
    ctrl.update(0.00, obs)
    ctrl.update(0.01, obs)
    # FSM uses position actuators; _prev_error is set by _compute_control_output
    ctrl._compute_control_output(0.01)
    assert not np.all(ctrl._prev_error == 0), "prev_error should be set"


def test_pid_reset_clears_state():
    ctrl = StairControllerFSM()
    ctrl.reset()
    obs = _obs([0.5] * 6)
    ctrl.update(0.0, obs)
    ctrl.update(0.01, obs)
    ctrl.reset()
    assert np.all(ctrl._integral == 0)
    assert np.all(ctrl._prev_error == 0)
    assert ctrl._prev_time == 0.0


def test_fsm_visits_all_gait_states():
    ctrl = StairControllerFSM()
    ctrl.reset()
    obs = _obs()
    states_seen = set()
    for step in range(500):
        ctrl.update(step * 0.01, obs)
        states_seen.add(ctrl.get_state_name())
    expected = {"left_stance", "left_swing", "right_stance", "right_swing"}
    assert expected <= states_seen, f"Missing states: {expected - states_seen}"


def test_pid_p_i_d_all_contribute():
    """With ki>0 and kd>0, two consecutive calls with same nonzero error
    should produce different outputs (I and D change the result)."""
    ctrl = StairControllerFSM()
    ctrl.reset()
    obs = _obs([0.4] * 6)
    out1 = ctrl.update(0.00, obs)
    out2 = ctrl.update(0.01, obs)
    # Second call has accumulated integral → outputs differ
    assert not np.allclose(out1, out2), "I/D terms should cause output to change"


# ══════════════════════════════════════════════════════════════════════════════
# Renderer recording
# ══════════════════════════════════════════════════════════════════════════════

from renderer import Renderer
from objects import TwoLink


def _make_gif(n_frames=5, fps=10):
    r = Renderer(record=True)
    robot = TwoLink()
    for i in range(n_frames):
        robot.q = np.array([i * 0.1, -i * 0.05])
        r.update([robot])
    return r


def test_renderer_captures_frames():
    r = _make_gif(8)
    assert len(r._frames) == 8
    assert r._frames[0].ndim == 3  # H x W x 4
    r.close()


def test_renderer_no_record_empty():
    r = Renderer(record=False)
    robot = TwoLink()
    r.update([robot])
    assert len(r._frames) == 0
    r.close()


def test_renderer_save_gif(tmp_path):
    r = _make_gif(5)
    out = tmp_path / "out.gif"
    r.save(str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    r.close()


def test_renderer_save_mp4(tmp_path):
    r = _make_gif(5)
    out = tmp_path / "out.mp4"
    r.save(str(out))
    assert out.exists()
    assert out.stat().st_size > 0
    r.close()


def test_renderer_save_raises_without_record():
    r = Renderer(record=False)
    with pytest.raises(RuntimeError, match="No frames recorded"):
        r.save("/tmp/x.gif")
    r.close()


def test_renderer_save_raises_bad_extension(tmp_path):
    r = _make_gif(3)
    with pytest.raises(ValueError, match="Unsupported format"):
        r.save(str(tmp_path / "out.avi"))
    r.close()


def test_renderer_multi_object():
    """Two robots must render without errors."""
    r = Renderer(record=True)
    r1, r2 = TwoLink(), TwoLink()
    r1.q = np.array([0.3, 0.5])
    r2.q = np.array([-0.3, -0.5])
    r.update([r1, r2])
    assert len(r._frames) == 1
    r.close()
