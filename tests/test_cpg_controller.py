"""Unit tests for the CPG gait controller."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from examples.bipedal_stairs.control.cpg_controller import CPGController


@pytest.fixture
def ctrl():
    c = CPGController()
    c.reset()
    return c


def make_obs(x: float = 0.0, z: float = 0.09, vx: float = 0.0):
    qpos = np.zeros(9)
    qpos[0] = x
    qpos[1] = z
    qvel = np.zeros(9)
    qvel[0] = vx
    return {"qpos": qpos, "qvel": qvel}


class TestCPGOutput:
    def test_output_shape(self, ctrl):
        out = ctrl.update(0.0, make_obs())
        assert out.shape == (6,), "Must return 6 joint angles"

    def test_output_clipped(self, ctrl):
        for t in np.linspace(0, 5, 50):
            out = ctrl.update(float(t), make_obs(x=-t * 0.5))
            assert np.all(np.abs(out) <= 1.57 + 1e-6), "Angles must stay within ±π/2"

    def test_output_nonzero_after_warmup(self, ctrl):
        """Robot should be actively walking after a few steps."""
        for i in range(20):
            ctrl.update(i * 0.1, make_obs())
        out = ctrl.update(2.0, make_obs(x=-0.3))
        assert np.any(np.abs(out) > 0.05), "CPG should produce non-zero commands"


class TestStairFactor:
    def test_flat_ground(self, ctrl):
        ctrl.update(0.0, make_obs(x=0.0))
        assert ctrl._stair_factor() == pytest.approx(0.0)

    def test_mid_stair(self, ctrl):
        ctrl.update(0.0, make_obs(x=-1.5))   # 0.70 m past stair_start_x=0.80
        sf = ctrl._stair_factor()
        assert 0.0 < sf < ctrl.stair_count

    def test_beyond_top(self, ctrl):
        ctrl.update(0.0, make_obs(x=-10.0))
        assert ctrl._stair_factor() == pytest.approx(float(ctrl.stair_count))


class TestLandingTaper:
    def test_taper_is_one_on_stairs(self, ctrl):
        ctrl.update(0.0, make_obs(x=-1.5))
        assert ctrl._landing_taper() == pytest.approx(1.0)

    def test_taper_fades_on_platform(self, ctrl):
        landing_x = ctrl.stair_start_x + ctrl.stair_count * ctrl.stair_step_depth
        ctrl.update(0.0, make_obs(x=-(landing_x + 0.8)))   # 0.8 m into platform
        taper = ctrl._landing_taper()
        assert 0.0 < taper < 1.0

    def test_taper_zero_at_full_stop(self, ctrl):
        landing_x = ctrl.stair_start_x + ctrl.stair_count * ctrl.stair_step_depth
        ctrl.update(0.0, make_obs(x=-(landing_x + ctrl.landing_brake_dist + 1.0)))
        assert ctrl._landing_taper() == pytest.approx(0.0)


class TestReset:
    def test_reset_clears_phase(self, ctrl):
        ctrl.update(0.0, make_obs(x=-1.0))   # initialises t0
        ctrl.update(0.5, make_obs(x=-1.0))   # phase advances
        assert ctrl._phase != 0.0
        ctrl.reset()
        assert ctrl._phase == 0.0
        assert ctrl._t0 is None

    def test_state_name_after_reset(self, ctrl):
        assert ctrl.get_state_name() == "idle"
        ctrl.update(0.0, make_obs())
        ctrl.update(0.5, make_obs())
        assert ctrl.get_state_name() != "idle"
        ctrl.reset()
        assert ctrl.get_state_name() == "idle"
