# Bipedal Stair-Climbing — MuJoCo + ZeroMQ

A 2-D bipedal robot that walks on flat ground and climbs a 5-step staircase,
simulated in **MuJoCo 3** and controlled by a **Central Pattern Generator (CPG)**
over a **ZeroMQ** message bus.

![simulation](examples/bipedal_stairs/sim/preview.gif)

---

## Assignment checklist

| Requirement | Status | Details |
|---|---|---|
| Simulator scene | ✅ | MuJoCo 3, `bipedal_robot.xml` |
| 3 DOF frozen → 2-D system | ✅ | `slide_y`, `hinge_x`, `hinge_z` absent; only `slide_x`, `slide_z`, `hinge_y` |
| Non-PD/PID controller | ✅ | CPG (Central Pattern Generator) with torso-pitch feedback |
| External communication protocol | ✅ | ZeroMQ PUSH/PULL over TCP |
| Repository structure | ✅ | `pyproject.toml`, CI, tests, pre-commit |

---

## Architecture

```
┌─────────────────────────────┐        ZeroMQ TCP        ┌────────────────────────────┐
│        SimulatorNode        │                          │      ControllerNode        │
│      (sim/sim_node.py)      │                          │   (control/ctrl_node.py)   │
│                             │                          │                            │
│  MuJoCo physics (500 Hz)    │  ──obs tcp://:5555──►   │  CPG controller            │
│                             │  ◄──cmd tcp://:5556──   │  + torso-pitch feedback    │
└─────────────────────────────┘                          └────────────────────────────┘
```

**Message format (JSON over ZeroMQ)**

| Direction | Port | Payload |
|---|---|---|
| sim → ctrl | 5555 | `{"t": float, "qpos": [9], "qvel": [9]}` |
| ctrl → sim | 5556 | `{"ctrl": [6]}` |

The nodes can run in the **same process** (threads) or as **separate OS processes**
on the same machine or over a network — just change `OBS_ADDR` / `CMD_ADDR`
in `communication/zmq_transport.py`.

---

## Robot model

```
torso  (slide_x, slide_z, hinge_y)
├── left_hip  → left_knee  → left_ankle   [hip, knee, ankle pitch]
└── right_hip → right_knee → right_ankle  [hip, knee, ankle pitch]
```

**Degrees of freedom**: 3 rigid-body DOF + 6 joint DOF = 9 total  
**2-D constraint**: Y translation and X/Z rotations are absent from the model,
so the robot moves only in the X–Z plane.

**Staircase**: 5 steps, tread = 0.42 m, riser = 0.13 m → total rise = 0.65 m

---

## Controller: CPG (Central Pattern Generator)

The CPG produces smooth, rhythmic joint commands without any PID loop:

```
phase += 2π · freq · dt          # accumulating phase (continuous clock)

left_hip  =  A · sin(phase)      # anti-phase legs
right_hip = -A · sin(phase)

left_knee  = -(b + C · max(0, cos(phase))^1.2)   # clearance at liftoff
right_knee = -(b + C · max(0, -cos(phase))^1.2)

left_ankle  =  D · sin(phase + φ)
right_ankle = -D · sin(phase + φ)
```

**Adaptation on stairs**

| Parameter | Flat ground | Stairs (step 5) |
|---|---|---|
| Frequency | 0.62 Hz | 0.48 Hz |
| Hip amplitude A | 0.52 rad | 0.66 rad |
| Ankle amplitude D | 0.12 rad | 0.29 rad |
| Knee clearance C | 0.46 rad | 0.64 rad |

The ankle amplitude is blended from flat (0.12 rad — nearly flat foot) to stair
propulsion (0.22 + 0.07·sf₁ rad) as the robot enters the staircase.  This gives
deliberate, visible foot placement on each step (~1 s/step) instead of rushing
through at high speed.

**Stability features**
- *Torso-pitch feedback*: `Δhip = 0.5 · θ_torso` — resists forward/backward lean
- *Velocity balance*: `Δhip = −0.25 · ẋ` — damps CoM oscillations
- *Landing taper*: all amplitudes fade to zero once the robot is on the platform
- *Stiff torso*: hinge_y stiffness=9000 Nm/rad, damping=160 Nm·s/rad (was 6000/120)

---

## Installation

```bash
# Clone / enter repo
git clone <repo-url> && cd bipedal-stairs

# Install with uv (recommended)
uv pip install ".[dev]"

# Or with pip
pip install ".[dev]"
```

**Dependencies**: `mujoco>=3.0`, `numpy`, `pyzmq`, `imageio[ffmpeg]`

---

## Running

### All-in-one (simulator + controller in one command)

```bash
python -m examples.bipedal_stairs.run             # 10 000 steps (20 s)
python -m examples.bipedal_stairs.run --steps 6000
```

### Two separate processes (shows external protocol clearly)

```bash
# Terminal 1 — controller (start first)
python -m examples.bipedal_stairs.control.ctrl_node

# Terminal 2 — simulator
python -m examples.bipedal_stairs.sim.sim_node --steps 6000
```

### Record video

```bash
xvfb-run python record_sim.py     # Linux headless
python record_sim.py              # Windows / macOS with display
```

Output: `simulation_mujoco.mp4` (1280×720, 10 s)

---

## Tests

```bash
pytest                   # all tests
pytest -v tests/         # verbose
```

Test coverage:

- `tests/test_cpg_controller.py` — output shape, joint limits, stair factor, landing taper, reset
- `tests/test_zmq_transport.py`  — ZeroMQ round-trip (single message, field correctness, timeout)

---

## Repository structure

```
bipedal-stairs/
├── examples/bipedal_stairs/
│   ├── run.py                        ← main entry point (two-thread mode)
│   ├── communication/
│   │   ├── zmq_transport.py          ← ZeroMQ PUSH/PULL layer  ★
│   │   └── layer.py                  ← legacy in-process queue (reference)
│   ├── sim/
│   │   ├── sim_node.py               ← simulator thread / process
│   │   ├── environment.py            ← MuJoCo environment wrapper
│   │   └── bipedal_robot.xml         ← robot + staircase model
│   └── control/
│       ├── ctrl_node.py              ← controller thread / process
│       ├── cpg_controller.py         ← CPG gait algorithm  ★
│       └── stair_controller.py       ← FSM reference (not used in default run)
├── tests/
│   ├── test_cpg_controller.py
│   └── test_zmq_transport.py
├── record_sim.py                     ← standalone video recorder
├── pyproject.toml
├── noxfile.py
└── .github/workflows/ci.yaml
```

---

## CI

GitHub Actions runs on every push/PR:
- Python 3.10, 3.11, 3.12
- `pytest tests/` (headless, no GPU required)
- `ruff check` + `ruff format --check`
