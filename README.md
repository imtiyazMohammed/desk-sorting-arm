# Autonomous Desk-Sorting Robotic Arm — Phase A



https://github.com/user-attachments/assets/2be9d9cc-dfaf-4284-88d5-a9f24a1ce587



[![Tests](https://github.com/imtiyazMohammed/desk-sorting-arm/actions/workflows/tests.yml/badge.svg)](https://github.com/imtiyazMohammed/desk-sorting-arm/actions/workflows/tests.yml)

Kinematics simulation (Phase A), computer-vision foundation (Phase B), and
parametric CAD (Phase D). No hardware required to run anything in this
repository.

**Requires Python 3.10-3.14.** `build123d` (Phase D parametric CAD) declares
`requires-python ">=3.10,<3.15"`, so the Python 3.9 support present through
Phase A has been dropped. Raspberry Pi OS Bookworm ships 3.11 and Trixie ships
3.13, so 3.9 is not needed for the deployment target. CI covers 3.10-3.13.

## Quickstart

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run the test suite (375 tests, ~7 seconds)
python3 -m pytest tests/ -v

# 3. Try the module smoke tests
python3 -m src.geometry              # geometry + hardware summary
python3 -m src.forward_kinematics    # FK sanity checks
python3 -m src.inverse_kinematics    # IK sample solves
python3 -m src.trajectory            # trajectory profiling
python3 -m src.image_source          # synthetic camera frames
python3 -m src.camera_calibration    # synthetic calibration vs ground truth

# 4. Generate the CAD parts
python3 -m cad.base_pedestal         # pedestal + desk-clamp upper jaw
python3 -m cad.desk_clamp_lower_jaw  # clamp lower jaw (captive M8 nut)
python3 -m cad.desk_clamp_knob       # hand knob
#   ... each takes --report (dimensions only) and --output PATH

# 5. Manual hardware tools (need a webcam; not part of the test suite)
python3 scripts/preview_camera.py
python3 scripts/calibrate_camera.py
#   ... both take --synthetic to run without a camera

# 6. Read the proof-of-concept document
docs/PROOF_OF_CONCEPT.md
```

Heavy vision dependencies (`ultralytics`, which pulls in torch) are kept out of
`requirements.txt` until Session B.2 actually imports them:

```bash
python3 -m pip install -r requirements-vision.txt
```

## Repository layout

```
desk_arm/
├── src/
│   ├── geometry.py             # ArmGeometry + hardware specs (source of truth)
│   ├── forward_kinematics.py   # Hand-derived FK using homogeneous transforms
│   ├── arm_chain.py            # ikpy Chain constructor
│   ├── inverse_kinematics.py   # IK with reachability + singularity checks
│   ├── trajectory.py           # Trapezoidal velocity profiler
│   ├── visualizer.py           # 3D matplotlib rendering
│   ├── image_source.py         # ImageSource ABC + synthetic and webcam sources
│   └── camera_calibration.py   # Chessboard intrinsics + synthetic target
├── cad/
│   ├── base_pedestal.py        # Pedestal + desk-clamp upper jaw (build123d)
│   ├── desk_clamp_lower_jaw.py # Clamp lower jaw, captive M8 nut
│   ├── desk_clamp_knob.py      # Fluted hand knob
│   ├── _design.py              # Shared DesignStatus / DesignRuleError
│   ├── _primitives.py          # Shared solids (hex_prism)
│   ├── output/                 # Generated STLs
│   └── README.md               # Parametric approach, clamp design, assembly
├── scripts/
│   ├── preview_camera.py       # Manual webcam check, not run by CI
│   └── calibrate_camera.py     # Interactive intrinsic calibration
├── tests/
│   ├── test_kinematics.py      # 32 tests - Phase A kinematics
│   ├── test_image_source.py    # 71 tests - Session B.1 image acquisition
│   ├── test_camera_calibration.py  # 100 tests - Session B.2 calibration
│   └── test_cad.py             # 172 tests - Sessions D.1/D.1b parametric CAD
├── docs/
│   ├── PROOF_OF_CONCEPT.md     # Design + math + validation
│   └── figures/                # Generated diagrams
├── requirements.txt
├── requirements-vision.txt     # Deferred: ultralytics / torch
└── README.md
```

## Design constraints locked in Phase A

| Parameter | Value | Reason |
|-----------|-------|--------|
| DOF | 5 (base yaw, shoulder, elbow, wrist pitch, wrist roll) | Enables angled grasps + monitor cleaning |
| Mount position | Center of long desk edge (1200×600 mm desk) | Halves worst-case reach vs corner mount |
| L1 (upper arm) | 400 mm | |
| L2 (forearm) | 350 mm | |
| L3 (wrist → TCP) | 200 mm | |
| Base height | 100 mm | |
| Total reach | 950 mm | Covers 848.5 mm desk diagonal with 12% margin |
| Servos | DS3218 (5×) | 21.5 kg·cm at 6.8 V, digital metal-gear |
| Base pedestal | 70 mm tall, 85.3 mm body | `base_height_mm` 100 − 30 mm stack allowance |
| Desk mount | M8 edge clamp, 45 mm throat | No drilling; fits 15–35 mm desks, repositionable |

## Known outstanding items (deferred to later phases)

- **Shoulder torque budget**: 950 mm arm exceeds DS3218 20 kg·cm rating. To be resolved in Phase C via 2:1 gear reduction or dual-servo shoulder.
- **Singularity threshold**: Currently unit-inconsistent (mm-scaled Jacobian). To be normalized before deployment.
- **Orientation IK**: Position-only solves for now. Full pose IK (with grasp orientation) belongs to the grasp planner in Phase E.
- **Unverified servo dimensions**: The DS3218 body envelope is datasheet-confirmed, but its mounting-flange geometry and output-shaft placement are placeholders. See `ServoSpec.UNVERIFIED_FIELDS` and [cad/README.md](cad/README.md); measure a real unit before printing for final assembly.
- **Provisional base stack**: The 6 mm turntable plate and 24 mm shoulder bracket rise that set the pedestal height are estimates, replaced in Sessions D.2/D.3.
