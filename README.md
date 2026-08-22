# Autonomous Desk-Sorting Robotic Arm — Phase A



https://github.com/user-attachments/assets/2be9d9cc-dfaf-4284-88d5-a9f24a1ce587



[![Tests](https://github.com/imtiyazMohammed/desk-sorting-arm/actions/workflows/tests.yml/badge.svg)](https://github.com/imtiyazMohammed/desk-sorting-arm/actions/workflows/tests.yml)

Simulation-only kinematics stack. No hardware required.

**Requires Python 3.10-3.14.** `build123d` (Phase D parametric CAD) declares
`requires-python ">=3.10,<3.15"`, so the Python 3.9 support present through
Phase A has been dropped. Raspberry Pi OS Bookworm ships 3.11 and Trixie ships
3.13, so 3.9 is not needed for the deployment target. CI covers 3.10-3.13.

## Quickstart

```bash
# 1. Install dependencies
python3 -m pip install -r requirements.txt

# 2. Run the test suite (32 tests, ~2 seconds)
python3 -m pytest tests/ -v

# 3. Try the module smoke tests
python3 -m src.geometry              # geometry summary
python3 -m src.forward_kinematics    # FK sanity checks
python3 -m src.inverse_kinematics    # IK sample solves
python3 -m src.trajectory            # trajectory profiling

# 4. Read the proof-of-concept document
docs/PROOF_OF_CONCEPT.md
```

## Repository layout

```
desk_arm/
├── src/
│   ├── geometry.py             # ArmGeometry (single source of truth)
│   ├── forward_kinematics.py   # Hand-derived FK using homogeneous transforms
│   ├── arm_chain.py            # ikpy Chain constructor
│   ├── inverse_kinematics.py   # IK with reachability + singularity checks
│   ├── trajectory.py           # Trapezoidal velocity profiler
│   └── visualizer.py           # 3D matplotlib rendering
├── tests/
│   └── test_kinematics.py      # 32 tests covering all modules
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
| Servos | DS3218 (5×) | 20 kg·cm digital metal-gear |

## Known outstanding items (deferred to later phases)

- **Shoulder torque budget**: 950 mm arm exceeds DS3218 20 kg·cm rating. To be resolved in Phase C via 2:1 gear reduction or dual-servo shoulder.
- **Singularity threshold**: Currently unit-inconsistent (mm-scaled Jacobian). To be normalized before deployment.
- **Orientation IK**: Position-only solves for now. Full pose IK (with grasp orientation) belongs to the grasp planner in Phase E.
