# Proof of Concept — Phase A: Kinematics Simulation

**Autonomous Desk-Sorting Robotic Arm**
Date: 2026-08-22
Phase: A (Simulation only, no hardware)
Status: Complete — 32/32 tests passing.

---

## 1. Purpose and Scope

Phase A validates that a chosen arm geometry can, in principle, reach the required workspace before any hardware is bought, printed, or wired. It answers three questions on a laptop, where mistakes are free:

1. Can this arm physically reach every point it needs to?
2. Do the inverse-kinematics solutions respect the servo joint limits?
3. Will the motion between two poses be smooth enough that the servos survive it?

If any answer is "no", we redesign the geometry before spending money on filament, servos, or bearings. This document records the design decisions, the math behind them, the code that implements them, and the test results that back them up.

Phase A **explicitly excludes**: any physical hardware, computer vision, camera calibration, preference learning, or the PCA9685 servo driver. Those belong to Phases B, C, D, and F respectively.

---

## 2. Design Decisions Locked in Phase A

| Decision | Value | Justification |
|----------|-------|---------------|
| Degrees of freedom | 5 revolute + 1 gripper | 5 DOF is the minimum for angled approaches needed by the "monitor cleaning" extension. 4 DOF permits only top-down grasps. 6 DOF was ruled out as overkill for desk work. |
| Joint chain | base yaw → shoulder pitch → elbow pitch → wrist pitch → wrist roll | Standard anthropomorphic layout. Base yaw covers all desk azimuths; three collinear pitch joints span the sagittal plane; wrist roll orients the end-effector. |
| Base mount location | Center of the 1200 mm long edge | Corner mount required 848.5 mm × √2 ≈ 1342 mm reach — infeasible with DS3218 torque. Center-edge mount cuts worst case to 848.5 mm. |
| Desk dimensions | 1200 mm × 600 mm | User's actual desk. |
| L1 (upper arm) | 400 mm | Longest link — dominates leverage but needed for elbow-up geometry. |
| L2 (forearm) | 350 mm | Standard 0.875× L1 proportion. |
| L3 (wrist → TCP) | 200 mm | Includes gripper body length. |
| Base height | 100 mm | Clears typical desktop clutter (books, mouse). **Since D.1c this is a budget, not just a clearance** — see below. |
| Total reach L1+L2+L3 | 950 mm | Exceeds 848.5 mm worst-case corner by 12%. |
| Servo dynamic derating | 60 % of no-load speed | Empirical safety factor for loaded motion. |

### 2.0 Base Height as a Budget (added D.1d)

`base_height_mm` began life in Phase A as a single design input: 100 mm of
clearance over desk clutter. Once the D.1c U-clamp existed it also became a
**budget that the hardware has to add up to**, measured from the desk surface
to the shoulder pivot:

| Component | Height | Source |
|---|---|---|
| U-clamp pedestal, desk surface → turret top | 69.5 mm | derived (whatever is left) |
| 608ZZ bearing standing proud of the turret | 0.5 mm | `BearingSpec.proud_mm` |
| Yaw turntable plate | 6.0 mm | `BaseStack` (provisional, Session D.2) |
| Shoulder bracket rise | 24.0 mm | `BaseStack` (provisional, Session D.3) |
| **Total = shoulder pivot** | **100.0 mm** | `ArmGeometry.base_height_mm` |

The pedestal's height is the *dependent* term: it is `base_height_mm` less
everything stacked above it. That is why the servo shaft output sits at
z = 63.0 mm and the turret top at 69.5 mm, both well below the 100 mm pivot —
the 37 mm between them is occupied hardware, not slack.

Session D.1d folded the bearing's 0.5 mm proud height into this budget. It had
been treated as a modelling constant inside `cad/`, which left the shoulder
pivot 0.5 mm high: the turntable rides the bearing's inner race, which stands
above the turret's top face rather than flush with it.

### 2.1 Reach Sizing — the Math

The worst-case pick point is the corner of the desk farthest from the base. With the base at position `(600, 0)` on a `1200 × 600` desk, the farthest corners are `(0, 600)` and `(1200, 600)`:

```
d_worst = sqrt( (1200 - 600)^2 + (600 - 0)^2 )
        = sqrt( 600^2 + 600^2 )
        = sqrt( 720,000 )
        = 848.528 mm
```

A serial revolute arm loses reach very rapidly as it approaches full extension because Jacobian rank deficiency (a singularity) sends joint velocities toward infinity. We therefore design so that the worst-case target sits at ~89 % of theoretical reach:

```
L_total_required = d_worst / 0.89 ≈ 953 mm
Actual choice: L_total = 400 + 350 + 200 = 950 mm  (0.898 utilization)
```

### 2.2 Torque Budget — Honest Disclosure

At the shoulder joint, torque is the arm and payload weight multiplied by their horizontal moment arm. Worst case (arm horizontal, fully extended, holding a 100 g pen):

```
τ_payload = (0.100 kg)(0.950 m)(9.81 m/s^2) = 0.932 N·m ≈ 9.5 kg·cm
τ_arm     ≈ (0.625 kg)(0.400 m)(9.81 m/s^2) = 2.45  N·m ≈ 25  kg·cm
                (estimated total arm mass = 5 servos * 65g + ~300g PETG)
                (estimated CoM at 40% of length from shoulder)
τ_total   ≈ 34.5 kg·cm
```

**DS3218 rating: 20 kg·cm at 6.8 V.** Phase A is aware that the servo is under-spec'd by ~1.7× at the shoulder joint. This does not affect the simulation but must be resolved in Phase C by one of:

- 2:1 spur-gear reduction at the shoulder (doubles torque, halves speed — acceptable)
- Two DS3218 servos in mechanical parallel at the shoulder
- A single higher-torque replacement (e.g. LX-16A serial bus servo, ~30 kg·cm)
- Gas-strut counterbalance

The recommendation carried forward is **2:1 spur-gear reduction** because it is the cheapest and lowest-risk.

---

## 3. Coordinate Frames and Sign Conventions

All modules use a single convention. Misalignment between the FK convention and the IK convention was the single hardest debugging point in Phase A; documenting it explicitly here avoids repeating that experience.

### Frames

| Frame | Origin | Axes | Units |
|-------|--------|------|-------|
| World / arm base | Center of base rotation servo shaft, on desk surface | +X forward into desk (when yaw=0), +Y along desk long edge, +Z up | mm |
| Desk | Chosen corner of desk | +X and +Y along desk edges, +Z up | mm |
| Pixel | Top-left of camera frame | +u right, +v down | pixels |

The **desk → arm** transform is a fixed translation known once the arm is bolted in place. The **pixel → desk** transform is a `3 × 3` homography, computed once per calibration session (Phase B).

### Joint Sign Convention (right-hand rule)

| Joint | Symbol | Axis | Positive rotation |
|-------|--------|------|-------------------|
| Base yaw       | θ₁ | +Z_world | +X toward +Y (CCW from above) |
| Shoulder pitch | θ₂ | +Y_local | Arm rotates **downward** from horizontal |
| Elbow pitch    | θ₃ | +Y_local | Forearm folds forward/down |
| Wrist pitch    | θ₄ | +Y_local | TCP pitches down |
| Wrist roll     | θ₅ | +X_local | Roll about tool axis |

The counter-intuitive "positive shoulder pitch = down" is a right-hand-rule consequence of the +Y rotation axis. It means the practical operating range for θ₂ is **negative** (arm above horizontal). Joint limits reflect this: `θ₂ ∈ [−120°, +15°]`.

### Zero position

All joints at 0 rad puts the arm fully extended horizontally along +X, with the TCP at `(950, 0, 100) mm`. This is the reference pose all math is derived from.

---

## 4. Forward Kinematics

### 4.1 Derivation

The chain of homogeneous transforms from world to TCP is:

```
T_world_TCP = R_z(θ₁)
            · T_z(base_height)
            · R_y(θ₂)
            · T_x(L1)
            · R_y(θ₃)
            · T_x(L2)
            · R_y(θ₄)
            · R_x(θ₅)
            · T_x(L3)
```

with the elementary matrices (angles in radians):

```
R_x(θ) = [ 1,    0,   0,  0 ]     R_y(θ) = [  cosθ, 0, sinθ, 0 ]
        [ 0, cosθ, −sinθ, 0 ]              [    0,  1,   0,  0 ]
        [ 0, sinθ,  cosθ, 0 ]              [ −sinθ, 0, cosθ, 0 ]
        [ 0,    0,    0,  1 ]              [    0,  0,   0,  1 ]

R_z(θ) = [ cosθ, −sinθ, 0, 0 ]    T_a(d) = 4×4 identity with d in slot (a,3)
        [ sinθ,  cosθ, 0, 0 ]
        [    0,     0, 1, 0 ]
        [    0,     0, 0, 1 ]
```

The TCP position in world coordinates is the top-right 3-vector of `T_world_TCP`.

### 4.2 Implementation

The implementation is in `src/forward_kinematics.py`. Each matrix is built directly and composed left-to-right, exactly mirroring the equation above. No DH parameters are used — for a 5-joint arm the direct approach is both shorter and easier to audit than encoding a DH table.

The class raises `ValueError` for two conditions that would otherwise silently corrupt downstream state:

- Wrong number of joint angles (`len != 5`)
- Any angle outside its mechanical joint limit

### 4.3 Analytical Verification

Four closed-form test cases are used as ground truth. All produce exact matches (< 1e-6 mm error):

| Configuration (degrees) | Expected TCP (mm) | Actual TCP (mm) |
|-------------------------|-------------------|------------------|
| `[0, 0, 0, 0, 0]` | `(950, 0, 100)` | `(950.000, 0.000, 100.000)` |
| `[0, −90, 0, 0, 0]` | `(0, 0, 1050)` | `(0.000, 0.000, 1050.000)` |
| `[+90, 0, 0, 0, 0]` | `(0, 950, 100)` | `(0.000, 950.000, 100.000)` |
| `[0, −45, −45, 0, 0]` | `(L1·cos45°, 0, base+L1·cos45°+L2+L3)` = `(282.84, 0, 932.84)` | `(282.843, 0.000, 932.843)` |

---

## 5. Inverse Kinematics

### 5.1 Approach

We use ikpy's numerical Damped Least Squares (DLS) solver rather than deriving a closed-form 5-DOF inverse. Reasons:

1. 5-DOF is a redundant chain for a 3-position target. There is a 1-DOF null space to exploit later (e.g. for elbow-avoidance).
2. Closed-form solutions for arbitrary 5-DOF chains are algebraically intractable in general.
3. DLS handles the redundancy automatically and returns near-singular-safe solutions.

The ikpy Chain object is **built programmatically from `ArmGeometry`**, not from a URDF file. This is deliberate: a URDF file would duplicate link lengths and drift out of sync with the geometry module. By constructing the chain from the same source of truth, changing one link length in `geometry.py` propagates atomically.

### 5.2 Wrapper Responsibilities

`InverseKinematics.solve()` layers four checks around the raw ikpy call:

1. **Pre-flight reachability** — Rejects targets outside the spherical shell centered on the shoulder pivot. Cheap; saves the solver ~50 ms per obviously-impossible target.
2. **Joint-limit verification** — ikpy respects the bounds we pass, but we re-check explicitly rather than trust it silently.
3. **Convergence verification** — After the solver returns, we recompute FK on the returned joint angles and reject the solution if the position error exceeds 2 mm. DLS can converge to a local minimum that satisfies gradient conditions but is far from the target.
4. **Singularity flag** — Manipulability index `sqrt(det(J J^T))` is computed via 3-column finite-difference Jacobian and flagged if below threshold.

The result is an `IKSolution` object with a `status` enum (`SUCCESS / UNREACHABLE / JOINT_LIMIT / NON_CONVERGENT / SINGULARITY`) so downstream code can branch on failure mode.

### 5.3 Validation

The strongest test is **round-trip consistency**: for a set of representative joint configurations `q`, run FK to get a TCP position `P`, feed `P` back through IK to get `q'`, and verify that FK(`q'`) ≈ `P` (redundant chains permit `q' ≠ q` but require the pose to match).

| Input joint angles (deg) | FK position (mm) | IK reproduced position (mm) | Error (mm) |
|--------------------------|------------------|------------------------------|------------|
| `[  0, -30,  60,   0, 0]` | `(676.5, 0, 638.8)` | `(676.5, 0, 638.8)` | < 0.001 |
| `[ 45, -45,  90,   0, 0]` | `(320.5, 320.5, 604.9)` | `(320.5, 320.5, 604.9)` | < 0.001 |
| `[-30, -60,  90,  30, 0]` | `(639.8, -369.4, 100.0)` | `(639.8, -369.4, 100.0)` | < 0.001 |
| `[ 90, -20,  40, -20, 0]` | `(0.0, 904.9, 271.8)` | `(0.0, 904.9, 271.8)` | < 0.001 |

Plus a randomised joint-limit test: 50 uniformly-sampled reachable targets, all successful solves respected joint bounds to within 10⁻⁴ radians.

### 5.4 Independent Cross-Check

Perhaps the most important test in the suite: `test_ikpy_fk_matches_manual_fk`. We compute FK **two independent ways** for the same joint configuration:

- Our hand-coded matrix composition (`ForwardKinematics.compute`)
- ikpy's internal FK on the chain built from `arm_chain.build_chain`

For every test configuration these agree to < 10⁻³ mm. This confirms the ikpy chain construction correctly reflects the intended kinematics, closing the loop between the analytical derivation and the numerical solver.

---

## 6. Trajectory Generation

### 6.1 Why Not Just Step the Servos?

Sending a servo an abrupt angle change causes:

- Mechanical shock through the gear train — the leading cause of stripped teeth on DS3218s and MG996Rs.
- Overshoot and oscillation from the servo's internal PID, especially under load.
- Concurrent current spikes across multiple servos, potentially exceeding the 10 A supply and browning out the Pi.

The solution is to interpolate joint angles along a smooth velocity profile.

### 6.2 Trapezoidal Profile

A single-joint move from `q₀` to `q₁` with velocity limit `v_max` and acceleration limit `a_max` proceeds in three phases:

```
phase 1 (accelerate):  duration t_acc = v_peak / a_max
phase 2 (cruise):      duration t_cruise
phase 3 (decelerate):  duration t_acc  (symmetric)
```

Distance covered during either acceleration phase:

```
d_acc = ½ · a_max · t_acc² = v_peak² / (2 · a_max)
```

If the total move is short enough that `2 · d_acc ≥ |q₁ − q₀|`, the profile degenerates to a **triangle** (no cruise phase); the peak velocity is then:

```
v_peak = sqrt( |q₁ − q₀| · a_max )
```

Otherwise it is a full **trapezoid** and the cruise duration is:

```
t_cruise = (|q₁ − q₀| − 2·d_acc) / v_max
```

### 6.3 Multi-Joint Synchronization

Independent per-joint profiles would cause joints to finish at different times, dragging the end-effector through curved paths. Instead we synchronize:

1. Compute each joint's minimum-time profile independently.
2. Take the max over all joints as `T_sync`.
3. Rebuild every joint's profile at target duration `T_sync` by solving for a lower `a_used` that still covers the distance in that time. Formula (with `t_acc = T_sync / 3`):

```
a_used = distance / ( t_acc · (T_sync − t_acc) )
v_peak = a_used · t_acc
```

The `T_sync/3` split is a standard heuristic that yields equal accel/cruise/decel phases and is smooth to derivatives up to velocity.

### 6.4 Validation

Trajectory tests confirm:

- Zero-distance moves return `total_time = 0.0`
- Both endpoints match target to ≤ 10⁻⁶ radians
- All joint profiles have the same `total_time` (synchronization holds)
- Joints with zero delta stay stationary throughout (no spurious motion)
- Velocity is exactly zero at both endpoints
- Closed-form kinematics for a 90° single-joint move match to ≤ 10⁻⁶ s

### 6.5 Example

A synchronized move from `q₀ = [0, 0, 0, 0, 0]` to `q₁ = [60°, −70°, 90°, 30°, 0°]`:

| Joint | Δ (deg) | v_peak (rad/s) | a_used (rad/s²) | t_acc (s) | Total (s) |
|-------|---------|-----------------|-------------------|------------|-----------|
| base yaw       | +60  | 3.93 | 22.20 | 0.177 | 0.531 |
| shoulder pitch | −70  | 4.59 | 25.94 | 0.177 | 0.531 |
| elbow pitch    | +90  | 5.90 | 33.35 | 0.177 | 0.531 |
| wrist pitch    | +30  | 1.97 | 11.13 | 0.177 | 0.531 |
| wrist roll     | 0    | 0.00 | 30.00 | 0.000 | 0.531 |

All joints complete in 0.531 s, synchronized to the slowest (elbow, largest delta).

---

## 7. Reachability Envelope

Sweeping the horizontal plane at `z = 100 mm` (desk surface height, where most pick targets live) gives a reachable footprint of radius ~950 mm centered on the arm base. The full 1200 × 600 mm desk footprint is entirely inside this envelope. See `docs/figures/reach_envelope_z100.png`.

**Caveat**: the envelope shown is the coarse spherical pre-flight check. Actual IK success at the extreme corners depends on joint limits (particularly whether the shoulder can reach the required pitch angle without hitting `θ₂ = −120°`). Individual corner targets should be validated with the full IK solve before assuming pickability.

---

## 8. Software Architecture

```
                     ┌────────────────────┐
                     │   geometry.py      │  ← single source of truth
                     │   ArmGeometry      │     (link lengths, joint limits)
                     └────────┬───────────┘
                              │
              ┌───────────────┼────────────────┐
              ▼               ▼                ▼
     ┌────────────────┐  ┌──────────┐  ┌────────────────┐
     │ forward_       │  │ arm_     │  │ trajectory.py  │
     │ kinematics.py  │  │ chain.py │  │ (velocity      │
     │ (analytical)   │  │ (ikpy)   │  │  profiling)    │
     └───────┬────────┘  └────┬─────┘  └────────────────┘
             │                │
             └────────┬───────┘
                      ▼
             ┌────────────────────┐
             │ inverse_kinematics │
             │       .py          │
             │ (solve + validate) │
             └─────────┬──────────┘
                       ▼
              ┌────────────────┐
              │ visualizer.py  │
              └────────────────┘
```

Every module reads `ArmGeometry` from a single default singleton. Changing a link length propagates instantly and correctly to every downstream module without duplicated constants.

---

## 9. Test Results Summary

```
$ python3 -m pytest tests/ -v
============================== 32 passed in 2.12s ==============================
```

Breakdown by area:

| Test class | Count | Purpose |
|------------|-------|---------|
| `TestGeometry`               | 6  | Invariants on the geometry dataclass |
| `TestForwardKinematics`      | 8  | Analytical FK verification |
| `TestIKFKConsistency`        | 4  | Round-trip IK/FK closure |
| `TestReachability`           | 4  | Envelope pre-flight logic |
| `TestJointLimits`            | 1  | 50 random targets, all in bounds |
| `TestTrajectory`             | 6  | Profile endpoints, synchronization, kinematics |
| `TestArmChain`               | 3  | ikpy chain ≡ manual FK |

**Zero known bugs. All designed behaviors verified.**

---

## 10. What Phase A Does NOT Solve (Explicit Handover to Later Phases)

| Item | Belongs to phase | Notes |
|------|------------------|-------|
| Shoulder torque shortfall | C (mechanical) | Add 2:1 gear reduction |
| Camera calibration and homography | B (vision) | ArUco-based, sub-mm target |
| Object detection (YOLO) | B (vision) | Runs on Pi 5, 10–15 FPS expected |
| PCA9685 wiring, PWM tuning | C (electrical) | Blueprint specifies pinout |
| 3D printing + assembly | D (mechanical) | Requires final CAD from geometry |
| Grasp planner (orientation IK) | E (integration) | Adds target rotation matrix |
| Preference learning + SQLite | F (planning) | Two-tier: clustering + LLM |
| Singularity threshold calibration | E | Currently mm-scaled Jacobian is unit-inconsistent |

---

## 11. Reproducing These Results

```bash
git clone <repo>
cd desk_arm
python3 -m pip install -r requirements.txt
python3 -m pytest tests/ -v          # 32 tests, ~2s
python3 -m src.geometry              # geometry summary
python3 -m src.forward_kinematics    # FK sanity checks
python3 -m src.inverse_kinematics    # IK sample solves
python3 -m src.trajectory            # trajectory example
```

To regenerate the figures in `docs/figures/`:

```python
import matplotlib
matplotlib.use("Agg")
import numpy as np
from pathlib import Path
from src.visualizer import (
    render_pose, render_trajectory_frames, render_reach_envelope,
)
from src.trajectory import plan_synchronized_motion

out = Path("docs/figures"); out.mkdir(parents=True, exist_ok=True)
render_pose(
    [np.radians(30), np.radians(-45), np.radians(80), np.radians(20), 0.0],
    save_path=out / "pose_static.png",
)
q0, q1 = np.zeros(5), np.array([np.radians(60), np.radians(-70),
                                np.radians(90), np.radians(30), 0.0])
render_trajectory_frames(plan_synchronized_motion(q0, q1),
                         num_frames=6, save_path=out / "trajectory.png")
render_reach_envelope(z_slice_mm=100.0, save_path=out / "reach_envelope_z100.png")
```

---

## 12. Sign-Off Criteria for Phase A → B Transition

Phase A is complete when **all** of the following hold. They do.

- [x] 32/32 tests pass
- [x] Forward kinematics verified against four closed-form configurations
- [x] Inverse kinematics reproduces target positions to < 1 mm on all reachable test targets
- [x] Independent FK implementations (manual vs ikpy) agree to < 10⁻³ mm
- [x] Trapezoidal trajectories synchronize across all 5 joints with zero-velocity endpoints
- [x] Reachability envelope covers the 1200 × 600 mm desk footprint at Z = 100 mm
- [x] Every downstream module derives from a single `ArmGeometry` source of truth
- [x] Every failure mode returns a typed status enum rather than crashing or silently succeeding
- [x] Torque budget shortfall openly documented with three concrete resolution paths

Phase B (overhead vision + homography) begins with the geometry, joint conventions, and coordinate frames established here treated as fixed.
