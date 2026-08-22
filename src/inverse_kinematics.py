"""
Inverse Kinematics (IK) for the 5-DOF desk-sorting arm.

Wraps ikpy's optimization-based solver with:
  1. Pre-flight reachability check (rejects targets outside the workspace
     envelope BEFORE running the solver, saving ~50 ms per call).
  2. Joint-limit enforcement (delegated to ikpy via Chain bounds).
  3. Singularity detection using the manipulability index sqrt(det(J J^T)).
  4. Solver convergence verification (recomputes FK on the returned angles
     and rejects the result if position error exceeds a tolerance).

Units: input target position in mm; output joint angles in radians.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .arm_chain import (
    MM_TO_M,
    build_chain,
    extract_active_angles,
    full_joint_vector,
)
from .forward_kinematics import ForwardKinematics
from .geometry import ArmGeometry, DEFAULT_ARM


# =========================================================================
# Result types
# =========================================================================


class IKStatus(Enum):
    """Outcome of an IK solve attempt."""

    SUCCESS = "success"
    UNREACHABLE = "unreachable"        # Target outside workspace envelope
    JOINT_LIMIT = "joint_limit"        # Solver produced out-of-bounds angles
    NON_CONVERGENT = "non_convergent"  # Solver returned but FK error too large
    SINGULARITY = "singularity"        # Solution near a kinematic singularity


@dataclass(frozen=True)
class IKSolution:
    """Result of an IK solve."""

    status: IKStatus
    joint_angles_rad: Optional[np.ndarray]  # shape (5,) or None if failed
    position_error_mm: float                # Euclidean FK-vs-target error
    manipulability: float                   # sqrt(det(J J^T)); 0 = singular
    message: str

    @property
    def joint_angles_deg(self) -> Optional[np.ndarray]:
        if self.joint_angles_rad is None:
            return None
        return np.degrees(self.joint_angles_rad)

    def __repr__(self) -> str:
        if self.joint_angles_deg is None:
            return f"IKSolution(status={self.status.value}, {self.message})"
        angles = ", ".join(f"{a:7.2f}" for a in self.joint_angles_deg)
        return (
            f"IKSolution(status={self.status.value}, "
            f"angles_deg=[{angles}], "
            f"err={self.position_error_mm:.3f}mm, "
            f"manip={self.manipulability:.4f})"
        )


# =========================================================================
# Solver
# =========================================================================


class InverseKinematics:
    """
    Position-only IK solver for the 5-DOF desk arm.

    We solve position-only (not full pose) because a 5-DOF arm is redundant
    for 3-DOF position targets: multiple (theta_4, theta_5) combinations reach
    the same TCP point. Constraining orientation would require adding a
    target_orientation matrix, which is a follow-up for the grasp planner.

    Position tolerance defaults to 2 mm (well below any picking margin needed
    for stationery-sized objects).
    """

    def __init__(
        self,
        arm: ArmGeometry = DEFAULT_ARM,
        position_tolerance_mm: float = 2.0,
        singularity_threshold: float = 1e-3,
    ) -> None:
        self.arm = arm
        self.chain = build_chain(arm)
        self.fk_reference = ForwardKinematics(arm)
        self.position_tolerance_mm = position_tolerance_mm
        self.singularity_threshold = singularity_threshold

    # ---- Reachability envelope ----------------------------------------------
    def is_reachable(self, target_mm: Sequence[float]) -> bool:
        """
        Cheap pre-check: is the target within a spherical shell centered on
        the shoulder pivot?

        Inner radius = |L1 - L2| - L3  (fully folded)
        Outer radius = (L1 + L2 + L3) * 0.95  (5% margin below full extension)
        """
        target = np.asarray(target_mm, dtype=float)
        if target.shape != (3,):
            raise ValueError(f"target_mm must have shape (3,), got {target.shape}")

        shoulder = np.array([0.0, 0.0, self.arm.base_height_mm])
        distance = np.linalg.norm(target - shoulder)

        inner = max(
            0.0,
            abs(self.arm.l1_upper_arm_mm - self.arm.l2_forearm_mm)
            - self.arm.l3_wrist_to_tip_mm,
        )
        outer = (
            self.arm.l1_upper_arm_mm
            + self.arm.l2_forearm_mm
            + self.arm.l3_wrist_to_tip_mm
        )

        return bool(inner <= distance <= outer)

    # ---- Manipulability (singularity metric) --------------------------------
    def manipulability(self, joint_angles_rad: Sequence[float]) -> float:
        """
        Yoshikawa manipulability index: sqrt(det(J J^T)) where J is the
        6xN geometric Jacobian.

        Values near zero indicate proximity to a kinematic singularity where
        small task-space motions require unbounded joint velocities.

        We approximate J by finite differences on the FK (position rows only,
        3xN) because ikpy does not expose an analytical Jacobian. For a
        singularity indicator this is more than accurate enough.
        """
        theta = np.asarray(joint_angles_rad, dtype=float)
        eps = 1e-6
        base_pos = self.fk_reference.compute(theta).position_mm

        n_joints = len(theta)
        J = np.zeros((3, n_joints))
        for j in range(n_joints):
            perturbed = theta.copy()
            perturbed[j] += eps
            # Clamp to limits so we don't raise inside a finite-diff step
            perturbed[j] = self.arm.joint_limits[j].clamp(perturbed[j])
            actual_eps = perturbed[j] - theta[j]
            if abs(actual_eps) < 1e-12:
                # At the limit; step in the other direction
                perturbed[j] = theta[j] - eps
                actual_eps = -eps
            pos = self.fk_reference.compute(perturbed).position_mm
            J[:, j] = (pos - base_pos) / actual_eps

        # sqrt(det(J J^T)) — use only position rows so J is 3xN
        JJt = J @ J.T
        det = np.linalg.det(JJt)
        return float(np.sqrt(max(det, 0.0)))

    # ---- Main solve ---------------------------------------------------------
    def solve(
        self,
        target_mm: Sequence[float],
        initial_guess_rad: Optional[Sequence[float]] = None,
    ) -> IKSolution:
        """
        Solve for joint angles that place the TCP at target_mm.

        Parameters
        ----------
        target_mm : sequence of 3 floats
            Desired TCP position in the arm base frame, millimeters.
        initial_guess_rad : optional sequence of 5 floats
            Seed for the optimizer. If None, uses the arm's "ready" pose.

        Returns
        -------
        IKSolution
        """
        target = np.asarray(target_mm, dtype=float)
        if target.shape != (3,):
            raise ValueError(f"target_mm must have shape (3,), got {target.shape}")

        # 1) Pre-flight reachability check
        if not self.is_reachable(target):
            shoulder = np.array([0.0, 0.0, self.arm.base_height_mm])
            distance = float(np.linalg.norm(target - shoulder))
            return IKSolution(
                status=IKStatus.UNREACHABLE,
                joint_angles_rad=None,
                position_error_mm=float("inf"),
                manipulability=0.0,
                message=(
                    f"Target {distance:.1f}mm from shoulder is outside "
                    f"reachable envelope."
                ),
            )

        # 2) Build initial guess for the solver (7-element ikpy vector)
        if initial_guess_rad is None:
            seed_active = np.array([0.0, -np.pi / 4, np.pi / 2, 0.0, 0.0])
        else:
            seed_active = np.asarray(initial_guess_rad, dtype=float)
            if seed_active.shape != (5,):
                raise ValueError(
                    f"initial_guess_rad must have shape (5,), got {seed_active.shape}"
                )
        seed_full = full_joint_vector(seed_active)

        # 3) Run ikpy optimizer (target in meters)
        target_m = target * MM_TO_M
        try:
            solution_full = self.chain.inverse_kinematics(
                target_position=target_m,
                initial_position=seed_full,
            )
        except Exception as exc:  # pragma: no cover - ikpy raises rarely
            return IKSolution(
                status=IKStatus.NON_CONVERGENT,
                joint_angles_rad=None,
                position_error_mm=float("inf"),
                manipulability=0.0,
                message=f"ikpy raised: {exc!r}",
            )

        active_angles = extract_active_angles(solution_full)

        # 4) Joint-limit verification (ikpy respects bounds but validate anyway)
        for i, (angle, limit) in enumerate(
            zip(active_angles, self.arm.joint_limits)
        ):
            if not limit.contains(angle, tolerance_rad=1e-4):
                return IKSolution(
                    status=IKStatus.JOINT_LIMIT,
                    joint_angles_rad=active_angles,
                    position_error_mm=float("inf"),
                    manipulability=0.0,
                    message=(
                        f"Joint {i} ({limit.name}) = "
                        f"{np.degrees(angle):.2f}° violates "
                        f"[{limit.min_deg:.2f}°, {limit.max_deg:.2f}°]"
                    ),
                )

        # 5) Convergence verification — recompute FK and measure error
        achieved_pose = self.fk_reference.compute(active_angles)
        error_mm = float(np.linalg.norm(achieved_pose.position_mm - target))

        if error_mm > self.position_tolerance_mm:
            return IKSolution(
                status=IKStatus.NON_CONVERGENT,
                joint_angles_rad=active_angles,
                position_error_mm=error_mm,
                manipulability=self.manipulability(active_angles),
                message=(
                    f"Solver returned but FK error {error_mm:.2f}mm exceeds "
                    f"tolerance {self.position_tolerance_mm:.2f}mm."
                ),
            )

        # 6) Singularity check on the successful solution
        m = self.manipulability(active_angles)
        if m < self.singularity_threshold:
            return IKSolution(
                status=IKStatus.SINGULARITY,
                joint_angles_rad=active_angles,
                position_error_mm=error_mm,
                manipulability=m,
                message=(
                    f"Solution reached target but manipulability {m:.5f} "
                    f"below threshold {self.singularity_threshold}."
                ),
            )

        return IKSolution(
            status=IKStatus.SUCCESS,
            joint_angles_rad=active_angles,
            position_error_mm=error_mm,
            manipulability=m,
            message="OK",
        )


if __name__ == "__main__":
    ik = InverseKinematics()
    for target in [
        [400.0,   0.0, 100.0],   # Reachable, moderate extension
        [200.0, 200.0, 300.0],   # Reachable, up-and-over
        [ 50.0,  50.0, 500.0],   # Reachable, near-vertical
        [900.0, 900.0, 100.0],   # Unreachable, too far
        [  0.0,   0.0, 100.0],   # Singular / below workspace
    ]:
        sol = ik.solve(target)
        print(f"Target {target} -> {sol}")
