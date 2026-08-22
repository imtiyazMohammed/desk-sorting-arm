"""
Arm geometry definition — single source of truth for all downstream modules.

Every kinematics, planning, and visualization module MUST import ArmGeometry
from here rather than hardcoding link lengths. This guarantees that changing
a physical dimension propagates atomically across the entire codebase.

Coordinate convention
---------------------
- All lengths in millimeters (mm).
- All angles in radians internally; conversions to degrees are explicit.
- Arm base frame: origin at the center of the base rotation servo shaft,
  Z-axis pointing up, X-axis pointing "forward" into the desk work area
  when all joint angles are zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class JointLimit:
    """Mechanical joint limit in radians. Frozen so tests can trust invariants."""

    min_rad: float
    max_rad: float
    name: str

    def __post_init__(self) -> None:
        if self.min_rad >= self.max_rad:
            raise ValueError(
                f"Joint {self.name}: min_rad ({self.min_rad}) must be strictly "
                f"less than max_rad ({self.max_rad})."
            )

    def clamp(self, angle_rad: float) -> float:
        """Clamp an angle to the valid mechanical range."""
        return float(np.clip(angle_rad, self.min_rad, self.max_rad))

    def contains(self, angle_rad: float, tolerance_rad: float = 1e-6) -> bool:
        """True iff the angle is within joint limits (with numerical tolerance)."""
        return (
            self.min_rad - tolerance_rad
            <= angle_rad
            <= self.max_rad + tolerance_rad
        )

    @property
    def min_deg(self) -> float:
        return float(np.degrees(self.min_rad))

    @property
    def max_deg(self) -> float:
        return float(np.degrees(self.max_rad))


@dataclass(frozen=True)
class ArmGeometry:
    """
    Physical geometry of the 5-DOF desk-sorting arm.

    The kinematic chain (base to end-effector) is:
        Base yaw (Z) -> Shoulder pitch (Y) -> Elbow pitch (Y)
                     -> Wrist pitch (Y)    -> Wrist roll (X)

    All lengths in mm. All limits in radians.
    """

    # ---- Link lengths (mm) ---------------------------------------------------
    base_height_mm: float = 100.0   # Desk surface -> shoulder pivot
    l1_upper_arm_mm: float = 400.0  # Shoulder pivot -> elbow pivot
    l2_forearm_mm: float = 350.0    # Elbow pivot -> wrist pivot
    l3_wrist_to_tip_mm: float = 200.0  # Wrist pivot -> gripper tip (TCP)

    # ---- Desk workspace (mm) -------------------------------------------------
    desk_width_mm: float = 1200.0
    desk_depth_mm: float = 600.0
    # Base mount position on the desk (center of long edge).
    # Desk frame origin is one corner; base is placed on the near long edge,
    # centered along its length, offset "into" the desk by base_inset_mm.
    base_x_on_desk_mm: float = 600.0
    base_y_on_desk_mm: float = 0.0

    # ---- Joint limits --------------------------------------------------------
    # DS3218 servos physically travel ~270°. We restrict further to avoid
    # mechanical interference with adjacent links.
    # NOTE sign convention (see forward_kinematics.py docstring):
    #   +theta_2 rotates shoulder DOWN from horizontal. Therefore the practical
    #   operating range spans NEGATIVE angles (arm above horizontal) with a
    #   small positive allowance to dip below the desk edge.
    joint_limits: Tuple[JointLimit, ...] = field(
        default_factory=lambda: (
            JointLimit(np.radians(-135.0), np.radians(135.0), "base_yaw"),
            JointLimit(np.radians(-120.0), np.radians(15.0),  "shoulder_pitch"),
            JointLimit(np.radians(-135.0), np.radians(135.0), "elbow_pitch"),
            JointLimit(np.radians(-100.0), np.radians(100.0), "wrist_pitch"),
            JointLimit(np.radians(-135.0), np.radians(135.0), "wrist_roll"),
        )
    )

    # ---- Servo dynamic limits (from DS3218 datasheet) ------------------------
    # Speed at 6.8 V, no load: 0.16 s / 60°  ->  6.545 rad/s theoretical.
    # We derate to 60% for realistic loaded motion.
    servo_max_speed_rad_s: float = 6.545 * 0.60
    servo_max_accel_rad_s2: float = 30.0  # Empirical safe accel bound

    # ---- Derived / convenience ----------------------------------------------
    @property
    def total_reach_mm(self) -> float:
        """Maximum theoretical reach (fully extended arm)."""
        return self.l1_upper_arm_mm + self.l2_forearm_mm + self.l3_wrist_to_tip_mm

    @property
    def safe_reach_mm(self) -> float:
        """Recommended max operating radius (85% of full extension)."""
        return 0.85 * self.total_reach_mm

    @property
    def num_dof(self) -> int:
        return len(self.joint_limits)

    def worst_case_desk_reach_mm(self) -> float:
        """
        Distance from base to the farthest desk corner.
        Used to sanity-check that geometry can cover the entire workspace.
        """
        corners = np.array(
            [
                [0.0, 0.0],
                [self.desk_width_mm, 0.0],
                [0.0, self.desk_depth_mm],
                [self.desk_width_mm, self.desk_depth_mm],
            ]
        )
        base = np.array([self.base_x_on_desk_mm, self.base_y_on_desk_mm])
        distances = np.linalg.norm(corners - base, axis=1)
        return float(distances.max())

    def coverage_report(self) -> str:
        """Human-readable geometry summary printed at startup."""
        worst = self.worst_case_desk_reach_mm()
        reachable = worst <= self.safe_reach_mm
        return (
            f"ArmGeometry summary\n"
            f"-------------------\n"
            f"  Base height          : {self.base_height_mm:.1f} mm\n"
            f"  Upper arm (L1)       : {self.l1_upper_arm_mm:.1f} mm\n"
            f"  Forearm  (L2)        : {self.l2_forearm_mm:.1f} mm\n"
            f"  Wrist->tip (L3)      : {self.l3_wrist_to_tip_mm:.1f} mm\n"
            f"  Total reach          : {self.total_reach_mm:.1f} mm\n"
            f"  Safe reach (85%)     : {self.safe_reach_mm:.1f} mm\n"
            f"  DOF                  : {self.num_dof}\n"
            f"  Worst-case corner    : {worst:.1f} mm from base\n"
            f"  Full-desk reachable? : {reachable}\n"
        )


# Default singleton used across the project.
DEFAULT_ARM = ArmGeometry()


if __name__ == "__main__":
    print(DEFAULT_ARM.coverage_report())
