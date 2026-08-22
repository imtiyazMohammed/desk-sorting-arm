"""
Forward Kinematics (FK) for the 5-DOF desk-sorting arm.

Derived from first principles using homogeneous transformation matrices, NOT
from a URDF or DH table. This gives us an independent reference implementation
we can use to verify the ikpy-based IK solver in downstream modules.

Kinematic chain (parent -> child):
    world -> base_yaw (Rz) -> base_column (Tz base_height)
          -> shoulder_pitch (Ry) -> upper_arm (Tx L1)
          -> elbow_pitch (Ry)    -> forearm (Tx L2)
          -> wrist_pitch (Ry)    -> wrist_to_tip (Tx L3)
          -> wrist_roll (Rx)     -> end effector (tool center point, TCP)

Zero position convention (all joint angles = 0):
    - Arm fully extended horizontally in the +X_base direction.
    - TCP at (L1 + L2 + L3, 0, base_height) = (950, 0, 100) mm for default geometry.

Sign convention (right-hand rule about the specified axis):
    - +θ1 (base_yaw) rotates the arm from +X toward +Y (CCW viewed from above).
    - +θ2 (shoulder_pitch) rotates the arm DOWN from horizontal.
      Practical operating range therefore uses NEGATIVE θ2 (arm above desk).
    - +θ3 (elbow_pitch) folds the elbow "forward/down".
    - +θ4 (wrist_pitch) pitches the TCP down.
    - +θ5 (wrist_roll) rolls the gripper about its own tool axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .geometry import ArmGeometry, DEFAULT_ARM


# =========================================================================
# Elementary homogeneous transforms
# =========================================================================


def rot_x(angle_rad: float) -> np.ndarray:
    """4x4 homogeneous rotation about the X axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0,  c,  -s,  0.0],
            [0.0,  s,   c,  0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rot_y(angle_rad: float) -> np.ndarray:
    """4x4 homogeneous rotation about the Y axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [ c,  0.0,  s,  0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s,  0.0,  c,  0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def rot_z(angle_rad: float) -> np.ndarray:
    """4x4 homogeneous rotation about the Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(
        [
            [ c,  -s,  0.0, 0.0],
            [ s,   c,  0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def trans(x: float, y: float, z: float) -> np.ndarray:
    """4x4 homogeneous translation."""
    T = np.eye(4)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


# =========================================================================
# Public API
# =========================================================================


@dataclass(frozen=True)
class Pose:
    """End-effector pose: position (mm) and rotation matrix (3x3)."""

    position_mm: np.ndarray  # shape (3,), units mm
    rotation: np.ndarray     # shape (3, 3)

    def __post_init__(self) -> None:
        if self.position_mm.shape != (3,):
            raise ValueError(
                f"position_mm must have shape (3,), got {self.position_mm.shape}"
            )
        if self.rotation.shape != (3, 3):
            raise ValueError(
                f"rotation must have shape (3, 3), got {self.rotation.shape}"
            )

    def __repr__(self) -> str:
        p = self.position_mm
        return f"Pose(x={p[0]:8.2f}mm, y={p[1]:8.2f}mm, z={p[2]:8.2f}mm)"


class ForwardKinematics:
    """
    Compute TCP pose from joint angles for a 5-DOF revolute arm.

    Example
    -------
    >>> fk = ForwardKinematics()
    >>> pose = fk.compute(np.zeros(5))
    >>> pose.position_mm
    array([950.,   0., 100.])
    """

    def __init__(self, arm: ArmGeometry = DEFAULT_ARM) -> None:
        self.arm = arm

    # ---- Chain build ---------------------------------------------------------
    def _forward_chain(self, theta_rad: Sequence[float]) -> list[np.ndarray]:
        """
        Build the chain of world-frame transforms up to and including the TCP.
        Returns a list of 4x4 matrices, one per joint frame plus the TCP.
        """
        theta = np.asarray(theta_rad, dtype=float)
        if theta.shape != (self.arm.num_dof,):
            raise ValueError(
                f"Expected {self.arm.num_dof} joint angles, got shape {theta.shape}"
            )

        # Sequential world-frame transforms
        T_world_base = np.eye(4)  # base of the arm at world origin
        T_after_yaw = T_world_base @ rot_z(theta[0])
        T_shoulder = T_after_yaw @ trans(0.0, 0.0, self.arm.base_height_mm)
        T_after_shoulder = T_shoulder @ rot_y(theta[1])
        T_elbow = T_after_shoulder @ trans(self.arm.l1_upper_arm_mm, 0.0, 0.0)
        T_after_elbow = T_elbow @ rot_y(theta[2])
        T_wrist = T_after_elbow @ trans(self.arm.l2_forearm_mm, 0.0, 0.0)
        T_after_wrist_pitch = T_wrist @ rot_y(theta[3])
        T_after_wrist_roll = T_after_wrist_pitch @ rot_x(theta[4])
        T_tcp = T_after_wrist_roll @ trans(self.arm.l3_wrist_to_tip_mm, 0.0, 0.0)

        return [
            T_world_base,
            T_after_yaw,
            T_shoulder,
            T_after_shoulder,
            T_elbow,
            T_after_elbow,
            T_wrist,
            T_after_wrist_pitch,
            T_after_wrist_roll,
            T_tcp,
        ]

    # ---- Public API ----------------------------------------------------------
    def compute(self, joint_angles_rad: Sequence[float]) -> Pose:
        """
        Compute TCP pose given the 5 joint angles (radians).

        Raises
        ------
        ValueError
            If joint angles do not match the DOF, or if any angle lies outside
            the mechanical joint limits.
        """
        theta = np.asarray(joint_angles_rad, dtype=float)

        # Joint limit enforcement (fail loudly rather than silently clamping)
        for i, (angle, limit) in enumerate(zip(theta, self.arm.joint_limits)):
            if not limit.contains(angle):
                raise ValueError(
                    f"Joint {i} ({limit.name}) angle {np.degrees(angle):.2f}° "
                    f"is outside limits [{limit.min_deg:.2f}°, {limit.max_deg:.2f}°]"
                )

        T_tcp = self._forward_chain(theta)[-1]
        return Pose(
            position_mm=T_tcp[:3, 3].copy(),
            rotation=T_tcp[:3, :3].copy(),
        )

    def joint_positions(
        self, joint_angles_rad: Sequence[float]
    ) -> np.ndarray:
        """
        Return the world-frame (X, Y, Z) positions of every kinematic anchor:
        base, shoulder, elbow, wrist, TCP.

        Shape: (5, 3), units mm. Used by the visualizer to draw links.
        """
        chain = self._forward_chain(joint_angles_rad)
        # Indices: 0 base, 2 shoulder, 4 elbow, 6 wrist, 9 tcp
        indices = [0, 2, 4, 6, 9]
        return np.array([chain[i][:3, 3] for i in indices])


if __name__ == "__main__":
    fk = ForwardKinematics()

    # Sanity checks: known configurations
    print("Test 1: All zeros (arm horizontal along +X)")
    print(f"  {fk.compute(np.zeros(5))}\n")

    print("Test 2: Shoulder -90° (arm straight up)")
    print(f"  {fk.compute([0, -np.pi/2, 0, 0, 0])}\n")

    print("Test 3: Base +90° (arm horizontal along +Y)")
    print(f"  {fk.compute([np.pi/2, 0, 0, 0, 0])}\n")

    print("Test 4: Shoulder -45°, elbow -45° (folded)")
    pose = fk.compute([0, -np.pi/4, -np.pi/4, 0, 0])
    print(f"  {pose}")
