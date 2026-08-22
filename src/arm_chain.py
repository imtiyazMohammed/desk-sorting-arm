"""
Build an ikpy Chain object programmatically from ArmGeometry.

Why not use a URDF file?
------------------------
A URDF would duplicate the link lengths in a separate file that could silently
drift out of sync with `geometry.py`. By constructing the Chain directly from
the ArmGeometry singleton, changing a link length in one place propagates
atomically to both the manual FK and the ikpy IK solver.

The Chain built here MUST match the sign convention of forward_kinematics.py.
This is verified by tests/test_ik_matches_fk.py which checks that ikpy's
internal FK on the same joint angles yields the same TCP position (within
sub-millimeter tolerance) as our manual FK.

IMPORTANT: ikpy uses SI units (meters). ArmGeometry uses millimeters. All
translation vectors passed to URDFLink are converted mm -> m here.
"""

from __future__ import annotations

import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

from .geometry import ArmGeometry, DEFAULT_ARM

MM_TO_M = 1e-3


def build_chain(arm: ArmGeometry = DEFAULT_ARM) -> Chain:
    """
    Construct the ikpy Chain matching the FK convention.

    The chain has 6 links: 1 origin link (fixed) + 5 revolute joints. The
    active_links_mask marks the origin as inactive so the solver only
    optimizes over the 5 real joints.

    Returns
    -------
    Chain
        An ikpy Chain ready for .forward_kinematics() and .inverse_kinematics().
    """
    links = [
        # Immovable world -> base anchor
        OriginLink(),

        # Joint 1: base yaw about world Z, offset up by base_height
        URDFLink(
            name="base_yaw",
            origin_translation=[0.0, 0.0, arm.base_height_mm * MM_TO_M],
            origin_orientation=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 1.0],
            bounds=(arm.joint_limits[0].min_rad, arm.joint_limits[0].max_rad),
        ),

        # Joint 2: shoulder pitch about local Y, no translation from base_yaw frame
        URDFLink(
            name="shoulder_pitch",
            origin_translation=[0.0, 0.0, 0.0],
            origin_orientation=[0.0, 0.0, 0.0],
            rotation=[0.0, 1.0, 0.0],
            bounds=(arm.joint_limits[1].min_rad, arm.joint_limits[1].max_rad),
        ),

        # Joint 3: elbow pitch about local Y, translated by L1 along local X
        URDFLink(
            name="elbow_pitch",
            origin_translation=[arm.l1_upper_arm_mm * MM_TO_M, 0.0, 0.0],
            origin_orientation=[0.0, 0.0, 0.0],
            rotation=[0.0, 1.0, 0.0],
            bounds=(arm.joint_limits[2].min_rad, arm.joint_limits[2].max_rad),
        ),

        # Joint 4: wrist pitch about local Y, translated by L2 along local X
        URDFLink(
            name="wrist_pitch",
            origin_translation=[arm.l2_forearm_mm * MM_TO_M, 0.0, 0.0],
            origin_orientation=[0.0, 0.0, 0.0],
            rotation=[0.0, 1.0, 0.0],
            bounds=(arm.joint_limits[3].min_rad, arm.joint_limits[3].max_rad),
        ),

        # Joint 5: wrist roll about local X, no translation
        URDFLink(
            name="wrist_roll",
            origin_translation=[0.0, 0.0, 0.0],
            origin_orientation=[0.0, 0.0, 0.0],
            rotation=[1.0, 0.0, 0.0],
            bounds=(arm.joint_limits[4].min_rad, arm.joint_limits[4].max_rad),
        ),

        # Fixed link: wrist -> TCP, offset by L3 along local X (no rotation)
        URDFLink(
            name="tcp",
            origin_translation=[arm.l3_wrist_to_tip_mm * MM_TO_M, 0.0, 0.0],
            origin_orientation=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],  # zero rotation vector => fixed
            bounds=(0.0, 0.0),
        ),
    ]

    # Mask: origin fixed, 5 joints active, TCP fixed
    active_mask = [False, True, True, True, True, True, False]

    chain = Chain(
        name="desk_arm",
        links=links,
        active_links_mask=active_mask,
    )
    return chain


def full_joint_vector(joint_angles_rad: np.ndarray) -> np.ndarray:
    """
    Pad a 5-element active joint vector with the zeros ikpy expects for the
    inactive origin and TCP links, producing a length-7 vector.
    """
    theta = np.asarray(joint_angles_rad, dtype=float)
    if theta.shape != (5,):
        raise ValueError(f"Expected 5 joint angles, got shape {theta.shape}")
    return np.concatenate([[0.0], theta, [0.0]])


def extract_active_angles(full_joint_vector_rad: np.ndarray) -> np.ndarray:
    """Extract the 5 active joint angles from ikpy's length-7 vector."""
    return np.asarray(full_joint_vector_rad, dtype=float)[1:6]
