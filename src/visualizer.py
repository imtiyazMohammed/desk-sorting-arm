"""
Matplotlib 3D visualizer for arm poses and trajectories.

Renders:
  - The desk workspace as a translucent rectangle
  - The base pedestal, upper arm, forearm, wrist link, and gripper as line segments
  - The joint centers as colored spheres
  - The TCP position as a highlighted marker

Includes:
  - `render_pose(joint_angles)`     -> single static figure
  - `render_trajectory(traj, fps)` -> animation (returns matplotlib animation)
  - `save_trajectory_frames(traj, out_path)` -> exports keyframes to PNG grid
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

from .forward_kinematics import ForwardKinematics
from .geometry import ArmGeometry, DEFAULT_ARM
from .trajectory import SynchronizedTrajectory


JOINT_LABELS = ["base", "shoulder", "elbow", "wrist", "TCP"]
JOINT_COLORS = ["#333333", "#e63946", "#f4a261", "#2a9d8f", "#264653"]


def _setup_axes(ax: Axes3D, arm: ArmGeometry) -> None:
    """Configure axes limits and labels for the arm workspace."""
    reach = arm.total_reach_mm
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(0, reach + arm.base_height_mm + 100.0)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_box_aspect((1, 1, 0.7))


def _draw_desk(ax: Axes3D, arm: ArmGeometry) -> None:
    """Draw the desk as a translucent rectangle at Z=0."""
    # Base sits at (0,0). Desk in world frame:
    # x-axis of world coincides with the "depth into desk" direction
    # y-axis of world is along the desk long edge
    # We draw the desk centered on the base_x_on_desk position.
    y_half = arm.desk_width_mm / 2.0
    x_max = arm.desk_depth_mm
    corners = np.array(
        [
            [0.0,     -y_half, 0.0],
            [x_max,   -y_half, 0.0],
            [x_max,    y_half, 0.0],
            [0.0,      y_half, 0.0],
            [0.0,     -y_half, 0.0],  # close the loop
        ]
    )
    ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
            color="#888888", linewidth=1.0, linestyle="--", label="desk edge")


def _draw_arm(
    ax: Axes3D,
    joint_angles_rad: Sequence[float],
    arm: ArmGeometry,
    fk: ForwardKinematics,
    highlight_tcp: bool = True,
) -> None:
    """Draw arm as line segments between joint centers."""
    positions = fk.joint_positions(joint_angles_rad)  # shape (5, 3)

    # Base pedestal (dashed vertical line from desk to shoulder)
    ax.plot(
        [0, 0], [0, 0], [0, arm.base_height_mm],
        color="#555555", linewidth=2, linestyle=":",
    )

    # Links
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="#264653", linewidth=3, marker="",
    )

    # Joint markers
    for i, (pos, color, label) in enumerate(
        zip(positions, JOINT_COLORS, JOINT_LABELS)
    ):
        size = 120 if (highlight_tcp and label == "TCP") else 60
        ax.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            color=color, s=size, zorder=5, label=label,
        )


def render_pose(
    joint_angles_rad: Sequence[float],
    arm: ArmGeometry = DEFAULT_ARM,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> Figure:
    """Render a single arm pose and return the matplotlib Figure."""
    fk = ForwardKinematics(arm)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    _setup_axes(ax, arm)
    _draw_desk(ax, arm)
    _draw_arm(ax, joint_angles_rad, arm, fk)

    if title:
        ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


def render_trajectory_frames(
    traj: SynchronizedTrajectory,
    num_frames: int = 6,
    arm: ArmGeometry = DEFAULT_ARM,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Render a grid of `num_frames` snapshots of the arm along the trajectory.
    Cheaper and more portable than an animation for a PoC document.
    """
    fk = ForwardKinematics(arm)
    times = np.linspace(0.0, traj.total_time, num_frames)

    cols = min(num_frames, 3)
    rows = int(np.ceil(num_frames / cols))
    fig = plt.figure(figsize=(5 * cols, 4.5 * rows))

    for i, t in enumerate(times):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _setup_axes(ax, arm)
        _draw_desk(ax, arm)
        angles = traj.sample(t)
        _draw_arm(ax, angles, arm, fk)
        tcp = fk.compute(angles).position_mm
        ax.set_title(
            f"t = {t:.2f} s   "
            f"TCP=({tcp[0]:.0f}, {tcp[1]:.0f}, {tcp[2]:.0f}) mm",
            fontsize=10,
        )

    fig.suptitle(
        f"Synchronized trajectory  (total {traj.total_time:.2f} s)",
        fontsize=13,
    )
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig


def render_reach_envelope(
    arm: ArmGeometry = DEFAULT_ARM,
    z_slice_mm: float = 100.0,
    grid_step_mm: float = 40.0,
    save_path: Optional[Path] = None,
) -> Figure:
    """
    Sweep the horizontal plane at z=z_slice_mm and mark which (X, Y) points are
    within the arm's reachable envelope. Overlay the desk footprint.

    Useful sanity check: at desk height, which parts of the desk can we
    actually pick from?
    """
    from .inverse_kinematics import InverseKinematics

    ik = InverseKinematics(arm)
    xs = np.arange(-arm.total_reach_mm, arm.total_reach_mm, grid_step_mm)
    ys = np.arange(-arm.total_reach_mm, arm.total_reach_mm, grid_step_mm)
    X, Y = np.meshgrid(xs, ys)
    reach_mask = np.zeros_like(X, dtype=bool)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            reach_mask[i, j] = ik.is_reachable([X[i, j], Y[i, j], z_slice_mm])

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, reach_mask.astype(int), levels=[-0.5, 0.5, 1.5],
                colors=["#f5f5f5", "#a8dadc"], alpha=0.9)

    # Desk footprint (base is at origin; desk extends into +X for depth,
    # y_half in both +/- Y along the mounting edge)
    y_half = arm.desk_width_mm / 2.0
    x_max = arm.desk_depth_mm
    ax.plot(
        [0, x_max, x_max, 0, 0],
        [-y_half, -y_half, y_half, y_half, -y_half],
        color="#e63946", linewidth=2, label="desk footprint",
    )

    ax.scatter([0], [0], color="#264653", s=100, zorder=5, label="arm base")
    ax.set_xlabel("X (mm)  (depth into desk)")
    ax.set_ylabel("Y (mm)  (along mounting edge)")
    ax.set_title(f"Reachable footprint at z = {z_slice_mm:.0f} mm")
    ax.legend(loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    return fig
