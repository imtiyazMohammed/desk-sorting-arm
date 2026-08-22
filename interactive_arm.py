"""
Interactive arm playground.

Run:
    python3 interactive_arm.py

Opens a 3D matplotlib window with 5 sliders (one per joint) below the plot.
Drag any slider and the arm redraws instantly. TCP position is shown in the
title. Also runs the pre-flight reachability check so you can see when you're
in a valid workspace region.

Buttons:
    - "Home"           reset all joints to zero (arm horizontal forward)
    - "Ready"          shoulder up, elbow forward (typical starting pose)
    - "Solve to (X,Y,Z)" enter a target and watch IK snap the arm to it
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from src.forward_kinematics import ForwardKinematics
from src.geometry import DEFAULT_ARM
from src.inverse_kinematics import InverseKinematics, IKStatus


# =========================================================================
# Setup
# =========================================================================

ARM = DEFAULT_ARM
FK = ForwardKinematics(ARM)
IK = InverseKinematics(ARM)

# Colors matching src/visualizer.py so it feels consistent
JOINT_LABELS = ["base", "shoulder", "elbow", "wrist", "TCP"]
JOINT_COLORS = ["#333333", "#e63946", "#f4a261", "#2a9d8f", "#264653"]


# =========================================================================
# Build the figure
# =========================================================================

fig = plt.figure(figsize=(12, 9))
fig.canvas.manager.set_window_title("Desk arm — interactive playground")

# Main 3D plot fills the top ~70% of the window
ax_3d = fig.add_subplot(111, projection="3d")
plt.subplots_adjust(left=0.05, right=0.75, bottom=0.35, top=0.95)


def setup_axes():
    """Configure axes limits and labels for the arm workspace."""
    reach = ARM.total_reach_mm
    ax_3d.set_xlim(-reach, reach)
    ax_3d.set_ylim(-reach, reach)
    ax_3d.set_zlim(0, reach + ARM.base_height_mm + 100.0)
    ax_3d.set_xlabel("X (mm)")
    ax_3d.set_ylabel("Y (mm)")
    ax_3d.set_zlabel("Z (mm)")
    ax_3d.set_box_aspect((1, 1, 0.7))


def draw_desk():
    """Draw desk footprint as a translucent rectangle at Z=0."""
    y_half = ARM.desk_width_mm / 2.0
    x_max = ARM.desk_depth_mm
    corners = np.array([
        [0,     -y_half, 0],
        [x_max, -y_half, 0],
        [x_max,  y_half, 0],
        [0,      y_half, 0],
        [0,     -y_half, 0],
    ])
    ax_3d.plot(
        corners[:, 0], corners[:, 1], corners[:, 2],
        color="#888888", linewidth=1.0, linestyle="--", label="desk edge",
    )


def draw_arm(joint_angles_rad):
    """Draw arm as line segments between joint centers."""
    positions = FK.joint_positions(joint_angles_rad)  # shape (5, 3)

    # Base pedestal (vertical dotted line from desk to shoulder)
    ax_3d.plot(
        [0, 0], [0, 0], [0, ARM.base_height_mm],
        color="#555555", linewidth=2, linestyle=":",
    )
    # Arm segments
    ax_3d.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        color="#264653", linewidth=3,
    )
    # Joint markers
    for pos, color, label in zip(positions, JOINT_COLORS, JOINT_LABELS):
        size = 120 if label == "TCP" else 60
        ax_3d.scatter(
            [pos[0]], [pos[1]], [pos[2]],
            color=color, s=size, zorder=5, label=label,
        )
    return positions


# =========================================================================
# Redraw on slider change
# =========================================================================

def redraw(_event=None):
    """Called every time a slider or text field changes."""
    angles_deg = np.array([s.val for s in sliders])
    angles_rad = np.radians(angles_deg)

    # Preserve the user's current camera angle so the view doesn't jump around
    # every time they drag a slider.
    elev, azim = ax_3d.elev, ax_3d.azim

    ax_3d.cla()
    setup_axes()
    draw_desk()
    positions = draw_arm(angles_rad)
    tcp = positions[-1]

    # Status line — is the TCP within the reachable envelope?
    tcp_reachable = IK.is_reachable(tcp)
    status = "REACHABLE" if tcp_reachable else "OUT OF ENVELOPE"

    ax_3d.set_title(
        f"TCP = ({tcp[0]:+7.1f}, {tcp[1]:+7.1f}, {tcp[2]:+7.1f}) mm      "
        f"[{status}]",
        fontsize=12, family="monospace",
    )
    ax_3d.legend(loc="upper left", fontsize=8)

    # Restore camera
    ax_3d.view_init(elev=elev, azim=azim)

    fig.canvas.draw_idle()


# =========================================================================
# Sliders — one per joint
# =========================================================================

sliders = []
slider_specs = [
    ("Base yaw (θ₁)",       0),
    ("Shoulder pitch (θ₂)", 1),
    ("Elbow pitch (θ₃)",    2),
    ("Wrist pitch (θ₄)",    3),
    ("Wrist roll (θ₅)",     4),
]

# One slider per joint. Y positions stack vertically at the bottom of the fig.
for i, (label, joint_idx) in enumerate(slider_specs):
    limit = ARM.joint_limits[joint_idx]
    ax_slider = fig.add_axes([0.10, 0.28 - i * 0.045, 0.55, 0.03])
    slider = Slider(
        ax=ax_slider,
        label=label,
        valmin=limit.min_deg,
        valmax=limit.max_deg,
        valinit=0.0,
        valstep=0.5,
        color="#2a9d8f",
    )
    slider.on_changed(redraw)
    sliders.append(slider)


# =========================================================================
# Preset buttons
# =========================================================================

def set_all(angles_deg):
    """Programmatically set every slider (fires their callbacks)."""
    for s, v in zip(sliders, angles_deg):
        s.set_val(v)


def go_home(_event):
    set_all([0.0, 0.0, 0.0, 0.0, 0.0])


def go_ready(_event):
    set_all([0.0, -45.0, 90.0, 0.0, 0.0])


def go_folded(_event):
    set_all([0.0, -90.0, 130.0, 60.0, 0.0])


btn_home_ax   = fig.add_axes([0.78, 0.85, 0.18, 0.05])
btn_ready_ax  = fig.add_axes([0.78, 0.78, 0.18, 0.05])
btn_folded_ax = fig.add_axes([0.78, 0.71, 0.18, 0.05])

btn_home   = Button(btn_home_ax,   "Home (all zeros)")
btn_ready  = Button(btn_ready_ax,  "Ready pose")
btn_folded = Button(btn_folded_ax, "Folded")

btn_home.on_clicked(go_home)
btn_ready.on_clicked(go_ready)
btn_folded.on_clicked(go_folded)


# =========================================================================
# IK box — type a target coordinate, watch the arm move to it
# =========================================================================

ik_label_ax = fig.add_axes([0.78, 0.55, 0.18, 0.05])
ik_label_ax.axis("off")
ik_label_ax.text(0.0, 0.5, "Solve IK to target (mm):",
                 fontsize=10, weight="bold", va="center")

ik_x_ax = fig.add_axes([0.83, 0.48, 0.13, 0.04])
ik_y_ax = fig.add_axes([0.83, 0.42, 0.13, 0.04])
ik_z_ax = fig.add_axes([0.83, 0.36, 0.13, 0.04])

box_x = TextBox(ik_x_ax, "X ", initial="400")
box_y = TextBox(ik_y_ax, "Y ", initial="200")
box_z = TextBox(ik_z_ax, "Z ", initial="150")


def solve_and_apply(_event):
    """Parse the text boxes, run IK, drive the sliders to the solution."""
    try:
        x = float(box_x.text)
        y = float(box_y.text)
        z = float(box_z.text)
    except ValueError:
        print("Invalid number in X/Y/Z box.")
        return

    result = IK.solve([x, y, z])
    print(f"IK target ({x}, {y}, {z}) -> {result.status.value}  "
          f"err={result.position_error_mm:.2f}mm  msg={result.message}")

    if result.status == IKStatus.SUCCESS:
        set_all(result.joint_angles_deg.tolist())
    else:
        print(f"  Could not reach target: {result.message}")


btn_solve_ax = fig.add_axes([0.78, 0.29, 0.18, 0.05])
btn_solve = Button(btn_solve_ax, "Solve IK →")
btn_solve.on_clicked(solve_and_apply)


# =========================================================================
# Initial render
# =========================================================================

setup_axes()
draw_desk()
draw_arm(np.zeros(5))
ax_3d.set_title("Drag sliders to move the arm.", fontsize=12)
ax_3d.legend(loc="upper left", fontsize=8)

print("Interactive arm playground running.")
print("Drag sliders below the 3D view to move the arm.")
print("Use the preset buttons for common poses.")
print("Type X/Y/Z into the boxes and click 'Solve IK' to test reach.")
print("Click and drag inside the 3D plot to rotate the camera.")
print("Close the window when done.")

plt.show()