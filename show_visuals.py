"""Quick demo script — renders three views of the arm and opens them."""
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.trajectory import plan_synchronized_motion
from src.visualizer import (
    render_pose,
    render_reach_envelope,
    render_trajectory_frames,
)

out = Path("docs/figures")
out.mkdir(parents=True, exist_ok=True)

# Demo 1: a single arm pose reaching into the desk
print("Rendering static pose...")
render_pose(
    joint_angles_rad=[
        np.radians(30),   # base yaw:      30° left
        np.radians(-45),  # shoulder:      45° up from horizontal
        np.radians(80),   # elbow:         bent 80°
        np.radians(20),   # wrist pitch:   20° down
        0.0,              # wrist roll:    neutral
    ],
    title="Sample pose: arm reaching mid-desk",
    save_path=out / "demo_pose.png",
)

# Demo 2: a full synchronized motion, shown as 6 snapshots in time
print("Rendering trajectory frames...")
q_start = np.zeros(5)  # all joints at zero
q_end = np.array([
    np.radians(60),
    np.radians(-70),
    np.radians(90),
    np.radians(30),
    0.0,
])
traj = plan_synchronized_motion(q_start, q_end)
render_trajectory_frames(traj, num_frames=6, save_path=out / "demo_trajectory.png")

# Demo 3: the reachability envelope at desk height (z = 100 mm)
print("Rendering reach envelope...")
render_reach_envelope(z_slice_mm=100.0, save_path=out / "demo_envelope.png")

print("\nDone. Opening images...")
for name in ["demo_pose.png", "demo_trajectory.png", "demo_envelope.png"]:
    subprocess.run(["open", str(out / name)])  # macOS "open" command
