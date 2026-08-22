#!/usr/bin/env python3
"""
Interactive camera calibration -- NOT part of the automated test suite.

Shows a live camera feed with the chessboard detection drawn on it, captures
views on demand, then solves for the camera's intrinsics and writes them to
JSON. Run it once per camera (and again if you change capture resolution --
intrinsics are only valid for the resolution they were measured at).

`tests/test_camera_calibration.py` covers the same code path against a
synthetic target with exact ground truth; this script is the hardware half.

Usage
-----
    python3 scripts/calibrate_camera.py
    python3 scripts/calibrate_camera.py --device 1 --width 1280 --height 720
    python3 scripts/calibrate_camera.py --rows 6 --cols 9 --square-size 25
    python3 scripts/calibrate_camera.py --synthetic     # no camera needed

Controls
--------
    SPACE   capture the current view (only accepted if a board is detected)
    u       undo the last capture
    ESC/q   stop capturing and calibrate

Printing the target
-------------------
Any chessboard works. `--rows` and `--cols` count **inner corners**, not
squares: a board printed with 10 x 7 squares is `--cols 9 --rows 6`. Measure a
printed square with callipers and pass the real figure to `--square-size`;
printers rarely scale exactly, and an error there scales every distance the
vision stack later reports.

Getting a good calibration
--------------------------
Aim for ~20 views and move the board between every one. What matters is
variety, not count:

- **Tilt it.** Focal length and distance are not separable from head-on views
  alone. Angle the board 20-40 degrees in both axes.
- **Fill the corners.** Distortion is strongest at the frame edge, so the
  board has to visit the edges for those terms to be constrained at all.
- **Vary distance.** Near and far, not one comfortable working distance.
- **Keep it sharp.** Motion blur moves detected corners; pause before capture.

A stack of similar frontal views produces a confident, wrong answer.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Allow `python3 scripts/calibrate_camera.py` from the repository root without
# requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.camera_calibration import (  # noqa: E402  - deliberate post-path-setup
    CalibrationError,
    CalibrationStatus,
    CalibrationTarget,
    Calibrator,
    SyntheticChessboardSource,
)
from src.image_source import (  # noqa: E402
    ImageSource,
    ImageSourceError,
    SourceStatus,
    WebcamImageSource,
)

WINDOW_NAME = "desk-arm camera calibration"

#: Default destination for the recovered intrinsics.
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "cad" / "calibration"
    / "camera_intrinsics.json"
)

#: Views to aim for. Fewer still solves, but the fit gets fragile.
TARGET_FRAME_COUNT = 20

#: RMS above this is reported as a warning rather than a result.
RMS_WARNING_PX = 1.0

OK_COLOR: Tuple[int, int, int] = (0, 220, 0)
BUSY_COLOR: Tuple[int, int, int] = (0, 165, 255)
TEXT_COLOR: Tuple[int, int, int] = (0, 255, 0)
SHADOW_COLOR: Tuple[int, int, int] = (0, 0, 0)


def draw_hud(frame: np.ndarray, lines: Tuple[str, ...]) -> None:
    """Draw shadowed status text down the top-left corner, in place."""
    for row, text in enumerate(lines):
        origin = (10, 24 + row * 22)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    SHADOW_COLOR, 3, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    TEXT_COLOR, 1, cv2.LINE_AA)


def draw_coverage(frame: np.ndarray, captures: List[np.ndarray]) -> None:
    """
    Mark where previously captured boards sat in the frame.

    Distortion terms are only constrained where the board has actually been,
    so seeing the captured centres accumulate is the most useful feedback the
    tool can give: sparse regions are where to aim next.
    """
    for corners in captures:
        centre = corners.reshape(-1, 2).mean(axis=0)
        cv2.circle(frame, (int(centre[0]), int(centre[1])), 4, BUSY_COLOR, -1)


def build_source(args: argparse.Namespace, target: CalibrationTarget) -> ImageSource:
    """Construct the requested image source from parsed CLI arguments."""
    if args.synthetic:
        return SyntheticChessboardSource(target=target)
    return WebcamImageSource(
        device_index=args.device,
        width=args.width,
        height=args.height,
        warmup_frames=args.warmup,
    )


def capture_views(source: ImageSource, calibrator: Calibrator) -> int:
    """
    Run the interactive capture loop. Returns a process exit code.

    Frames are only accepted when a complete board is detected, so the user
    cannot accumulate views that contribute nothing.
    """
    captures: List[np.ndarray] = []
    message = ""

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        while True:
            try:
                frame = source.get_frame()
            except ImageSourceError as exc:
                if exc.status is SourceStatus.READ_FAILED:
                    print(f"Stream ended: {exc}", file=sys.stderr)
                    return 1
                raise

            corners = calibrator.find_corners(frame)
            display = frame.copy()
            if display.ndim == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

            if corners is not None:
                cv2.drawChessboardCorners(
                    display, calibrator.target.pattern_size, corners, True
                )
            draw_coverage(display, captures)
            draw_hud(
                display,
                (
                    f"captured {len(captures)} / {TARGET_FRAME_COUNT}",
                    "BOARD DETECTED" if corners is not None else "no board",
                    "SPACE capture   u undo   ESC/q done",
                    message,
                ),
            )
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                return 0
            if key == ord(" "):
                if corners is None:
                    message = "no board in view -- not captured"
                elif calibrator.add_frame(frame):
                    captures.append(corners)
                    message = f"captured view {len(captures)}"
                else:
                    message = "detection failed on capture -- try again"
            elif key == ord("u"):
                if calibrator.remove_last_frame():
                    captures.pop()
                    message = f"undid last capture ({len(captures)} left)"
                else:
                    message = "nothing to undo"

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 0
    finally:
        cv2.destroyAllWindows()
        # macOS needs a few event-loop turns to actually tear the window down.
        for _ in range(4):
            cv2.waitKey(1)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", type=int, default=0,
                        help="camera index (default: 0)")
    parser.add_argument("--width", type=int, default=None,
                        help="requested capture width in px")
    parser.add_argument("--height", type=int, default=None,
                        help="requested capture height in px")
    parser.add_argument("--warmup", type=int, default=5,
                        help="frames discarded after opening (default: 5)")
    parser.add_argument("--rows", type=int, default=6,
                        help="INNER corners down the board (default: 6)")
    parser.add_argument("--cols", type=int, default=9,
                        help="INNER corners across the board (default: 9)")
    parser.add_argument("--square-size", dest="square_size", type=float, default=25.0,
                        help="printed square edge length in mm (default: 25.0)")
    parser.add_argument("--min-frames", dest="min_frames", type=int, default=5,
                        help="fewest views calibrate() will accept (default: 5)")
    parser.add_argument("--fix-k3", dest="fix_k3", action="store_true",
                        help="hold the 6th-order radial term at zero; a better-"
                             "conditioned fit for ordinary non-fisheye lenses")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"intrinsics destination (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--synthetic", action="store_true",
                        help="calibrate a rendered board instead of a camera "
                             "(no hardware needed; for trying the tool out)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    try:
        target = CalibrationTarget(
            rows=args.rows, cols=args.cols, square_size_mm=args.square_size
        )
        calibrator = Calibrator(
            target, min_frames=args.min_frames, fix_k3=args.fix_k3
        )
        source = build_source(args, target)
    except (CalibrationError, ImageSourceError) as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    print(f"Target: {target.cols} x {target.rows} inner corners, "
          f"{target.square_size_mm:.1f} mm squares "
          f"(a {target.cols + 1} x {target.rows + 1} square board)")

    try:
        source.open()
    except ImageSourceError as exc:
        print(f"Could not start capture: {exc}", file=sys.stderr)
        return 2

    try:
        exit_code = capture_views(source, calibrator)
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        source.close()

    if exit_code != 0:
        return exit_code

    print(f"\nCaptured {calibrator.frame_count} views. Calibrating...")
    try:
        intrinsics = calibrator.calibrate()
    except CalibrationError as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        if exc.status is CalibrationStatus.TOO_FEW_FRAMES:
            print("Run again and capture more views.", file=sys.stderr)
        return 1

    print()
    print(intrinsics.summary())

    if intrinsics.rms_reprojection_error > RMS_WARNING_PX:
        print(
            f"WARNING: RMS reprojection error is "
            f"{intrinsics.rms_reprojection_error:.3f} px, above the "
            f"{RMS_WARNING_PX:.1f} px threshold.\n"
            "         The views were probably too few, too similar, or "
            "blurred.\n"
            "         Recalibrate with more varied board angles, distances and\n"
            "         positions before trusting these numbers.",
            file=sys.stderr,
        )

    written = intrinsics.save_json(args.output)
    print(f"Wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
