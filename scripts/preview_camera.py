#!/usr/bin/env python3
"""
Manual webcam sanity check -- NOT part of the automated test suite.

Opens a live OpenCV window showing the webcam feed with a crosshair drawn at
the exact centre of the frame, plus a small heads-up display of resolution and
measured frame rate. Run it once when a camera is available to confirm that
:class:`src.image_source.WebcamImageSource` talks to real hardware correctly;
CI covers the same class with an injected fake capture device.

Usage
-----
    python3 scripts/preview_camera.py
    python3 scripts/preview_camera.py --device 1 --width 1280 --height 720
    python3 scripts/preview_camera.py --synthetic       # no camera needed

Press ``q`` or ``Esc`` to quit.

On macOS the first run triggers a camera-permission prompt. If the process was
launched from a terminal, permission is granted to the *terminal application*,
not to Python -- so if the feed stays black, check
System Settings > Privacy & Security > Camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Allow `python3 scripts/preview_camera.py` from the repository root without
# requiring the package to be installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.image_source import (  # noqa: E402  - deliberate post-path-setup import
    DeskObject,
    ImageSource,
    ImageSourceError,
    SourceStatus,
    SyntheticImageSource,
    WebcamImageSource,
)

WINDOW_NAME = "desk-arm camera preview"

#: BGR colours for the overlay. Chosen to stay legible against a grey desk.
CROSSHAIR_COLOR: Tuple[int, int, int] = (0, 255, 0)
TEXT_COLOR: Tuple[int, int, int] = (0, 255, 0)
TEXT_SHADOW_COLOR: Tuple[int, int, int] = (0, 0, 0)

CROSSHAIR_ARM_PX = 20
CROSSHAIR_GAP_PX = 6
CROSSHAIR_RADIUS_PX = 30
CROSSHAIR_THICKNESS = 1


def draw_crosshair(frame: np.ndarray) -> None:
    """
    Draw a centre crosshair on ``frame`` in place.

    The gap at the centre keeps the exact centre pixel visible rather than
    painting over it -- that pixel is the one that matters when aligning the
    overhead camera above the desk.
    """
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2

    cv2.line(
        frame, (cx - CROSSHAIR_ARM_PX - CROSSHAIR_GAP_PX, cy),
        (cx - CROSSHAIR_GAP_PX, cy), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame, (cx + CROSSHAIR_GAP_PX, cy),
        (cx + CROSSHAIR_ARM_PX + CROSSHAIR_GAP_PX, cy),
        CROSSHAIR_COLOR, CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame, (cx, cy - CROSSHAIR_ARM_PX - CROSSHAIR_GAP_PX),
        (cx, cy - CROSSHAIR_GAP_PX), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame, (cx, cy + CROSSHAIR_GAP_PX),
        (cx, cy + CROSSHAIR_ARM_PX + CROSSHAIR_GAP_PX),
        CROSSHAIR_COLOR, CROSSHAIR_THICKNESS,
    )
    cv2.circle(
        frame, (cx, cy), CROSSHAIR_RADIUS_PX, CROSSHAIR_COLOR, CROSSHAIR_THICKNESS
    )


def draw_hud(frame: np.ndarray, lines: Tuple[str, ...]) -> None:
    """Draw shadowed status text down the top-left corner, in place."""
    for row, text in enumerate(lines):
        origin = (10, 24 + row * 22)
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            TEXT_SHADOW_COLOR, 3, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            TEXT_COLOR, 1, cv2.LINE_AA,
        )


def build_source(args: argparse.Namespace) -> ImageSource:
    """Construct the requested image source from parsed CLI arguments."""
    if args.synthetic:
        return SyntheticImageSource(
            objects=[
                DeskObject(x_mm=300.0, y_mm=200.0, label="pen"),
                DeskObject(
                    x_mm=850.0, y_mm=420.0, width_mm=90.0, height_mm=90.0,
                    color_bgr=(60, 180, 60), label="sticky_note_pad",
                ),
            ],
            px_per_mm=args.px_per_mm,
            noise_sigma=4.0,
            seed=0,
        )
    return WebcamImageSource(
        device_index=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup,
    )


def run_preview(source: ImageSource) -> int:
    """
    Display frames until the user quits. Returns a process exit code.

    A read failure mid-stream (camera unplugged, claimed by another app) is
    reported and ends the loop cleanly rather than raising a traceback at the
    user, since this is an interactive tool.
    """
    frame_count = 0
    started = time.monotonic()
    fps = 0.0

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

            frame_count += 1
            elapsed = time.monotonic() - started
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count, started = 0, time.monotonic()

            display = frame.copy()
            draw_crosshair(display)
            height, width = frame.shape[:2]
            draw_hud(
                display,
                (
                    f"{source.name}  {width}x{height}",
                    f"{fps:5.1f} fps" if fps else "  ... fps",
                    f"centre px ({width // 2}, {height // 2})",
                    "q / Esc to quit",
                ),
            )
            cv2.imshow(WINDOW_NAME, display)

            # 1 ms wait is what drives the GUI event loop; without it the
            # window never paints.
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 == Esc
                return 0
            # A window closed via its title-bar button stops being visible.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                return 0
    finally:
        cv2.destroyAllWindows()
        # macOS needs a few event-loop turns to actually tear the window down.
        for _ in range(4):
            cv2.waitKey(1)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--device", type=int, default=0, help="camera index (default: 0)"
    )
    parser.add_argument("--width", type=int, default=None, help="requested width in px")
    parser.add_argument(
        "--height", type=int, default=None, help="requested height in px"
    )
    parser.add_argument("--fps", type=float, default=None, help="requested frame rate")
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="frames discarded after opening, for auto-exposure (default: 5)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="preview SyntheticImageSource instead of a camera (no hardware needed)",
    )
    parser.add_argument(
        "--px-per-mm", dest="px_per_mm", type=float, default=0.5,
        help="synthetic source resolution (default: 0.5 px/mm)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source = build_source(args)
    except ImageSourceError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2

    try:
        source.open()
    except ImageSourceError as exc:
        print(f"Could not start preview: {exc}", file=sys.stderr)
        return 2

    try:
        return run_preview(source)
    except KeyboardInterrupt:
        return 0
    finally:
        source.close()


if __name__ == "__main__":
    raise SystemExit(main())
