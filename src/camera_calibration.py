"""
Camera intrinsic calibration -- OpenCV's chessboard workflow, plus a synthetic
target that makes it testable without hardware.

Why intrinsics matter here
--------------------------
The Phase B vision stack turns pixel coordinates into desk millimetres via a
homography. A homography is a projective map between two *planes*, and lens
distortion is not projective: it bends straight lines. Fitting a homography to
distorted pixels smears that error across the whole workspace, worst at the
frame edges where the desk corners live. Calibrating first, then undistorting,
keeps the homography solving the problem it can actually solve.

What this module provides
-------------------------
- :class:`CalibrationTarget` -- the physical chessboard's description.
- :class:`CameraIntrinsics` -- the recovered camera matrix, distortion
  coefficients, image size and RMS reprojection error, with JSON round-trip.
- :class:`Calibrator` -- accumulate frames, solve, undistort.
- :class:`SyntheticChessboardSource` -- an :class:`~src.image_source.ImageSource`
  that renders a chessboard through *known* intrinsics using
  ``cv2.projectPoints``. Because the ground truth is exact, the test suite can
  assert that calibration recovers the numbers it was given, rather than merely
  asserting that it converged to something.

Units and conventions
---------------------
Square size is in millimetres, matching ``src.geometry``. Object points are
therefore in millimetres, which makes the translation vectors OpenCV returns
millimetres too. Distortion coefficients are the 5-element
``(k1, k2, p1, p2, k3)`` form. Pixel coordinates follow OpenCV: origin at the
top-left pixel's centre, ``+u`` right, ``+v`` down.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.image_source import ImageSource, ImageSourceError, SourceStatus

__all__ = [
    "CalibrationStatus",
    "CalibrationError",
    "CalibrationTarget",
    "CameraIntrinsics",
    "Calibrator",
    "SyntheticChessboardSource",
    "default_calibration_poses",
    "DEFAULT_DISTORTION",
    "INTRINSICS_SCHEMA_VERSION",
]

#: Bumped if the JSON layout ever changes incompatibly.
INTRINSICS_SCHEMA_VERSION = 1

#: A plausible mild barrel distortion, used as the synthetic ground truth.
#: Order is (k1, k2, p1, p2, k3).
DEFAULT_DISTORTION = (-0.25, 0.08, 0.0012, -0.0009, 0.015)


class CalibrationStatus(Enum):
    """
    Structured failure modes for calibration.

    Mirrors ``IKStatus`` / ``SourceStatus`` / ``DesignStatus``: every failure
    names its cause so a caller can branch on it rather than parse a message.
    """

    OK = auto()
    """No error."""

    INVALID_TARGET = auto()
    """The chessboard description is self-inconsistent."""

    NO_CORNERS_FOUND = auto()
    """A frame was supplied but no complete chessboard was detected in it."""

    TOO_FEW_FRAMES = auto()
    """Calibration was attempted with fewer views than the solver needs."""

    FRAME_SIZE_MISMATCH = auto()
    """A frame's dimensions differ from those already accumulated."""

    MALFORMED_FRAME = auto()
    """A frame is not a usable 8-bit grayscale or BGR image."""

    SOLVER_FAILED = auto()
    """OpenCV's solver did not return a usable result."""

    INVALID_INTRINSICS = auto()
    """An intrinsics record is malformed, e.g. when loaded from JSON."""


class CalibrationError(RuntimeError):
    """
    Raised by every calibration failure path.

    Attributes
    ----------
    status:
        The :class:`CalibrationStatus` describing *why* the operation failed.
    """

    def __init__(self, status: CalibrationStatus, message: str) -> None:
        super().__init__(f"[{status.name}] {message}")
        self.status = status


@dataclass(frozen=True)
class CalibrationTarget:
    """
    The physical chessboard being photographed.

    .. important::
       ``rows`` and ``cols`` count **inner corners**, not squares. A board
       printed with 10 x 7 squares has 9 x 6 inner corners, so that board is
       ``CalibrationTarget(rows=6, cols=9, square_size_mm=25.0)``. This is the
       single most common way to get chessboard calibration wrong: OpenCV's
       ``findChessboardCorners`` takes inner corners, and passing the square
       count makes detection fail silently on every frame.

    Prefer a board whose two counts differ in parity -- 9 x 6 rather than
    9 x 7. A pattern with one odd and one even dimension has an unambiguous
    orientation, so the detector cannot return corners in a flipped order
    between frames and scramble the point correspondences.

    Attributes
    ----------
    rows:
        Inner corners down the board (the vertical count).
    cols:
        Inner corners across the board (the horizontal count).
    square_size_mm:
        Edge length of one square, in millimetres. Measure the printed board
        rather than trusting the source file; printers scale.
    """

    rows: int = 6
    cols: int = 9
    square_size_mm: float = 25.0

    def __post_init__(self) -> None:
        if self.rows < 2 or self.cols < 2:
            raise CalibrationError(
                CalibrationStatus.INVALID_TARGET,
                f"A chessboard needs at least 2 x 2 inner corners, got "
                f"{self.cols} x {self.rows}. Remember these count inner "
                "corners, not squares.",
            )
        if self.rows == self.cols:
            raise CalibrationError(
                CalibrationStatus.INVALID_TARGET,
                f"A square {self.cols} x {self.rows} pattern has an ambiguous "
                "orientation: the detector may order corners differently "
                "between frames. Use a board whose counts differ.",
            )
        if self.square_size_mm <= 0.0:
            raise CalibrationError(
                CalibrationStatus.INVALID_TARGET,
                f"square_size_mm must be positive, got {self.square_size_mm}.",
            )

    @property
    def pattern_size(self) -> Tuple[int, int]:
        """``(cols, rows)`` -- the order ``cv2.findChessboardCorners`` expects."""
        return (self.cols, self.rows)

    @property
    def corner_count(self) -> int:
        """Total inner corners the detector must find for a frame to count."""
        return self.rows * self.cols

    @property
    def board_size_mm(self) -> Tuple[float, float]:
        """``(width, height)`` of the full printed board, including the outer ring."""
        return (
            (self.cols + 1) * self.square_size_mm,
            (self.rows + 1) * self.square_size_mm,
        )

    def object_points(self) -> np.ndarray:
        """
        Inner-corner positions in the board's own frame, in millimetres.

        Shape ``(rows * cols, 3)``, Z identically zero because the board is
        planar. Ordered to match ``findChessboardCorners``: along a row first,
        then down.
        """
        grid = np.zeros((self.corner_count, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[0 : self.cols, 0 : self.rows].T.reshape(-1, 2)
        return grid * float(self.square_size_mm)

    def board_centre_mm(self) -> np.ndarray:
        """Centre of the inner-corner grid, in the board frame."""
        return np.array(
            [
                (self.cols - 1) * self.square_size_mm / 2.0,
                (self.rows - 1) * self.square_size_mm / 2.0,
                0.0,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CameraIntrinsics:
    """
    A calibrated camera's intrinsic parameters.

    Attributes
    ----------
    camera_matrix:
        The 3x3 matrix ``K``, with focal lengths in pixels on the diagonal and
        the principal point in the last column.
    distortion_coefficients:
        Five-element ``(k1, k2, p1, p2, k3)``. ``k*`` are radial, ``p*``
        tangential.
    image_size:
        ``(width, height)`` in pixels. Intrinsics are only valid for the
        resolution they were measured at -- changing capture resolution
        invalidates them.
    rms_reprojection_error:
        RMS distance, in pixels, between detected corners and corners
        reprojected through the fitted model. Under ~0.5 px is a good
        calibration; over ~1.0 px means the frames were too few, too similar,
        or blurred.
    """

    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_size: Tuple[int, int]
    rms_reprojection_error: float

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"camera_matrix must be 3x3, got shape {matrix.shape}.",
            )
        coefficients = np.asarray(
            self.distortion_coefficients, dtype=np.float64
        ).ravel()
        if coefficients.size != 5:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"distortion_coefficients must have 5 elements "
                f"(k1, k2, p1, p2, k3), got {coefficients.size}.",
            )
        if len(self.image_size) != 2 or any(int(v) <= 0 for v in self.image_size):
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"image_size must be a positive (width, height), got "
                f"{self.image_size!r}.",
            )
        if self.rms_reprojection_error < 0.0:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"rms_reprojection_error must be non-negative, got "
                f"{self.rms_reprojection_error}.",
            )
        # Normalise stored representations so equality and indexing are
        # predictable regardless of what the caller passed in.
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "distortion_coefficients", coefficients)
        object.__setattr__(
            self, "image_size", (int(self.image_size[0]), int(self.image_size[1]))
        )
        object.__setattr__(
            self, "rms_reprojection_error", float(self.rms_reprojection_error)
        )

    # ---- Convenience accessors -------------------------------------------

    @property
    def fx(self) -> float:
        """Focal length along X, in pixels."""
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        """Focal length along Y, in pixels."""
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        """Principal point X, in pixels."""
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        """Principal point Y, in pixels."""
        return float(self.camera_matrix[1, 2])

    @property
    def image_width(self) -> int:
        return self.image_size[0]

    @property
    def image_height(self) -> int:
        return self.image_size[1]

    # ---- Persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-Python representation, suitable for ``json.dump``."""
        return {
            "schema": INTRINSICS_SCHEMA_VERSION,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.distortion_coefficients.tolist(),
            "image_size": list(self.image_size),
            "rms_reprojection_error": self.rms_reprojection_error,
        }

    def save_json(self, path: Path) -> Path:
        """
        Write the intrinsics to ``path`` as JSON, creating parent directories.

        Returns the path actually written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load_json(cls, path: Path) -> "CameraIntrinsics":
        """
        Read intrinsics previously written by :meth:`save_json`.

        Raises
        ------
        CalibrationError
            With :attr:`CalibrationStatus.INVALID_INTRINSICS` if the file is
            not valid JSON, is missing a field, or holds a schema version this
            code does not understand.
        """
        path = Path(path)
        try:
            payload = json.loads(path.read_text())
        except FileNotFoundError as exc:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"No intrinsics file at {path}.",
            ) from exc
        except json.JSONDecodeError as exc:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"{path} is not valid JSON: {exc}.",
            ) from exc

        if not isinstance(payload, dict):
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"{path} must contain a JSON object, got "
                f"{type(payload).__name__}.",
            )
        schema = payload.get("schema")
        if schema != INTRINSICS_SCHEMA_VERSION:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"{path} declares schema {schema!r}, but this code understands "
                f"schema {INTRINSICS_SCHEMA_VERSION}.",
            )
        missing = {
            "camera_matrix",
            "distortion_coefficients",
            "image_size",
            "rms_reprojection_error",
        } - payload.keys()
        if missing:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"{path} is missing required field(s): {sorted(missing)}.",
            )
        try:
            return cls(
                camera_matrix=np.asarray(payload["camera_matrix"], dtype=np.float64),
                distortion_coefficients=np.asarray(
                    payload["distortion_coefficients"], dtype=np.float64
                ),
                image_size=tuple(payload["image_size"]),
                rms_reprojection_error=payload["rms_reprojection_error"],
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationError(
                CalibrationStatus.INVALID_INTRINSICS,
                f"{path} holds malformed intrinsics: {exc}.",
            ) from exc

    def summary(self) -> str:
        """Human-readable one-block summary."""
        return (
            f"CameraIntrinsics\n"
            f"----------------\n"
            f"  Image size    : {self.image_width} x {self.image_height} px\n"
            f"  Focal length  : fx = {self.fx:.2f}, fy = {self.fy:.2f} px\n"
            f"  Principal pt  : cx = {self.cx:.2f}, cy = {self.cy:.2f} px\n"
            f"  Distortion    : k1 = {self.distortion_coefficients[0]:+.5f}, "
            f"k2 = {self.distortion_coefficients[1]:+.5f}, "
            f"k3 = {self.distortion_coefficients[4]:+.5f}\n"
            f"                  p1 = {self.distortion_coefficients[2]:+.5f}, "
            f"p2 = {self.distortion_coefficients[3]:+.5f}\n"
            f"  RMS reproj.   : {self.rms_reprojection_error:.4f} px\n"
        )


def _as_grayscale(frame: np.ndarray, context: str) -> np.ndarray:
    """
    Coerce a frame to single-channel 8-bit, or explain why it cannot be.

    Accepts the BGR frames :class:`~src.image_source.ImageSource` produces and
    plain grayscale alike, so callers are not forced to convert first.
    """
    if not isinstance(frame, np.ndarray):
        raise CalibrationError(
            CalibrationStatus.MALFORMED_FRAME,
            f"{context}: expected np.ndarray, got {type(frame).__name__}.",
        )
    if frame.dtype != np.uint8:
        raise CalibrationError(
            CalibrationStatus.MALFORMED_FRAME,
            f"{context}: expected an 8-bit image, got dtype {frame.dtype}.",
        )
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise CalibrationError(
        CalibrationStatus.MALFORMED_FRAME,
        f"{context}: expected (H, W) grayscale or (H, W, 3) BGR, got shape "
        f"{frame.shape}.",
    )


class Calibrator:
    """
    Accumulates chessboard views and solves for the camera's intrinsics.

    Typical use::

        calibrator = Calibrator(CalibrationTarget(rows=6, cols=9))
        for frame in frames:
            calibrator.add_frame(frame)          # True if the board was seen
        intrinsics = calibrator.calibrate()
        clean = Calibrator.undistort(frame, intrinsics)

    Parameters
    ----------
    target:
        The chessboard being photographed.
    min_frames:
        Fewest accumulated views :meth:`calibrate` will accept. OpenCV can
        solve from 3 planar views in principle, but the result is badly
        conditioned; 5 is a floor, and 15-20 varied poses is what actually
        produces a trustworthy calibration.
    subpixel_window:
        Half-size of the ``cornerSubPix`` search window, in pixels. Must be
        small enough not to span neighbouring corners.
    fix_k3:
        Hold the sixth-order radial term at zero instead of fitting it.

        ``k1``, ``k2`` and ``k3`` are strongly correlated over the radius
        range a real target covers, so fitting all three tends to produce
        large, offsetting coefficients that describe the same distortion as a
        smaller, better-conditioned pair. For ordinary (non-fisheye) lenses,
        ``k3`` contributes almost nothing and fixing it at zero gives a more
        stable fit. Defaults to False so behaviour matches stock
        ``cv2.calibrateCamera``; set it True for the webcam or Pi Camera.

        Note that the *individual* coefficients are only weakly identified
        either way. What is reliably recovered is the distortion **field** --
        how far each pixel moves -- which is all the downstream homography
        needs. ``tests/test_camera_calibration.py`` asserts on that field
        rather than on coefficient values for this reason.
    """

    #: Detector flags. Adaptive thresholding copes with uneven desk lighting;
    #: normalisation helps when the board is under- or over-exposed.
    DETECT_FLAGS = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )

    #: Termination criteria for the sub-pixel corner refinement.
    SUBPIX_CRITERIA = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    def __init__(
        self,
        target: Optional[CalibrationTarget] = None,
        min_frames: int = 5,
        subpixel_window: Tuple[int, int] = (11, 11),
        fix_k3: bool = False,
    ) -> None:
        if min_frames < 3:
            raise CalibrationError(
                CalibrationStatus.INVALID_TARGET,
                f"min_frames must be at least 3 for a solvable system, got "
                f"{min_frames}.",
            )
        self._target = CalibrationTarget() if target is None else target
        self._min_frames = int(min_frames)
        self._subpixel_window = subpixel_window
        self._fix_k3 = bool(fix_k3)
        self._image_points: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None

    # ---- State ------------------------------------------------------------

    @property
    def target(self) -> CalibrationTarget:
        return self._target

    @property
    def frame_count(self) -> int:
        """Views accumulated so far (frames where the board was found)."""
        return len(self._image_points)

    @property
    def min_frames(self) -> int:
        return self._min_frames

    @property
    def image_size(self) -> Optional[Tuple[int, int]]:
        """``(width, height)`` fixed by the first accepted frame, else None."""
        return self._image_size

    @property
    def fix_k3(self) -> bool:
        """Whether the sixth-order radial term is held at zero."""
        return self._fix_k3

    @property
    def calibration_flags(self) -> int:
        """OpenCV solver flags implied by this calibrator's configuration."""
        return cv2.CALIB_FIX_K3 if self._fix_k3 else 0

    def reset(self) -> None:
        """Discard every accumulated view."""
        self._image_points.clear()
        self._image_size = None

    def remove_last_frame(self) -> bool:
        """
        Drop the most recently accumulated view.

        Returns True if a view was removed, False if none had been
        accumulated. Exists so an interactive tool can offer undo without
        reaching into the calibrator's internals; the accepted frame size is
        retained, since it is a property of the camera rather than of any one
        view.
        """
        if not self._image_points:
            return False
        self._image_points.pop()
        return True

    # ---- Detection --------------------------------------------------------

    def find_corners(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Locate the chessboard's inner corners to sub-pixel accuracy.

        Returns an ``(N, 1, 2)`` float32 array of pixel coordinates, or None if
        no complete board was found. Exposed separately from :meth:`add_frame`
        so an interactive tool can draw a live detection overlay without
        committing the frame.

        Raises
        ------
        CalibrationError
            With :attr:`CalibrationStatus.MALFORMED_FRAME` if the frame is not
            a usable 8-bit image.
        """
        gray = _as_grayscale(frame, "find_corners")
        found, corners = cv2.findChessboardCorners(
            gray, self._target.pattern_size, flags=self.DETECT_FLAGS
        )
        if not found:
            return None
        return cv2.cornerSubPix(
            gray, corners, self._subpixel_window, (-1, -1), self.SUBPIX_CRITERIA
        )

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Detect the chessboard in ``frame`` and accumulate it if found.

        Returns True when the board was detected and the view was kept, False
        when the frame simply did not contain a complete board -- a routine
        outcome while sweeping a target around, not an error.

        Raises
        ------
        CalibrationError
            :attr:`CalibrationStatus.MALFORMED_FRAME` if the frame is not a
            usable 8-bit image, or
            :attr:`CalibrationStatus.FRAME_SIZE_MISMATCH` if its dimensions
            differ from frames already accumulated. Intrinsics are only
            meaningful for a single resolution, so mixing sizes would silently
            corrupt the fit.
        """
        gray = _as_grayscale(frame, "add_frame")
        height, width = gray.shape[:2]
        size = (width, height)

        if self._image_size is None:
            self._image_size = size
        elif size != self._image_size:
            raise CalibrationError(
                CalibrationStatus.FRAME_SIZE_MISMATCH,
                f"add_frame received a {width}x{height} frame but earlier "
                f"frames were {self._image_size[0]}x{self._image_size[1]}. "
                "Intrinsics are only valid for one resolution.",
            )

        corners = self.find_corners(gray)
        if corners is None:
            return False
        self._image_points.append(corners)
        return True

    # ---- Solving ----------------------------------------------------------

    def calibrate(self) -> CameraIntrinsics:
        """
        Solve for the intrinsics from every accumulated view.

        Returns
        -------
        CameraIntrinsics
            The fitted model, including its RMS reprojection error.

        Raises
        ------
        CalibrationError
            :attr:`CalibrationStatus.TOO_FEW_FRAMES` if fewer than
            :attr:`min_frames` views were accumulated, or
            :attr:`CalibrationStatus.SOLVER_FAILED` if OpenCV could not
            produce a usable result.
        """
        if self.frame_count < self._min_frames:
            raise CalibrationError(
                CalibrationStatus.TOO_FEW_FRAMES,
                f"Calibration needs at least {self._min_frames} views with a "
                f"detected board; only {self.frame_count} accumulated. Move "
                "the target to more angles and distances and capture again.",
            )
        assert self._image_size is not None  # guaranteed once a frame is added

        object_points = [self._target.object_points()] * self.frame_count
        try:
            rms, matrix, coefficients, _, _ = cv2.calibrateCamera(
                object_points,
                self._image_points,
                self._image_size,
                None,
                None,
                flags=self.calibration_flags,
            )
        except cv2.error as exc:
            raise CalibrationError(
                CalibrationStatus.SOLVER_FAILED,
                f"cv2.calibrateCamera failed: {exc}",
            ) from exc

        if matrix is None or not np.all(np.isfinite(matrix)):
            raise CalibrationError(
                CalibrationStatus.SOLVER_FAILED,
                "cv2.calibrateCamera returned a non-finite camera matrix. The "
                "views are probably too similar to constrain the model.",
            )

        # OpenCV returns however many coefficients its default model used;
        # normalise to the 5-element form this project stores.
        coefficients = np.asarray(coefficients, dtype=np.float64).ravel()
        if coefficients.size < 5:
            coefficients = np.pad(coefficients, (0, 5 - coefficients.size))
        coefficients = coefficients[:5]

        return CameraIntrinsics(
            camera_matrix=matrix,
            distortion_coefficients=coefficients,
            image_size=self._image_size,
            rms_reprojection_error=float(rms),
        )

    # ---- Application ------------------------------------------------------

    @staticmethod
    def undistort(frame: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
        """
        Remove lens distortion from ``frame`` using ``intrinsics``.

        The result keeps the same camera matrix, so pixel coordinates in the
        undistorted image are directly comparable with the ideal pinhole
        projection -- which is what the Session B.2 homography needs.

        Raises
        ------
        CalibrationError
            :attr:`CalibrationStatus.MALFORMED_FRAME` if the frame is not a
            usable 8-bit image, or
            :attr:`CalibrationStatus.FRAME_SIZE_MISMATCH` if its size differs
            from the size the intrinsics were measured at.
        """
        if not isinstance(frame, np.ndarray):
            raise CalibrationError(
                CalibrationStatus.MALFORMED_FRAME,
                f"undistort: expected np.ndarray, got {type(frame).__name__}.",
            )
        if frame.ndim not in (2, 3):
            raise CalibrationError(
                CalibrationStatus.MALFORMED_FRAME,
                f"undistort: expected a 2D or 3D image, got shape {frame.shape}.",
            )
        height, width = frame.shape[:2]
        if (width, height) != intrinsics.image_size:
            raise CalibrationError(
                CalibrationStatus.FRAME_SIZE_MISMATCH,
                f"undistort received a {width}x{height} frame but the "
                f"intrinsics were measured at "
                f"{intrinsics.image_width}x{intrinsics.image_height}.",
            )
        return cv2.undistort(
            frame, intrinsics.camera_matrix, intrinsics.distortion_coefficients
        )


#: Pose recipe for the synthetic target: (rx, ry, rz in degrees, u_frac,
#: v_frac, distance_mm). ``u_frac``/``v_frac`` place the board's centre in the
#: frame as a fraction of width/height.
#:
#: Chosen for calibration conditioning rather than variety for its own sake:
#: focal length and distance are only separable when the board is seen tilted,
#: and distortion and principal point are only constrained where the board
#: actually visits the frame's edges. A stack of parallel frontal views is the
#: classic way to get a confident, wrong answer.
_POSE_RECIPE: Tuple[Tuple[float, float, float, float, float, float], ...] = (
    (0.0, 0.0, 0.0, 0.50, 0.50, 520.0),
    (-24.0, 0.0, 0.0, 0.50, 0.45, 500.0),
    (24.0, 0.0, 0.0, 0.50, 0.55, 500.0),
    (0.0, -26.0, 0.0, 0.45, 0.50, 500.0),
    (0.0, 26.0, 0.0, 0.55, 0.50, 500.0),
    (-18.0, -18.0, 8.0, 0.32, 0.34, 470.0),
    (-18.0, 18.0, -8.0, 0.68, 0.34, 470.0),
    (18.0, -18.0, -8.0, 0.32, 0.66, 470.0),
    (18.0, 18.0, 8.0, 0.68, 0.66, 470.0),
    (0.0, 0.0, 18.0, 0.35, 0.50, 560.0),
    (0.0, 0.0, -18.0, 0.65, 0.50, 560.0),
    (-12.0, 22.0, 0.0, 0.50, 0.30, 480.0),
    (12.0, -22.0, 0.0, 0.50, 0.70, 480.0),
    (-8.0, -8.0, 0.0, 0.50, 0.50, 400.0),
    (8.0, 8.0, 0.0, 0.50, 0.50, 640.0),
)


def _rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic X-then-Y-then-Z rotation, in degrees."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)
    mat_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
    mat_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
    mat_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
    return mat_z @ mat_y @ mat_x


def default_calibration_poses(
    target: CalibrationTarget,
    camera_matrix: np.ndarray,
    image_size: Tuple[int, int],
    distortion_coefficients: Sequence[float] = DEFAULT_DISTORTION,
    margin_px: float = 12.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build a well-conditioned set of board poses for the synthetic target.

    Each pose is an ``(rvec, tvec)`` pair in OpenCV's convention. Poses whose
    corners would fall outside the frame (or within ``margin_px`` of its edge,
    where ``findChessboardCorners`` gets unreliable) are dropped rather than
    silently producing undetectable frames.

    Parameters
    ----------
    target:
        The board being posed.
    camera_matrix, distortion_coefficients:
        The ground-truth camera the poses are checked against.
    image_size:
        ``(width, height)`` in pixels.
    margin_px:
        Keep-out band at the frame edge.

    Returns
    -------
    list of (rvec, tvec)
        Ready to hand to :class:`SyntheticChessboardSource`.
    """
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion_coefficients, dtype=np.float64).ravel()
    width, height = image_size
    focal_x, focal_y = camera_matrix[0, 0], camera_matrix[1, 1]
    centre_x, centre_y = camera_matrix[0, 2], camera_matrix[1, 2]
    board_centre = target.board_centre_mm()
    object_points = target.object_points().astype(np.float64)

    poses: List[Tuple[np.ndarray, np.ndarray]] = []
    for rx, ry, rz, u_frac, v_frac, distance in _POSE_RECIPE:
        rotation = _rotation_matrix(rx, ry, rz)
        # Place the board's centre so it projects to (u, v) at this distance.
        target_u, target_v = u_frac * width, v_frac * height
        centre_cam = np.array(
            [
                (target_u - centre_x) * distance / focal_x,
                (target_v - centre_y) * distance / focal_y,
                distance,
            ]
        )
        translation = centre_cam - rotation @ board_centre
        rvec, _ = cv2.Rodrigues(rotation)
        tvec = translation.reshape(3, 1)

        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, coefficients
        )
        pixels = projected.reshape(-1, 2)
        if (
            pixels[:, 0].min() >= margin_px
            and pixels[:, 1].min() >= margin_px
            and pixels[:, 0].max() <= width - margin_px
            and pixels[:, 1].max() <= height - margin_px
        ):
            poses.append((rvec, tvec))
    return poses


class SyntheticChessboardSource(ImageSource):
    """
    Renders a chessboard through *known* intrinsics. No camera required.

    Each square's outline is projected with ``cv2.projectPoints`` and filled --
    no ray tracing, no shading model. Because the projection applies the exact
    distortion coefficients supplied, the rendered image is what the modelled
    camera would have produced, so calibration run over these frames can be
    checked against ground truth rather than merely checked for convergence.

    Two details make the rendering accurate enough for sub-pixel corner
    detection, which is what the accuracy assertions depend on:

    - **Edge subdivision.** Distortion bends straight lines, so a square's edge
      is not a straight line in image space. Each edge is projected as
      ``edge_subdivisions`` short segments rather than one, tracking the curve.
    - **Supersampling.** The scene is rendered at ``supersample`` times the
      requested resolution and box-filtered down, which anti-aliases the
      square edges. Hard-edged binary rendering biases ``cornerSubPix``.

    Parameters
    ----------
    target:
        The board to draw.
    camera_matrix:
        Ground-truth 3x3 ``K`` for the modelled camera.
    distortion_coefficients:
        Ground-truth ``(k1, k2, p1, p2, k3)``.
    image_size:
        ``(width, height)`` in pixels.
    poses:
        ``(rvec, tvec)`` pairs, one per frame. Frames cycle through this list.
        Defaults to :func:`default_calibration_poses`.
    light_gray, dark_gray:
        Grey levels for the board's two square colours. The light value also
        fills the quiet zone around the board, which the detector needs.
    supersample:
        Render scale factor. 1 disables supersampling. The default 6 costs
        about 2 ms per frame and holds corner detection error near 0.08 px;
        dropping to 3 roughly doubles that.
    edge_subdivisions:
        Segments per square edge.
    """

    def __init__(
        self,
        target: Optional[CalibrationTarget] = None,
        camera_matrix: Optional[np.ndarray] = None,
        distortion_coefficients: Sequence[float] = DEFAULT_DISTORTION,
        image_size: Tuple[int, int] = (640, 480),
        poses: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
        light_gray: int = 235,
        dark_gray: int = 25,
        supersample: int = 6,
        edge_subdivisions: int = 8,
    ) -> None:
        super().__init__(name="synthetic-chessboard")

        self._target = CalibrationTarget() if target is None else target

        width, height = int(image_size[0]), int(image_size[1])
        if width < 1 or height < 1:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"image_size must be positive, got {image_size!r}.",
            )
        if supersample < 1:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"supersample must be at least 1, got {supersample}.",
            )
        if edge_subdivisions < 1:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"edge_subdivisions must be at least 1, got {edge_subdivisions}.",
            )
        if not (0 <= dark_gray < light_gray <= 255):
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"require 0 <= dark_gray ({dark_gray}) < light_gray "
                f"({light_gray}) <= 255.",
            )

        if camera_matrix is None:
            camera_matrix = np.array(
                [
                    [800.0, 0.0, width / 2.0 - 0.5],
                    [0.0, 800.0, height / 2.0 - 0.5],
                    [0.0, 0.0, 1.0],
                ]
            )
        self._camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        if self._camera_matrix.shape != (3, 3):
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"camera_matrix must be 3x3, got {self._camera_matrix.shape}.",
            )
        self._distortion = np.asarray(
            distortion_coefficients, dtype=np.float64
        ).ravel()
        if self._distortion.size != 5:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"distortion_coefficients must have 5 elements, got "
                f"{self._distortion.size}.",
            )

        self._image_size = (width, height)
        self._frame_shape = (height, width, self.CHANNELS)
        self._light_gray = int(light_gray)
        self._dark_gray = int(dark_gray)
        self._supersample = int(supersample)
        self._edge_subdivisions = int(edge_subdivisions)

        if poses is None:
            poses = default_calibration_poses(
                self._target, self._camera_matrix, self._image_size, self._distortion
            )
        self._poses = list(poses)
        if not self._poses:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                "No usable poses: every candidate placed the board partly "
                "outside the frame. Move the camera back or use a smaller "
                "board.",
            )
        self._pose_index = 0

        self._dark_square_outlines = self._build_dark_square_outlines()

    # ---- ImageSource contract --------------------------------------------

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Shape ``(height, width, 3)``. Known at construction; never raises."""
        return self._frame_shape

    def _open_impl(self) -> None:
        """No device to acquire; rendering is on demand."""

    def _close_impl(self) -> None:
        """No device to release."""

    def _read_frame(self) -> np.ndarray:
        frame = self.render_pose(self._pose_index)
        self._pose_index = (self._pose_index + 1) % len(self._poses)
        return frame

    # ---- Synthetic-source specifics --------------------------------------

    @property
    def target(self) -> CalibrationTarget:
        return self._target

    @property
    def camera_matrix(self) -> np.ndarray:
        """Ground-truth ``K`` these frames were rendered through."""
        return self._camera_matrix.copy()

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Ground-truth ``(k1, k2, p1, p2, k3)`` applied to these frames."""
        return self._distortion.copy()

    @property
    def pose_count(self) -> int:
        """How many distinct frames this source cycles through."""
        return len(self._poses)

    @property
    def poses(self) -> Tuple[Tuple[np.ndarray, np.ndarray], ...]:
        return tuple(self._poses)

    def ground_truth_intrinsics(self) -> CameraIntrinsics:
        """
        The intrinsics a perfect calibration would recover from these frames.

        RMS error is reported as zero: this is the exact model the renderer
        used, not a fit to it.
        """
        return CameraIntrinsics(
            camera_matrix=self._camera_matrix,
            distortion_coefficients=self._distortion,
            image_size=self._image_size,
            rms_reprojection_error=0.0,
        )

    def project_corners(self, pose_index: int) -> np.ndarray:
        """
        Exact pixel positions of the board's inner corners for one pose.

        Shape ``(N, 2)``. This is the ground truth a detector should recover,
        so it doubles as a way to measure detection accuracy directly.
        """
        rvec, tvec = self._poses[pose_index % len(self._poses)]
        projected, _ = cv2.projectPoints(
            self._target.object_points().astype(np.float64),
            rvec,
            tvec,
            self._camera_matrix,
            self._distortion,
        )
        return projected.reshape(-1, 2)

    def _build_dark_square_outlines(self) -> List[np.ndarray]:
        """
        Board-frame outlines of every dark square, subdivided along each edge.

        Only the dark squares are drawn; the light squares are the background
        the frame is cleared to, which also gives the board the light quiet
        zone the detector needs around its border.
        """
        square = self._target.square_size_mm
        steps = self._edge_subdivisions
        outlines: List[np.ndarray] = []

        # Squares are indexed so that inner corner (0, 0) sits at the junction
        # of squares (0, 0) and (1, 1); the board therefore spans one extra
        # square in each direction.
        for i in range(self._target.cols + 1):
            for j in range(self._target.rows + 1):
                if (i + j) % 2 != 0:
                    continue
                x0, x1 = (i - 1) * square, i * square
                y0, y1 = (j - 1) * square, j * square

                edge = np.linspace(0.0, 1.0, steps + 1)[:-1]
                top = np.stack([x0 + (x1 - x0) * edge, np.full(steps, y0)], axis=1)
                right = np.stack([np.full(steps, x1), y0 + (y1 - y0) * edge], axis=1)
                bottom = np.stack(
                    [x1 + (x0 - x1) * edge, np.full(steps, y1)], axis=1
                )
                left = np.stack([np.full(steps, x0), y1 + (y0 - y1) * edge], axis=1)

                loop = np.concatenate([top, right, bottom, left], axis=0)
                outlines.append(
                    np.concatenate(
                        [loop, np.zeros((loop.shape[0], 1))], axis=1
                    ).astype(np.float64)
                )
        return outlines

    def render_pose(self, pose_index: int) -> np.ndarray:
        """
        Render one pose to a BGR uint8 frame.

        Deterministic: the same index always yields byte-identical output.
        """
        rvec, tvec = self._poses[pose_index % len(self._poses)]
        scale = self._supersample
        width, height = self._image_size

        # Scaling K for supersampling is not simply multiplying by the factor:
        # box-filtering an s-by-s block maps output pixel u to input pixel
        # u*s + (s-1)/2, so the principal point picks up that half-block shift.
        scaled_matrix = self._camera_matrix.copy()
        scaled_matrix[0, 0] *= scale
        scaled_matrix[1, 1] *= scale
        scaled_matrix[0, 2] = self._camera_matrix[0, 2] * scale + (scale - 1) / 2.0
        scaled_matrix[1, 2] = self._camera_matrix[1, 2] * scale + (scale - 1) / 2.0

        canvas = np.full(
            (height * scale, width * scale), self._light_gray, dtype=np.uint8
        )
        # fillPoly's fixed-point mode: coordinates carry SHIFT fractional bits,
        # so squares land on sub-pixel boundaries instead of snapping to the
        # supersampled grid.
        shift = 4
        multiplier = 1 << shift
        for outline in self._dark_square_outlines:
            projected, _ = cv2.projectPoints(
                outline, rvec, tvec, scaled_matrix, self._distortion
            )
            polygon = np.round(projected.reshape(-1, 2) * multiplier).astype(np.int32)
            cv2.fillPoly(
                canvas, [polygon], self._dark_gray, lineType=cv2.LINE_8, shift=shift
            )

        if scale > 1:
            canvas = cv2.resize(
                canvas, (width, height), interpolation=cv2.INTER_AREA
            )
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


if __name__ == "__main__":
    # Smoke test, mirroring the `python3 -m src.<module>` convention used by
    # the other modules. Synthetic only -- no camera required.
    _source = SyntheticChessboardSource()
    _truth = _source.ground_truth_intrinsics()
    _calibrator = Calibrator(_source.target)

    with _source:
        for _index in range(_source.pose_count):
            _calibrator.add_frame(_source.get_frame())

    print("Synthetic calibration")
    print("---------------------")
    print(f"  Target        : {_source.target.cols} x {_source.target.rows} inner "
          f"corners, {_source.target.square_size_mm:.0f} mm squares")
    print(f"  Views used    : {_calibrator.frame_count} / {_source.pose_count}")
    print()

    _recovered = _calibrator.calibrate()
    print(_recovered.summary())

    print("  Recovery against exact ground truth")
    for _name in ("fx", "fy", "cx", "cy"):
        _t, _g = getattr(_truth, _name), getattr(_recovered, _name)
        print(f"    {_name:<3}: truth {_t:8.3f}  recovered {_g:8.3f}  "
              f"error {abs(_g - _t) / abs(_t) * 100:6.3f} %")

    # Individual distortion coefficients are only weakly identified; the field
    # they produce is what downstream code consumes, so compare that instead.
    _w, _h = _truth.image_size
    _gx, _gy = np.meshgrid(
        np.arange(0, _w, 16, dtype=np.float64),
        np.arange(0, _h, 16, dtype=np.float64),
    )
    _pts = np.stack([_gx.ravel(), _gy.ravel()], axis=1).reshape(-1, 1, 2)
    _a = cv2.undistortPoints(
        _pts, _truth.camera_matrix, _truth.distortion_coefficients,
        P=_truth.camera_matrix,
    ).reshape(-1, 2)
    _b = cv2.undistortPoints(
        _pts, _recovered.camera_matrix, _recovered.distortion_coefficients,
        P=_recovered.camera_matrix,
    ).reshape(-1, 2)
    _deviation = np.linalg.norm(_a - _b, axis=1)
    print()
    print("  Distortion field agreement (what actually matters)")
    print(f"    mean {_deviation.mean():.4f} px, max {_deviation.max():.4f} px")
