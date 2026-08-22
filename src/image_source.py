"""
Image acquisition abstraction for the Phase B vision stack.

Every vision consumer (ArUco calibration, YOLO detection, homography fitting)
takes an :class:`ImageSource` rather than touching ``cv2.VideoCapture``
directly. That indirection buys two things:

1. **Deterministic tests.** :class:`SyntheticImageSource` renders frames
   programmatically, so the vision pipeline can be exercised in CI on a
   machine with no camera and byte-identical results every run.
2. **Hardware swap-out.** Moving from the Mac webcam to the Raspberry Pi
   camera module later means adding one subclass, not editing call sites.

Coordinate conventions
----------------------
Consistent with ``docs/PROOF_OF_CONCEPT.md`` section 3:

- **Desk frame**: origin at a desk corner, ``+X`` along the 1200 mm edge,
  ``+Y`` along the 600 mm edge, millimeters.
- **Pixel frame**: origin top-left, ``+u`` right, ``+v`` down, pixels.

:class:`SyntheticImageSource` models a rectified overhead camera, so it maps
``+X_desk -> +u`` and ``+Y_desk -> +v`` with a single uniform scale. A real
overhead camera will not be rectified; the ``pixel -> desk`` homography fitted
in Session B.2 absorbs the difference. The synthetic mapping is exposed via
:meth:`SyntheticImageSource.desk_mm_to_pixel` so that homography can be
validated against a known ground truth.

Units are millimeters at the desk-facing API and pixels internally; conversion
happens only in :meth:`SyntheticImageSource.desk_mm_to_pixel` and its inverse.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from src.geometry import DEFAULT_ARM, ArmGeometry

__all__ = [
    "SourceStatus",
    "ImageSourceError",
    "ImageSource",
    "DeskObject",
    "SyntheticImageSource",
    "WebcamImageSource",
]


class SourceStatus(Enum):
    """
    Structured failure modes for image acquisition.

    Every :class:`ImageSourceError` carries one of these so callers can branch
    on the failure mode instead of string-matching an exception message. This
    mirrors the ``IKStatus`` pattern established in ``src.inverse_kinematics``.
    """

    OK = auto()
    """No error. Present for symmetry; never attached to an exception."""

    NOT_OPENED = auto()
    """A frame was requested from a source that is not currently open."""

    DEVICE_UNAVAILABLE = auto()
    """The backing capture device could not be opened at all."""

    READ_FAILED = auto()
    """The device is open but returned no frame (unplugged mid-stream, timeout)."""

    INVALID_CONFIG = auto()
    """Constructor arguments are self-inconsistent or out of range."""

    MALFORMED_FRAME = auto()
    """A frame was produced but failed dtype / shape validation."""


class ImageSourceError(RuntimeError):
    """
    Raised by every :class:`ImageSource` failure path.

    Attributes
    ----------
    status:
        The :class:`SourceStatus` describing *why* the operation failed.
    """

    def __init__(self, status: SourceStatus, message: str) -> None:
        super().__init__(f"[{status.name}] {message}")
        self.status = status


class ImageSource(ABC):
    """
    Abstract base for anything that can hand back BGR frames.

    Subclasses implement three hooks -- :meth:`_open_impl`, :meth:`_read_frame`
    and :meth:`_close_impl`. The public :meth:`open` / :meth:`get_frame` /
    :meth:`close` methods wrap them with lifecycle tracking and frame
    validation so that every source fails the same way for the same reasons.

    Both :meth:`open` and :meth:`close` are idempotent; calling either twice is
    a no-op rather than an error, which keeps cleanup paths simple.

    Frames are ``np.uint8`` arrays of shape ``(height, width, 3)`` in **BGR**
    channel order, matching OpenCV's convention.
    """

    #: Number of colour channels every source must produce (BGR).
    CHANNELS: int = 3

    #: Element type every source must produce.
    DTYPE = np.uint8

    def __init__(self, name: str) -> None:
        self._name = name
        self._is_open = False

    # ---- Identity ---------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable label, used in error messages and log lines."""
        return self._name

    @property
    def is_open(self) -> bool:
        """True between a successful :meth:`open` and the next :meth:`close`."""
        return self._is_open

    @property
    @abstractmethod
    def frame_shape(self) -> Tuple[int, int, int]:
        """
        Shape ``(height, width, channels)`` of frames from this source.

        Raises
        ------
        ImageSourceError
            With :attr:`SourceStatus.NOT_OPENED` if the shape is only knowable
            after the device has been probed and the source is not yet open.
        """

    # ---- Lifecycle --------------------------------------------------------

    def open(self) -> None:
        """
        Acquire the backing device. Idempotent.

        Raises
        ------
        ImageSourceError
            With :attr:`SourceStatus.DEVICE_UNAVAILABLE` if acquisition fails.
        """
        if self._is_open:
            return
        self._open_impl()
        self._is_open = True

    def close(self) -> None:
        """Release the backing device. Idempotent and safe to call in ``finally``."""
        if not self._is_open:
            return
        try:
            self._close_impl()
        finally:
            self._is_open = False

    def get_frame(self) -> np.ndarray:
        """
        Return the next frame as a ``(height, width, 3)`` uint8 BGR array.

        Raises
        ------
        ImageSourceError
            :attr:`SourceStatus.NOT_OPENED` if the source is closed,
            :attr:`SourceStatus.READ_FAILED` if the device yielded no frame,
            :attr:`SourceStatus.MALFORMED_FRAME` if the frame failed validation.
        """
        if not self._is_open:
            raise ImageSourceError(
                SourceStatus.NOT_OPENED,
                f"{self._name}: get_frame() called before open() (or after close()).",
            )
        frame = self._read_frame()
        self._validate_frame(frame)
        return frame

    # ---- Validation -------------------------------------------------------

    def _validate_frame(self, frame: object) -> None:
        """
        Assert the subclass honoured the frame contract.

        This is a guard against subclass bugs, not against user input, but it
        is cheap relative to a frame read and it converts a confusing
        downstream crash inside OpenCV into a precise error here.
        """
        if not isinstance(frame, np.ndarray):
            raise ImageSourceError(
                SourceStatus.MALFORMED_FRAME,
                f"{self._name}: expected np.ndarray, got {type(frame).__name__}.",
            )
        if frame.dtype != self.DTYPE:
            raise ImageSourceError(
                SourceStatus.MALFORMED_FRAME,
                f"{self._name}: expected dtype {np.dtype(self.DTYPE)}, "
                f"got {frame.dtype}.",
            )
        expected = self.frame_shape
        if frame.shape != expected:
            raise ImageSourceError(
                SourceStatus.MALFORMED_FRAME,
                f"{self._name}: expected frame shape {expected}, got {frame.shape}.",
            )

    # ---- Subclass hooks ---------------------------------------------------

    @abstractmethod
    def _open_impl(self) -> None:
        """Acquire the device. Called only when the source is currently closed."""

    @abstractmethod
    def _read_frame(self) -> np.ndarray:
        """Produce one frame. Called only when the source is open."""

    @abstractmethod
    def _close_impl(self) -> None:
        """Release the device. Called only when the source is currently open."""

    # ---- Context manager --------------------------------------------------

    def __enter__(self) -> "ImageSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "open" if self._is_open else "closed"
        return f"<{type(self).__name__} name={self._name!r} {state}>"


@dataclass(frozen=True)
class DeskObject:
    """
    An axis-aligned rectangular object lying on the synthetic desk.

    Positions and sizes are in **desk-frame millimeters**, so a test can say
    "a 60 x 15 mm pen at desk coordinates (400, 250)" and get the same pixels
    regardless of the chosen ``px_per_mm`` resolution.

    Attributes
    ----------
    x_mm, y_mm:
        Centre of the object in the desk frame.
    width_mm:
        Extent along desk ``+X``.
    height_mm:
        Extent along desk ``+Y``.
    color_bgr:
        Fill colour, OpenCV channel order, each component in ``[0, 255]``.
    label:
        Free-form tag. Unused by the renderer; carried so tests and, later,
        the detector-evaluation harness can assert on ground truth.
    """

    x_mm: float
    y_mm: float
    width_mm: float = 60.0
    height_mm: float = 15.0
    color_bgr: Tuple[int, int, int] = (40, 40, 200)
    label: str = "object"

    def __post_init__(self) -> None:
        if self.width_mm <= 0.0 or self.height_mm <= 0.0:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"DeskObject {self.label!r}: width_mm and height_mm must be "
                f"positive, got {self.width_mm} x {self.height_mm}.",
            )
        if len(self.color_bgr) != 3 or not all(
            0 <= int(c) <= 255 for c in self.color_bgr
        ):
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"DeskObject {self.label!r}: color_bgr must be three values in "
                f"[0, 255], got {self.color_bgr!r}.",
            )


class SyntheticImageSource(ImageSource):
    """
    Programmatically rendered overhead view of the desk. No hardware needed.

    The frame is a flat grey desk surface with zero or more :class:`DeskObject`
    rectangles drawn on it. Frame dimensions are **derived from**
    :data:`src.geometry.DEFAULT_ARM` -- ``desk_width_mm`` and ``desk_depth_mm``
    scaled by ``px_per_mm`` -- so the synthetic view's aspect ratio always
    matches the real workspace and changing the desk size in ``geometry.py``
    propagates here automatically.

    With the default ``noise_sigma=0.0`` the output is byte-identical on every
    call and every machine, which is what makes it usable as a test fixture.

    Parameters
    ----------
    objects:
        Objects to draw, in desk millimeters. Objects fully outside the desk
        are silently skipped; partially overlapping ones are clipped.
    px_per_mm:
        Resolution of the synthetic camera. The default 0.5 px/mm renders the
        1200 x 600 mm desk at 600 x 300 px.
    background_gray:
        Grey level of the desk surface, ``[0, 255]``.
    noise_sigma:
        Standard deviation of additive Gaussian sensor noise, in grey levels.
        Zero (the default) keeps frames deterministic across calls.
    seed:
        RNG seed used when ``noise_sigma > 0``. Fixing it makes a noisy source
        reproducible run-to-run, though successive frames still differ.
    arm:
        Geometry singleton supplying the desk dimensions. Injectable so tests
        can render a non-default desk without mutating the global.
    """

    def __init__(
        self,
        objects: Sequence[DeskObject] = (),
        px_per_mm: float = 0.5,
        background_gray: int = 128,
        noise_sigma: float = 0.0,
        seed: Optional[int] = None,
        arm: ArmGeometry = DEFAULT_ARM,
    ) -> None:
        super().__init__(name="synthetic")

        if px_per_mm <= 0.0:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"px_per_mm must be positive, got {px_per_mm}.",
            )
        if not 0 <= background_gray <= 255:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"background_gray must be in [0, 255], got {background_gray}.",
            )
        if noise_sigma < 0.0:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"noise_sigma must be non-negative, got {noise_sigma}.",
            )

        self._arm = arm
        self._px_per_mm = float(px_per_mm)
        self._background_gray = int(background_gray)
        self._noise_sigma = float(noise_sigma)
        self._objects = tuple(objects)
        self._rng = np.random.default_rng(seed)

        width_px = int(round(arm.desk_width_mm * self._px_per_mm))
        height_px = int(round(arm.desk_depth_mm * self._px_per_mm))
        if width_px < 1 or height_px < 1:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"px_per_mm={px_per_mm} renders the "
                f"{arm.desk_width_mm} x {arm.desk_depth_mm} mm desk at "
                f"{width_px} x {height_px} px; both must be at least 1.",
            )
        self._frame_shape = (height_px, width_px, self.CHANNELS)

        # The clean desk-plus-objects image is rendered once; per-frame work is
        # a copy (plus optional noise), so a 30 fps test loop stays cheap.
        self._base_frame = self._render_base_frame()

    # ---- ImageSource contract --------------------------------------------

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """Shape ``(height, width, 3)``. Known at construction; never raises."""
        return self._frame_shape

    def _open_impl(self) -> None:
        """No device to acquire; the base frame is already rendered."""

    def _close_impl(self) -> None:
        """No device to release."""

    def _read_frame(self) -> np.ndarray:
        frame = self._base_frame.copy()
        if self._noise_sigma > 0.0:
            noise = self._rng.normal(0.0, self._noise_sigma, size=frame.shape)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return frame

    # ---- Synthetic-source specifics --------------------------------------

    @property
    def px_per_mm(self) -> float:
        """Uniform desk-to-pixel scale of this synthetic camera."""
        return self._px_per_mm

    @property
    def objects(self) -> Tuple[DeskObject, ...]:
        """Ground-truth objects rendered into every frame."""
        return self._objects

    def desk_mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[float, float]:
        """
        Project a desk-frame point (mm) to pixel coordinates ``(u, v)``.

        This is the exact ground-truth mapping the Session B.2 homography must
        recover from ArUco correspondences, so tests can compare a fitted
        homography against it directly.
        """
        return (x_mm * self._px_per_mm, y_mm * self._px_per_mm)

    def pixel_to_desk_mm(self, u_px: float, v_px: float) -> Tuple[float, float]:
        """Inverse of :meth:`desk_mm_to_pixel`."""
        return (u_px / self._px_per_mm, v_px / self._px_per_mm)

    def _render_base_frame(self) -> np.ndarray:
        """Draw the desk surface and every object exactly once."""
        frame = np.full(self._frame_shape, self._background_gray, dtype=np.uint8)
        height_px, width_px = self._frame_shape[0], self._frame_shape[1]

        for obj in self._objects:
            u_min, v_min = self.desk_mm_to_pixel(
                obj.x_mm - obj.width_mm / 2.0, obj.y_mm - obj.height_mm / 2.0
            )
            u_max, v_max = self.desk_mm_to_pixel(
                obj.x_mm + obj.width_mm / 2.0, obj.y_mm + obj.height_mm / 2.0
            )

            # Clip to the frame; an object entirely off-desk contributes nothing.
            col0 = max(0, int(np.floor(u_min)))
            col1 = min(width_px, int(np.ceil(u_max)))
            row0 = max(0, int(np.floor(v_min)))
            row1 = min(height_px, int(np.ceil(v_max)))
            if col0 >= col1 or row0 >= row1:
                continue

            frame[row0:row1, col0:col1] = np.array(obj.color_bgr, dtype=np.uint8)

        return frame


class WebcamImageSource(ImageSource):
    """
    Live frames from a ``cv2.VideoCapture`` device.

    Written for the Mac's built-in webcam during development and the same code
    path on the Raspberry Pi later. Because CI has no camera, this class is
    covered by tests that inject a fake capture object rather than by tests
    that open real hardware.

    A requested resolution is a *hint*: capture backends routinely ignore
    ``CAP_PROP_FRAME_WIDTH`` / ``CAP_PROP_FRAME_HEIGHT`` and hand back their
    nearest supported mode. :meth:`open` therefore probes one real frame and
    reports the actual shape via :attr:`frame_shape`; compare it against
    :attr:`requested_resolution` to detect a silently-substituted mode.

    Parameters
    ----------
    device_index:
        Index passed to ``cv2.VideoCapture``. ``0`` is the default camera.
    width, height:
        Optional resolution hint in pixels. Both must be given together.
    fps:
        Optional frame-rate hint.
    warmup_frames:
        Frames to read and discard after opening. Many webcams (the Mac
        FaceTime camera included) need a moment of auto-exposure settling
        before their output is usable.
    backend:
        Optional ``cv2.CAP_*`` backend constant. Left unset, OpenCV chooses.
    """

    def __init__(
        self,
        device_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        warmup_frames: int = 0,
        backend: Optional[int] = None,
    ) -> None:
        super().__init__(name=f"webcam[{device_index}]")

        if (width is None) != (height is None):
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                "width and height must be supplied together or not at all "
                f"(got width={width}, height={height}).",
            )
        if width is not None and (width <= 0 or height <= 0):
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"Requested resolution must be positive, got {width} x {height}.",
            )
        if fps is not None and fps <= 0.0:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG, f"fps must be positive, got {fps}."
            )
        if warmup_frames < 0:
            raise ImageSourceError(
                SourceStatus.INVALID_CONFIG,
                f"warmup_frames must be non-negative, got {warmup_frames}.",
            )

        self._device_index = int(device_index)
        self._requested_width = width
        self._requested_height = height
        self._requested_fps = fps
        self._warmup_frames = int(warmup_frames)
        self._backend = backend

        self._capture: Optional[object] = None
        self._probed_shape: Optional[Tuple[int, int, int]] = None

    # ---- ImageSource contract --------------------------------------------

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        """
        Actual ``(height, width, 3)`` reported by the device.

        Raises
        ------
        ImageSourceError
            With :attr:`SourceStatus.NOT_OPENED` -- the true shape is only
            known once a frame has been probed, so it cannot be answered for a
            closed source.
        """
        if self._probed_shape is None:
            raise ImageSourceError(
                SourceStatus.NOT_OPENED,
                f"{self.name}: frame_shape is unknown until the device has "
                "been opened and probed.",
            )
        return self._probed_shape

    def _open_impl(self) -> None:
        capture = (
            cv2.VideoCapture(self._device_index)
            if self._backend is None
            else cv2.VideoCapture(self._device_index, self._backend)
        )

        if not capture.isOpened():
            capture.release()
            raise ImageSourceError(
                SourceStatus.DEVICE_UNAVAILABLE,
                f"{self.name}: cv2.VideoCapture could not open device "
                f"{self._device_index}. Check that a camera is connected and "
                "that this process has camera permission "
                "(macOS: System Settings > Privacy & Security > Camera).",
            )

        if self._requested_width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._requested_width))
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._requested_height))
        if self._requested_fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(self._requested_fps))

        self._capture = capture

        # Discard warm-up frames before probing, so the probed frame reflects a
        # settled exposure rather than the first dark frame.
        for _ in range(self._warmup_frames):
            capture.read()

        try:
            probe = self._read_frame()
        except ImageSourceError:
            capture.release()
            self._capture = None
            raise

        if probe.ndim != 3 or probe.shape[2] != self.CHANNELS:
            capture.release()
            self._capture = None
            raise ImageSourceError(
                SourceStatus.MALFORMED_FRAME,
                f"{self.name}: expected a 3-channel BGR frame, got shape "
                f"{probe.shape}.",
            )
        self._probed_shape = (int(probe.shape[0]), int(probe.shape[1]), self.CHANNELS)

    def _read_frame(self) -> np.ndarray:
        if self._capture is None:
            raise ImageSourceError(
                SourceStatus.NOT_OPENED, f"{self.name}: no capture device held."
            )
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise ImageSourceError(
                SourceStatus.READ_FAILED,
                f"{self.name}: device is open but returned no frame. The camera "
                "may have been disconnected or claimed by another process.",
            )
        return frame

    def _close_impl(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._probed_shape = None

    # ---- Webcam specifics -------------------------------------------------

    @property
    def device_index(self) -> int:
        """Index this source was constructed against."""
        return self._device_index

    @property
    def requested_resolution(self) -> Optional[Tuple[int, int]]:
        """The ``(width, height)`` hint given at construction, if any."""
        if self._requested_width is None:
            return None
        return (self._requested_width, int(self._requested_height))


if __name__ == "__main__":
    # Smoke test, mirroring the `python3 -m src.<module>` convention used by
    # the Phase A modules. Synthetic only -- no camera required.
    demo_objects = [
        DeskObject(x_mm=300.0, y_mm=200.0, label="pen"),
        DeskObject(
            x_mm=800.0,
            y_mm=420.0,
            width_mm=90.0,
            height_mm=90.0,
            color_bgr=(60, 180, 60),
            label="sticky_note_pad",
        ),
    ]
    with SyntheticImageSource(objects=demo_objects) as source:
        first = source.get_frame()
        second = source.get_frame()
        print("SyntheticImageSource")
        print("--------------------")
        print(f"  frame shape     : {first.shape}")
        print(f"  dtype           : {first.dtype}")
        print(f"  px_per_mm       : {source.px_per_mm}")
        print(f"  desk (mm)       : {DEFAULT_ARM.desk_width_mm} x "
              f"{DEFAULT_ARM.desk_depth_mm}")
        print(f"  deterministic   : {np.array_equal(first, second)}")
        for obj in source.objects:
            u, v = source.desk_mm_to_pixel(obj.x_mm, obj.y_mm)
            print(f"  {obj.label:<18}: desk ({obj.x_mm:.0f}, {obj.y_mm:.0f}) mm "
                  f"-> pixel ({u:.1f}, {v:.1f})")

    closed = SyntheticImageSource()
    try:
        closed.get_frame()
    except ImageSourceError as exc:
        print(f"  closed-source status: {exc.status.name}")
