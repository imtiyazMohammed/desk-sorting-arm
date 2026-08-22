"""
Test suite for the Session B.1 image-acquisition layer.

Test areas
----------
1. Abstract base-class contract (lifecycle, validation, context manager)
2. SyntheticImageSource -- shape/dtype, determinism, geometry coupling
3. SyntheticImageSource -- desk-mm to pixel projection and object rendering
4. WebcamImageSource -- driven entirely by an injected fake VideoCapture,
   so the suite runs on CI machines with no camera attached
5. Structured error handling -- every failure path carries a SourceStatus
"""

from __future__ import annotations

import numpy as np
import pytest

from src.geometry import DEFAULT_ARM, ArmGeometry
from src.image_source import (
    DeskObject,
    ImageSource,
    ImageSourceError,
    SourceStatus,
    SyntheticImageSource,
    WebcamImageSource,
)
import src.image_source as image_source_module


# =========================================================================
# Fakes
# =========================================================================


class FakeVideoCapture:
    """
    Stand-in for ``cv2.VideoCapture`` covering every branch of the webcam path.

    Parameters mirror the real failure modes: a device that will not open, a
    device that opens but never yields a frame, and a device that yields a
    frame of a resolution other than the one requested.
    """

    def __init__(
        self,
        index,
        backend=None,
        opened: bool = True,
        frame_shape=(480, 640, 3),
        fail_after: int | None = None,
    ) -> None:
        self.index = index
        self.backend = backend
        self._opened = opened
        self._frame_shape = frame_shape
        self._fail_after = fail_after
        self.read_count = 0
        self.released = False
        self.props: dict[int, float] = {}

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def set(self, prop_id, value) -> bool:  # noqa: A003 - mirrors the OpenCV API
        self.props[prop_id] = value
        return True

    def read(self):
        self.read_count += 1
        if self._fail_after is not None and self.read_count > self._fail_after:
            return False, None
        # A recognisable non-uniform frame, so shape assertions are meaningful.
        frame = np.full(self._frame_shape, 77, dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        self.released = True
        self._opened = False


@pytest.fixture
def patch_capture(monkeypatch):
    """
    Install a FakeVideoCapture factory in place of ``cv2.VideoCapture``.

    Returns a callable taking the same keyword arguments as FakeVideoCapture;
    calling it patches the module and returns a list that will receive every
    instance the code under test constructs.
    """

    def _install(**kwargs):
        created: list[FakeVideoCapture] = []

        def factory(index, backend=None):
            capture = FakeVideoCapture(index, backend, **kwargs)
            created.append(capture)
            return capture

        monkeypatch.setattr(image_source_module.cv2, "VideoCapture", factory)
        return created

    return _install


# =========================================================================
# 1. Abstract base-class contract
# =========================================================================


class TestImageSourceContract:
    def test_abstract_base_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ImageSource(name="nope")  # type: ignore[abstract]

    @pytest.mark.parametrize(
        "factory", [SyntheticImageSource, WebcamImageSource], ids=["synthetic", "webcam"]
    )
    def test_concrete_sources_are_image_sources(self, factory):
        assert issubclass(factory, ImageSource)

    def test_source_starts_closed(self):
        assert SyntheticImageSource().is_open is False

    def test_get_frame_before_open_raises_not_opened(self):
        source = SyntheticImageSource()
        with pytest.raises(ImageSourceError) as excinfo:
            source.get_frame()
        assert excinfo.value.status is SourceStatus.NOT_OPENED

    def test_get_frame_after_close_raises_not_opened(self):
        source = SyntheticImageSource()
        source.open()
        source.get_frame()
        source.close()
        with pytest.raises(ImageSourceError) as excinfo:
            source.get_frame()
        assert excinfo.value.status is SourceStatus.NOT_OPENED

    def test_open_and_close_are_idempotent(self):
        source = SyntheticImageSource()
        source.open()
        source.open()
        assert source.is_open is True
        source.close()
        source.close()
        assert source.is_open is False

    def test_context_manager_opens_and_closes(self):
        source = SyntheticImageSource()
        with source as ctx:
            assert ctx is source
            assert source.is_open is True
        assert source.is_open is False

    def test_context_manager_closes_on_exception(self):
        source = SyntheticImageSource()
        with pytest.raises(RuntimeError):
            with source:
                raise RuntimeError("boom")
        assert source.is_open is False

    def test_error_message_carries_status_name(self):
        source = SyntheticImageSource()
        with pytest.raises(ImageSourceError, match=r"\[NOT_OPENED\]"):
            source.get_frame()

    def test_repr_reports_lifecycle_state(self):
        source = SyntheticImageSource()
        assert "closed" in repr(source)
        source.open()
        assert "open" in repr(source)

    def test_malformed_frame_from_subclass_is_caught(self):
        """The base class must reject a subclass that breaks the frame contract."""

        class BadSource(SyntheticImageSource):
            def _read_frame(self) -> np.ndarray:
                return np.zeros(self.frame_shape, dtype=np.float32)

        with BadSource() as source:
            with pytest.raises(ImageSourceError) as excinfo:
                source.get_frame()
        assert excinfo.value.status is SourceStatus.MALFORMED_FRAME

    def test_wrong_shape_frame_from_subclass_is_caught(self):
        class WrongShapeSource(SyntheticImageSource):
            def _read_frame(self) -> np.ndarray:
                return np.zeros((7, 7, 3), dtype=np.uint8)

        with WrongShapeSource() as source:
            with pytest.raises(ImageSourceError) as excinfo:
                source.get_frame()
        assert excinfo.value.status is SourceStatus.MALFORMED_FRAME

    def test_non_array_frame_from_subclass_is_caught(self):
        class NotAnArraySource(SyntheticImageSource):
            def _read_frame(self):
                return "not a frame"

        with NotAnArraySource() as source:
            with pytest.raises(ImageSourceError) as excinfo:
                source.get_frame()
        assert excinfo.value.status is SourceStatus.MALFORMED_FRAME


# =========================================================================
# 2. SyntheticImageSource -- shape, dtype, determinism
# =========================================================================


class TestSyntheticImageSource:
    def test_frame_is_uint8_ndarray(self):
        with SyntheticImageSource() as source:
            frame = source.get_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8

    def test_frame_shape_matches_declared_shape(self):
        with SyntheticImageSource() as source:
            assert source.get_frame().shape == source.frame_shape

    def test_frame_is_three_channel_bgr(self):
        with SyntheticImageSource() as source:
            assert source.get_frame().shape[2] == 3

    def test_frame_shape_derives_from_arm_geometry(self):
        """Frame dimensions must come from ArmGeometry, not local constants."""
        px_per_mm = 0.5
        source = SyntheticImageSource(px_per_mm=px_per_mm)
        expected = (
            int(round(DEFAULT_ARM.desk_depth_mm * px_per_mm)),
            int(round(DEFAULT_ARM.desk_width_mm * px_per_mm)),
            3,
        )
        assert source.frame_shape == expected

    def test_frame_aspect_ratio_matches_desk_aspect_ratio(self):
        source = SyntheticImageSource(px_per_mm=1.0)
        height, width = source.frame_shape[0], source.frame_shape[1]
        assert width / height == pytest.approx(
            DEFAULT_ARM.desk_width_mm / DEFAULT_ARM.desk_depth_mm
        )

    def test_custom_geometry_changes_frame_shape(self):
        """A different desk in ArmGeometry must propagate to the frame size."""
        narrow_desk = ArmGeometry(desk_width_mm=800.0, desk_depth_mm=400.0)
        source = SyntheticImageSource(px_per_mm=1.0, arm=narrow_desk)
        assert source.frame_shape == (400, 800, 3)

    def test_px_per_mm_scales_resolution(self):
        low = SyntheticImageSource(px_per_mm=0.5)
        high = SyntheticImageSource(px_per_mm=1.0)
        assert high.frame_shape[0] == 2 * low.frame_shape[0]
        assert high.frame_shape[1] == 2 * low.frame_shape[1]

    def test_frames_are_deterministic_across_calls(self):
        with SyntheticImageSource(objects=[DeskObject(400.0, 300.0)]) as source:
            assert np.array_equal(source.get_frame(), source.get_frame())

    def test_frames_are_deterministic_across_instances(self):
        objects = [DeskObject(400.0, 300.0, label="pen")]
        with SyntheticImageSource(objects=objects) as a, SyntheticImageSource(
            objects=objects
        ) as b:
            assert np.array_equal(a.get_frame(), b.get_frame())

    def test_empty_desk_is_uniform_background(self):
        with SyntheticImageSource(background_gray=128) as source:
            frame = source.get_frame()
        assert np.all(frame == 128)

    def test_noise_makes_successive_frames_differ(self):
        with SyntheticImageSource(noise_sigma=12.0, seed=42) as source:
            assert not np.array_equal(source.get_frame(), source.get_frame())

    def test_seeded_noise_is_reproducible_across_instances(self):
        with SyntheticImageSource(noise_sigma=12.0, seed=7) as a, SyntheticImageSource(
            noise_sigma=12.0, seed=7
        ) as b:
            assert np.array_equal(a.get_frame(), b.get_frame())

    def test_noisy_frame_stays_uint8_and_in_range(self):
        with SyntheticImageSource(noise_sigma=200.0, seed=1) as source:
            frame = source.get_frame()
        assert frame.dtype == np.uint8
        assert frame.min() >= 0 and frame.max() <= 255

    def test_frame_is_a_copy_not_a_shared_buffer(self):
        """Mutating a returned frame must not corrupt later frames."""
        with SyntheticImageSource() as source:
            first = source.get_frame()
            first[:] = 0
            assert np.all(source.get_frame() == 128)


# =========================================================================
# 3. SyntheticImageSource -- projection and object rendering
# =========================================================================


class TestSyntheticProjection:
    def test_desk_origin_maps_to_pixel_origin(self):
        source = SyntheticImageSource(px_per_mm=0.5)
        assert source.desk_mm_to_pixel(0.0, 0.0) == pytest.approx((0.0, 0.0))

    def test_far_desk_corner_maps_to_frame_corner(self):
        source = SyntheticImageSource(px_per_mm=0.5)
        u, v = source.desk_mm_to_pixel(
            DEFAULT_ARM.desk_width_mm, DEFAULT_ARM.desk_depth_mm
        )
        assert (v, u) == pytest.approx(
            (source.frame_shape[0], source.frame_shape[1])
        )

    def test_projection_round_trips(self):
        source = SyntheticImageSource(px_per_mm=0.75)
        for x_mm, y_mm in [(0.0, 0.0), (137.5, 402.25), (1200.0, 600.0)]:
            u, v = source.desk_mm_to_pixel(x_mm, y_mm)
            assert source.pixel_to_desk_mm(u, v) == pytest.approx((x_mm, y_mm))

    def test_object_is_drawn_at_its_desk_position(self):
        obj = DeskObject(
            x_mm=600.0, y_mm=300.0, width_mm=100.0, height_mm=100.0,
            color_bgr=(10, 20, 30), label="block",
        )
        with SyntheticImageSource(objects=[obj], px_per_mm=0.5) as source:
            frame = source.get_frame()
            u, v = source.desk_mm_to_pixel(obj.x_mm, obj.y_mm)
        assert tuple(int(c) for c in frame[int(v), int(u)]) == (10, 20, 30)

    def test_background_away_from_object_is_untouched(self):
        obj = DeskObject(x_mm=100.0, y_mm=100.0, width_mm=40.0, height_mm=40.0)
        with SyntheticImageSource(objects=[obj], background_gray=128) as source:
            frame = source.get_frame()
            u, v = source.desk_mm_to_pixel(1000.0, 500.0)
        assert tuple(int(c) for c in frame[int(v), int(u)]) == (128, 128, 128)

    def test_object_footprint_area_matches_its_mm_size(self):
        obj = DeskObject(
            x_mm=600.0, y_mm=300.0, width_mm=100.0, height_mm=80.0,
            color_bgr=(0, 0, 255), label="pad",
        )
        px_per_mm = 1.0
        with SyntheticImageSource(objects=[obj], px_per_mm=px_per_mm) as source:
            frame = source.get_frame()
        painted = int(np.sum(np.all(frame == np.array([0, 0, 255]), axis=2)))
        expected = obj.width_mm * px_per_mm * obj.height_mm * px_per_mm
        assert painted == pytest.approx(expected, rel=0.05)

    def test_object_entirely_off_desk_is_skipped(self):
        off_desk = DeskObject(x_mm=5000.0, y_mm=5000.0, color_bgr=(0, 0, 255))
        with SyntheticImageSource(objects=[off_desk], background_gray=128) as source:
            assert np.all(source.get_frame() == 128)

    def test_object_partially_off_desk_is_clipped_not_an_error(self):
        straddling = DeskObject(
            x_mm=0.0, y_mm=300.0, width_mm=100.0, height_mm=50.0,
            color_bgr=(0, 0, 255), label="edge",
        )
        with SyntheticImageSource(objects=[straddling]) as source:
            frame = source.get_frame()
        assert np.any(np.all(frame == np.array([0, 0, 255]), axis=2))

    def test_objects_are_exposed_as_ground_truth(self):
        objects = [DeskObject(300.0, 200.0, label="pen")]
        source = SyntheticImageSource(objects=objects)
        assert len(source.objects) == 1
        assert source.objects[0].label == "pen"


# =========================================================================
# 4. SyntheticImageSource -- configuration validation
# =========================================================================


class TestSyntheticValidation:
    @pytest.mark.parametrize("px_per_mm", [0.0, -1.0])
    def test_non_positive_px_per_mm_rejected(self, px_per_mm):
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticImageSource(px_per_mm=px_per_mm)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    @pytest.mark.parametrize("gray", [-1, 256])
    def test_out_of_range_background_rejected(self, gray):
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticImageSource(background_gray=gray)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    def test_negative_noise_sigma_rejected(self):
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticImageSource(noise_sigma=-0.5)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    def test_degenerate_resolution_rejected(self):
        """A px_per_mm so small the desk renders sub-pixel must not silently pass."""
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticImageSource(px_per_mm=1e-6)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    @pytest.mark.parametrize("width_mm,height_mm", [(0.0, 10.0), (10.0, -5.0)])
    def test_non_positive_object_size_rejected(self, width_mm, height_mm):
        with pytest.raises(ImageSourceError) as excinfo:
            DeskObject(100.0, 100.0, width_mm=width_mm, height_mm=height_mm)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    @pytest.mark.parametrize("color", [(0, 0), (0, 0, 300), (-1, 0, 0)])
    def test_invalid_object_color_rejected(self, color):
        with pytest.raises(ImageSourceError) as excinfo:
            DeskObject(100.0, 100.0, color_bgr=color)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    def test_desk_object_is_immutable(self):
        obj = DeskObject(100.0, 100.0)
        with pytest.raises(Exception):
            obj.x_mm = 200.0  # type: ignore[misc]


# =========================================================================
# 5. WebcamImageSource -- exercised through an injected fake capture
# =========================================================================


class TestWebcamImageSource:
    def test_open_probes_frame_shape(self, patch_capture):
        patch_capture(frame_shape=(480, 640, 3))
        with WebcamImageSource(device_index=0) as source:
            assert source.frame_shape == (480, 640, 3)

    def test_get_frame_returns_uint8_ndarray_of_probed_shape(self, patch_capture):
        patch_capture(frame_shape=(720, 1280, 3))
        with WebcamImageSource() as source:
            frame = source.get_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        assert frame.shape == (720, 1280, 3)

    def test_device_index_is_passed_through(self, patch_capture):
        created = patch_capture()
        with WebcamImageSource(device_index=2) as source:
            assert source.device_index == 2
        assert created[0].index == 2

    def test_unopenable_device_raises_device_unavailable(self, patch_capture):
        created = patch_capture(opened=False)
        source = WebcamImageSource()
        with pytest.raises(ImageSourceError) as excinfo:
            source.open()
        assert excinfo.value.status is SourceStatus.DEVICE_UNAVAILABLE
        assert source.is_open is False
        assert created[0].released is True

    def test_read_failure_raises_read_failed(self, patch_capture):
        patch_capture(fail_after=1)  # the probe read succeeds, the next fails
        with WebcamImageSource() as source:
            with pytest.raises(ImageSourceError) as excinfo:
                source.get_frame()
        assert excinfo.value.status is SourceStatus.READ_FAILED

    def test_probe_failure_releases_device_and_stays_closed(self, patch_capture):
        created = patch_capture(fail_after=0)  # even the probe read fails
        source = WebcamImageSource()
        with pytest.raises(ImageSourceError) as excinfo:
            source.open()
        assert excinfo.value.status is SourceStatus.READ_FAILED
        assert source.is_open is False
        assert created[0].released is True

    def test_non_bgr_frame_is_rejected_at_open(self, patch_capture):
        created = patch_capture(frame_shape=(480, 640))  # grayscale, no channel axis
        source = WebcamImageSource()
        with pytest.raises(ImageSourceError) as excinfo:
            source.open()
        assert excinfo.value.status is SourceStatus.MALFORMED_FRAME
        assert created[0].released is True

    def test_frame_shape_before_open_raises_not_opened(self):
        source = WebcamImageSource()
        with pytest.raises(ImageSourceError) as excinfo:
            _ = source.frame_shape
        assert excinfo.value.status is SourceStatus.NOT_OPENED

    def test_get_frame_before_open_raises_not_opened(self):
        source = WebcamImageSource()
        with pytest.raises(ImageSourceError) as excinfo:
            source.get_frame()
        assert excinfo.value.status is SourceStatus.NOT_OPENED

    def test_close_releases_the_capture_device(self, patch_capture):
        created = patch_capture()
        source = WebcamImageSource()
        source.open()
        source.close()
        assert created[0].released is True
        assert source.is_open is False

    def test_frame_shape_after_close_raises_not_opened(self, patch_capture):
        patch_capture()
        source = WebcamImageSource()
        source.open()
        source.close()
        with pytest.raises(ImageSourceError) as excinfo:
            _ = source.frame_shape
        assert excinfo.value.status is SourceStatus.NOT_OPENED

    def test_resolution_hint_is_applied_to_the_device(self, patch_capture):
        created = patch_capture()
        with WebcamImageSource(width=1280, height=720, fps=30.0) as source:
            assert source.requested_resolution == (1280, 720)
        props = created[0].props
        assert props[image_source_module.cv2.CAP_PROP_FRAME_WIDTH] == 1280.0
        assert props[image_source_module.cv2.CAP_PROP_FRAME_HEIGHT] == 720.0
        assert props[image_source_module.cv2.CAP_PROP_FPS] == 30.0

    def test_ignored_resolution_hint_is_visible_not_fatal(self, patch_capture):
        """Backends may substitute a mode; the real shape must be reported."""
        patch_capture(frame_shape=(480, 640, 3))
        with WebcamImageSource(width=1920, height=1080) as source:
            assert source.requested_resolution == (1920, 1080)
            assert source.frame_shape == (480, 640, 3)
            assert source.get_frame().shape == (480, 640, 3)

    def test_warmup_frames_are_discarded_before_probing(self, patch_capture):
        created = patch_capture()
        with WebcamImageSource(warmup_frames=5) as source:
            source.get_frame()
        # 5 warm-up reads + 1 probe read + 1 get_frame read
        assert created[0].read_count == 7

    def test_backend_constant_is_forwarded(self, patch_capture):
        created = patch_capture()
        with WebcamImageSource(backend=1234):
            pass
        assert created[0].backend == 1234

    def test_no_resolution_hint_reports_none(self):
        assert WebcamImageSource().requested_resolution is None

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"width": 640},                    # height missing
            {"height": 480},                   # width missing
            {"width": 0, "height": 480},       # non-positive
            {"width": 640, "height": -1},      # non-positive
            {"fps": 0.0},                      # non-positive
            {"warmup_frames": -1},             # negative
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs):
        with pytest.raises(ImageSourceError) as excinfo:
            WebcamImageSource(**kwargs)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG
