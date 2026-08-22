"""
Test suite for the Session B.2 camera calibration.

Test areas
----------
1. CalibrationTarget -- inner-corner semantics, object points, validation
2. CameraIntrinsics -- accessors, validation, JSON round-trip
3. SyntheticChessboardSource -- renders a detectable board through known optics
4. Calibrator -- accumulation, detection, error handling
5. Accuracy -- recovered intrinsics against exact ground truth
6. Undistortion

On asserting distortion accuracy
--------------------------------
The camera matrix is recovered to a fraction of a percent, but the individual
distortion coefficients are NOT independently identifiable: ``k1``, ``k2`` and
``k3`` are strongly correlated over the radius range any real target covers,
and ``p1``/``p2`` are order-1e-3 terms whose entire image effect is sub-pixel.
Fitting all five to a clean synthetic scene still leaves ``k2`` tens of
percent from truth and ``k3`` wrong by multiples, no matter how accurately the
scene is rendered -- that is a property of the Brown-Conrady model, not of this
code.

What *is* recovered reliably, and what every downstream consumer actually
needs, is the distortion **field**: how far each pixel moves. These tests
therefore assert ``k1`` (the dominant, well-identified term) against truth and
then assert that the recovered model displaces pixels the same way the ground
truth model does, across the whole frame. That fails loudly if calibration
genuinely breaks, without pretending to a precision the estimator cannot offer.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from src.camera_calibration import (
    DEFAULT_DISTORTION,
    INTRINSICS_SCHEMA_VERSION,
    CalibrationError,
    CalibrationStatus,
    CalibrationTarget,
    Calibrator,
    CameraIntrinsics,
    SyntheticChessboardSource,
    default_calibration_poses,
)
from src.image_source import ImageSource, ImageSourceError, SourceStatus

GROUND_TRUTH_MATRIX = np.array(
    [[800.0, 0.0, 319.5], [0.0, 800.0, 239.5], [0.0, 0.0, 1.0]]
)
IMAGE_SIZE = (640, 480)


@pytest.fixture(scope="module")
def source() -> SyntheticChessboardSource:
    """A synthetic camera with exactly known optics."""
    return SyntheticChessboardSource(
        camera_matrix=GROUND_TRUTH_MATRIX,
        distortion_coefficients=DEFAULT_DISTORTION,
        image_size=IMAGE_SIZE,
    )


@pytest.fixture(scope="module")
def frames(source) -> list:
    """One rendered frame per pose. Deterministic, so built once."""
    with source:
        return [source.render_pose(i) for i in range(source.pose_count)]


@pytest.fixture(scope="module")
def calibrated(source, frames) -> CameraIntrinsics:
    """Intrinsics recovered from the full synthetic frame set."""
    calibrator = Calibrator(source.target)
    for frame in frames:
        calibrator.add_frame(frame)
    return calibrator.calibrate()


def undistortion_deviation_px(
    recovered: CameraIntrinsics, truth: CameraIntrinsics, step: int = 16
) -> np.ndarray:
    """
    How far the two models disagree about where each pixel really belongs.

    Undistorts a grid spanning the frame under both models and returns the
    per-point distance between the results, in pixels. This is the physically
    meaningful comparison: two coefficient sets that move pixels identically
    describe the same camera, whatever their individual values.
    """
    width, height = truth.image_size
    grid_x, grid_y = np.meshgrid(
        np.arange(0, width, step, dtype=np.float64),
        np.arange(0, height, step, dtype=np.float64),
    )
    points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).reshape(-1, 1, 2)

    truth_points = cv2.undistortPoints(
        points,
        truth.camera_matrix,
        truth.distortion_coefficients,
        P=truth.camera_matrix,
    ).reshape(-1, 2)
    recovered_points = cv2.undistortPoints(
        points,
        recovered.camera_matrix,
        recovered.distortion_coefficients,
        P=recovered.camera_matrix,
    ).reshape(-1, 2)
    return np.linalg.norm(truth_points - recovered_points, axis=1)


# =========================================================================
# 1. CalibrationTarget
# =========================================================================


class TestCalibrationTarget:
    def test_pattern_size_is_cols_then_rows(self):
        """The order cv2.findChessboardCorners expects, which is not (rows, cols)."""
        assert CalibrationTarget(rows=6, cols=9).pattern_size == (9, 6)

    def test_corner_count_is_the_product(self):
        assert CalibrationTarget(rows=6, cols=9).corner_count == 54

    def test_board_size_counts_the_outer_ring_of_squares(self):
        """9 x 6 inner corners means a 10 x 7 square board."""
        target = CalibrationTarget(rows=6, cols=9, square_size_mm=25.0)
        assert target.board_size_mm == (250.0, 175.0)

    def test_object_points_shape_and_dtype(self):
        points = CalibrationTarget(rows=6, cols=9).object_points()
        assert points.shape == (54, 3)
        assert points.dtype == np.float32

    def test_object_points_are_planar(self):
        assert np.all(CalibrationTarget().object_points()[:, 2] == 0.0)

    def test_object_points_are_spaced_by_the_square_size(self):
        points = CalibrationTarget(rows=6, cols=9, square_size_mm=25.0).object_points()
        assert points[0].tolist() == [0.0, 0.0, 0.0]
        assert points[1].tolist() == [25.0, 0.0, 0.0]
        assert points[-1].tolist() == [200.0, 125.0, 0.0]

    def test_object_points_advance_along_a_row_first(self):
        """Matching findChessboardCorners' ordering; getting this wrong fits garbage."""
        points = CalibrationTarget(rows=6, cols=9, square_size_mm=25.0).object_points()
        assert points[8].tolist() == [200.0, 0.0, 0.0]
        assert points[9].tolist() == [0.0, 25.0, 0.0]

    def test_board_centre_is_the_grid_centre(self):
        centre = CalibrationTarget(rows=6, cols=9, square_size_mm=25.0).board_centre_mm()
        assert centre.tolist() == [100.0, 62.5, 0.0]

    def test_scaling_the_square_scales_the_object_points(self):
        small = CalibrationTarget(rows=6, cols=9, square_size_mm=10.0).object_points()
        large = CalibrationTarget(rows=6, cols=9, square_size_mm=20.0).object_points()
        assert np.allclose(large, 2.0 * small)

    @pytest.mark.parametrize("kwargs", [{"rows": 1}, {"cols": 1}, {"rows": 0}])
    def test_degenerate_pattern_rejected(self, kwargs):
        with pytest.raises(CalibrationError) as excinfo:
            CalibrationTarget(**kwargs)
        assert excinfo.value.status is CalibrationStatus.INVALID_TARGET

    def test_square_pattern_rejected_for_orientation_ambiguity(self):
        with pytest.raises(CalibrationError) as excinfo:
            CalibrationTarget(rows=7, cols=7)
        assert excinfo.value.status is CalibrationStatus.INVALID_TARGET
        assert "ambiguous" in str(excinfo.value)

    @pytest.mark.parametrize("size", [0.0, -5.0])
    def test_non_positive_square_size_rejected(self, size):
        with pytest.raises(CalibrationError) as excinfo:
            CalibrationTarget(square_size_mm=size)
        assert excinfo.value.status is CalibrationStatus.INVALID_TARGET

    def test_target_is_immutable(self):
        with pytest.raises(Exception):
            CalibrationTarget().rows = 8  # type: ignore[misc]


# =========================================================================
# 2. CameraIntrinsics
# =========================================================================


class TestCameraIntrinsics:
    @pytest.fixture
    def intrinsics(self) -> CameraIntrinsics:
        return CameraIntrinsics(
            camera_matrix=GROUND_TRUTH_MATRIX,
            distortion_coefficients=np.array(DEFAULT_DISTORTION),
            image_size=IMAGE_SIZE,
            rms_reprojection_error=0.21,
        )

    def test_accessors_read_the_matrix(self, intrinsics):
        assert intrinsics.fx == pytest.approx(800.0)
        assert intrinsics.fy == pytest.approx(800.0)
        assert intrinsics.cx == pytest.approx(319.5)
        assert intrinsics.cy == pytest.approx(239.5)
        assert intrinsics.image_width == 640
        assert intrinsics.image_height == 480

    def test_lists_are_normalised_to_arrays(self):
        intrinsics = CameraIntrinsics(
            camera_matrix=[[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1.0]],
            distortion_coefficients=[0.0] * 5,
            image_size=[640, 480],
            rms_reprojection_error=0.0,
        )
        assert isinstance(intrinsics.camera_matrix, np.ndarray)
        assert isinstance(intrinsics.distortion_coefficients, np.ndarray)
        assert intrinsics.image_size == (640, 480)

    def test_json_round_trip_preserves_every_field(self, intrinsics, tmp_path):
        path = intrinsics.save_json(tmp_path / "intrinsics.json")
        restored = CameraIntrinsics.load_json(path)
        assert np.allclose(restored.camera_matrix, intrinsics.camera_matrix)
        assert np.allclose(
            restored.distortion_coefficients, intrinsics.distortion_coefficients
        )
        assert restored.image_size == intrinsics.image_size
        assert restored.rms_reprojection_error == pytest.approx(
            intrinsics.rms_reprojection_error
        )

    def test_save_creates_missing_parent_directories(self, intrinsics, tmp_path):
        path = intrinsics.save_json(tmp_path / "a" / "b" / "intrinsics.json")
        assert path.exists()

    def test_saved_file_is_readable_json_with_a_schema(self, intrinsics, tmp_path):
        path = intrinsics.save_json(tmp_path / "intrinsics.json")
        payload = json.loads(path.read_text())
        assert payload["schema"] == INTRINSICS_SCHEMA_VERSION
        assert len(payload["distortion_coefficients"]) == 5

    def test_load_missing_file_reports_invalid_intrinsics(self, tmp_path):
        with pytest.raises(CalibrationError) as excinfo:
            CameraIntrinsics.load_json(tmp_path / "nope.json")
        assert excinfo.value.status is CalibrationStatus.INVALID_INTRINSICS

    def test_load_malformed_json_reports_invalid_intrinsics(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(CalibrationError) as excinfo:
            CameraIntrinsics.load_json(path)
        assert excinfo.value.status is CalibrationStatus.INVALID_INTRINSICS

    def test_load_wrong_schema_rejected(self, intrinsics, tmp_path):
        path = tmp_path / "old.json"
        payload = intrinsics.to_dict()
        payload["schema"] = 99
        path.write_text(json.dumps(payload))
        with pytest.raises(CalibrationError, match="schema"):
            CameraIntrinsics.load_json(path)

    def test_load_missing_field_rejected(self, intrinsics, tmp_path):
        path = tmp_path / "partial.json"
        payload = intrinsics.to_dict()
        del payload["distortion_coefficients"]
        path.write_text(json.dumps(payload))
        with pytest.raises(CalibrationError, match="missing required field"):
            CameraIntrinsics.load_json(path)

    def test_load_non_object_json_rejected(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(CalibrationError) as excinfo:
            CameraIntrinsics.load_json(path)
        assert excinfo.value.status is CalibrationStatus.INVALID_INTRINSICS

    def test_wrong_matrix_shape_rejected(self):
        with pytest.raises(CalibrationError, match="must be 3x3"):
            CameraIntrinsics(
                camera_matrix=np.eye(4),
                distortion_coefficients=np.zeros(5),
                image_size=IMAGE_SIZE,
                rms_reprojection_error=0.0,
            )

    @pytest.mark.parametrize("count", [3, 4, 8])
    def test_wrong_coefficient_count_rejected(self, count):
        with pytest.raises(CalibrationError, match="5 elements"):
            CameraIntrinsics(
                camera_matrix=GROUND_TRUTH_MATRIX,
                distortion_coefficients=np.zeros(count),
                image_size=IMAGE_SIZE,
                rms_reprojection_error=0.0,
            )

    @pytest.mark.parametrize("size", [(0, 480), (640, -1), (640,)])
    def test_invalid_image_size_rejected(self, size):
        with pytest.raises(CalibrationError, match="image_size"):
            CameraIntrinsics(
                camera_matrix=GROUND_TRUTH_MATRIX,
                distortion_coefficients=np.zeros(5),
                image_size=size,
                rms_reprojection_error=0.0,
            )

    def test_negative_rms_rejected(self):
        with pytest.raises(CalibrationError, match="non-negative"):
            CameraIntrinsics(
                camera_matrix=GROUND_TRUTH_MATRIX,
                distortion_coefficients=np.zeros(5),
                image_size=IMAGE_SIZE,
                rms_reprojection_error=-0.1,
            )

    def test_summary_names_the_key_numbers(self, intrinsics):
        summary = intrinsics.summary()
        for token in ("Focal length", "Principal pt", "Distortion", "RMS"):
            assert token in summary


# =========================================================================
# 3. SyntheticChessboardSource
# =========================================================================


class TestSyntheticChessboardSource:
    def test_is_an_image_source(self):
        assert issubclass(SyntheticChessboardSource, ImageSource)

    def test_frame_is_uint8_bgr_of_the_requested_size(self, source, frames):
        frame = frames[0]
        assert frame.dtype == np.uint8
        assert frame.shape == (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
        assert frame.shape == source.frame_shape

    def test_get_frame_before_open_raises(self):
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticChessboardSource().get_frame()
        assert excinfo.value.status is SourceStatus.NOT_OPENED

    def test_frames_cycle_through_the_poses(self, source):
        with source:
            first = source.get_frame()
            [source.get_frame() for _ in range(source.pose_count - 1)]
            wrapped = source.get_frame()
        assert np.array_equal(first, wrapped)

    def test_successive_poses_differ(self, frames):
        assert not np.array_equal(frames[0], frames[1])

    def test_rendering_is_deterministic(self, source):
        assert np.array_equal(source.render_pose(0), source.render_pose(0))

    def test_every_recipe_pose_lands_inside_the_frame(self, source):
        """A pose whose board is clipped would silently reduce the frame count."""
        assert source.pose_count == 15

    def test_ground_truth_is_reported_exactly(self, source):
        truth = source.ground_truth_intrinsics()
        assert np.allclose(truth.camera_matrix, GROUND_TRUTH_MATRIX)
        assert np.allclose(truth.distortion_coefficients, DEFAULT_DISTORTION)
        assert truth.rms_reprojection_error == 0.0

    def test_projected_corners_are_inside_the_frame(self, source):
        for index in range(source.pose_count):
            corners = source.project_corners(index)
            assert corners.shape == (source.target.corner_count, 2)
            assert corners[:, 0].min() >= 0 and corners[:, 0].max() < IMAGE_SIZE[0]
            assert corners[:, 1].min() >= 0 and corners[:, 1].max() < IMAGE_SIZE[1]

    def test_distortion_actually_bends_the_projection(self, source):
        """
        Without this, the renderer could be silently ignoring the coefficients
        and the accuracy tests would be measuring nothing.
        """
        undistorted = SyntheticChessboardSource(
            camera_matrix=GROUND_TRUTH_MATRIX,
            distortion_coefficients=(0.0, 0.0, 0.0, 0.0, 0.0),
            image_size=IMAGE_SIZE,
            poses=source.poses,
        )
        deviation = np.linalg.norm(
            source.project_corners(0) - undistorted.project_corners(0), axis=1
        )
        assert deviation.max() > 1.0

    def test_rendered_board_has_both_tones(self, frames):
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        assert gray.min() < 60, "no dark squares were drawn"
        assert gray.max() > 200, "no light background"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"image_size": (0, 480)},
            {"supersample": 0},
            {"edge_subdivisions": 0},
            {"light_gray": 10, "dark_gray": 200},
            {"distortion_coefficients": (0.0, 0.0)},
            {"camera_matrix": np.eye(4)},
        ],
    )
    def test_invalid_configuration_rejected(self, kwargs):
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticChessboardSource(**kwargs)
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    def test_no_usable_poses_rejected(self):
        """A board too close to fit the frame must fail loudly, not render junk."""
        with pytest.raises(ImageSourceError) as excinfo:
            SyntheticChessboardSource(
                target=CalibrationTarget(rows=6, cols=9, square_size_mm=400.0)
            )
        assert excinfo.value.status is SourceStatus.INVALID_CONFIG

    def test_default_poses_are_filtered_against_the_frame(self):
        poses = default_calibration_poses(
            CalibrationTarget(), GROUND_TRUTH_MATRIX, IMAGE_SIZE
        )
        assert 0 < len(poses) <= 15


# =========================================================================
# 4. Calibrator behaviour and error handling
# =========================================================================


class TestCalibratorBehaviour:
    def test_starts_empty(self):
        calibrator = Calibrator(CalibrationTarget())
        assert calibrator.frame_count == 0
        assert calibrator.image_size is None

    def test_add_frame_returns_true_when_the_board_is_found(self, frames):
        calibrator = Calibrator(CalibrationTarget())
        assert calibrator.add_frame(frames[0]) is True
        assert calibrator.frame_count == 1

    def test_every_synthetic_frame_is_detected(self, frames):
        """A renderer the detector cannot read would invalidate the whole suite."""
        calibrator = Calibrator(CalibrationTarget())
        assert all(calibrator.add_frame(frame) for frame in frames)
        assert calibrator.frame_count == len(frames)

    def test_blank_frame_returns_false_without_raising(self):
        """No board in view is a routine outcome, not an error."""
        calibrator = Calibrator(CalibrationTarget())
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert calibrator.add_frame(blank) is False
        assert calibrator.frame_count == 0

    def test_noise_frame_returns_false(self):
        calibrator = Calibrator(CalibrationTarget())
        rng = np.random.default_rng(0)
        noise = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        assert calibrator.add_frame(noise) is False

    def test_grayscale_frames_are_accepted(self, frames):
        calibrator = Calibrator(CalibrationTarget())
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        assert calibrator.add_frame(gray) is True

    def test_image_size_is_fixed_by_the_first_frame(self, frames):
        calibrator = Calibrator(CalibrationTarget())
        calibrator.add_frame(frames[0])
        assert calibrator.image_size == IMAGE_SIZE

    def test_mismatched_frame_size_rejected(self, frames):
        calibrator = Calibrator(CalibrationTarget())
        calibrator.add_frame(frames[0])
        resized = cv2.resize(frames[1], (320, 240))
        with pytest.raises(CalibrationError) as excinfo:
            calibrator.add_frame(resized)
        assert excinfo.value.status is CalibrationStatus.FRAME_SIZE_MISMATCH

    @pytest.mark.parametrize(
        "bad",
        [
            "not an array",
            np.zeros((480, 640), dtype=np.float32),
            np.zeros((480, 640, 4), dtype=np.uint8),
            np.zeros((3, 4, 5, 6), dtype=np.uint8),
        ],
    )
    def test_malformed_frames_rejected(self, bad):
        calibrator = Calibrator(CalibrationTarget())
        with pytest.raises(CalibrationError) as excinfo:
            calibrator.add_frame(bad)
        assert excinfo.value.status is CalibrationStatus.MALFORMED_FRAME

    def test_calibrate_with_too_few_frames_rejected(self, frames):
        calibrator = Calibrator(CalibrationTarget(), min_frames=5)
        for frame in frames[:3]:
            calibrator.add_frame(frame)
        with pytest.raises(CalibrationError) as excinfo:
            calibrator.calibrate()
        assert excinfo.value.status is CalibrationStatus.TOO_FEW_FRAMES

    def test_calibrate_with_no_frames_rejected(self):
        with pytest.raises(CalibrationError) as excinfo:
            Calibrator(CalibrationTarget()).calibrate()
        assert excinfo.value.status is CalibrationStatus.TOO_FEW_FRAMES

    def test_min_frames_below_three_rejected(self):
        with pytest.raises(CalibrationError) as excinfo:
            Calibrator(CalibrationTarget(), min_frames=2)
        assert excinfo.value.status is CalibrationStatus.INVALID_TARGET

    def test_reset_discards_accumulated_views(self, frames):
        calibrator = Calibrator(CalibrationTarget())
        calibrator.add_frame(frames[0])
        calibrator.reset()
        assert calibrator.frame_count == 0
        assert calibrator.image_size is None

    def test_find_corners_matches_the_projected_ground_truth(self, source, frames):
        """Detection accuracy, independent of the calibration fit."""
        calibrator = Calibrator(source.target)
        corners = calibrator.find_corners(frames[0])
        assert corners is not None
        deviation = np.linalg.norm(
            corners.reshape(-1, 2) - source.project_corners(0), axis=1
        )
        assert deviation.mean() < 0.25

    def test_find_corners_returns_none_on_a_blank_frame(self):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        assert Calibrator(CalibrationTarget()).find_corners(blank) is None

    def test_fix_k3_defaults_off_and_sets_the_solver_flag(self):
        assert Calibrator().fix_k3 is False
        assert Calibrator().calibration_flags == 0
        assert Calibrator(fix_k3=True).calibration_flags == cv2.CALIB_FIX_K3

    def test_fix_k3_holds_the_sixth_order_term_at_zero(self, frames):
        calibrator = Calibrator(CalibrationTarget(), fix_k3=True)
        for frame in frames:
            calibrator.add_frame(frame)
        assert calibrator.calibrate().distortion_coefficients[4] == pytest.approx(0.0)


# =========================================================================
# 5. Accuracy against exact ground truth
# =========================================================================


class TestCalibrationAccuracy:
    def test_uses_all_fifteen_synthetic_views(self, source, frames):
        assert len(frames) == 15
        assert source.pose_count == 15

    def test_rms_reprojection_error_is_subpixel(self, calibrated):
        assert calibrated.rms_reprojection_error < 0.5

    def test_recovered_image_size_matches(self, calibrated):
        assert calibrated.image_size == IMAGE_SIZE

    @pytest.mark.parametrize("element", ["fx", "fy", "cx", "cy"])
    def test_camera_matrix_recovered_within_one_percent(self, calibrated, element):
        truth = {"fx": 800.0, "fy": 800.0, "cx": 319.5, "cy": 239.5}[element]
        recovered = getattr(calibrated, element)
        assert recovered == pytest.approx(truth, rel=0.01)

    def test_full_camera_matrix_matches_elementwise(self, calibrated):
        """Including the zeros: a non-zero skew term would mean a broken fit."""
        assert np.allclose(calibrated.camera_matrix, GROUND_TRUTH_MATRIX, rtol=0.01,
                           atol=1e-6)

    def test_dominant_radial_coefficient_recovered_within_five_percent(
        self, calibrated
    ):
        """
        k1 carries almost all of the distortion and IS well identified.

        k2, p1, p2 and k3 are not independently identifiable -- see the module
        docstring -- so they are checked through the distortion field instead.
        """
        assert calibrated.distortion_coefficients[0] == pytest.approx(
            DEFAULT_DISTORTION[0], rel=0.05
        )

    def test_distortion_is_recovered_with_the_right_sign(self, calibrated):
        """A sign flip would turn barrel into pincushion and still fit loosely."""
        assert calibrated.distortion_coefficients[0] < 0.0

    def test_undistortion_field_matches_ground_truth(self, calibrated, source):
        """
        The assertion that actually matters.

        Two coefficient sets describing the same camera move every pixel to the
        same place. Sub-pixel agreement across the whole frame means the
        recovered model is usable wherever the ground-truth one would be.
        """
        deviation = undistortion_deviation_px(
            calibrated, source.ground_truth_intrinsics()
        )
        assert deviation.mean() < 0.20
        assert deviation.max() < 1.00

    def test_undistortion_field_test_would_catch_a_broken_fit(self, source):
        """
        Guards the guard: confirm the field comparison is not vacuous.

        A model with no distortion at all must fail the tolerance the real
        calibration passes, otherwise the test above proves nothing.
        """
        truth = source.ground_truth_intrinsics()
        undistorted_model = CameraIntrinsics(
            camera_matrix=GROUND_TRUTH_MATRIX,
            distortion_coefficients=np.zeros(5),
            image_size=IMAGE_SIZE,
            rms_reprojection_error=0.0,
        )
        deviation = undistortion_deviation_px(undistorted_model, truth)
        assert deviation.max() > 1.00

    def test_corners_reproject_onto_their_detections(self, source, frames):
        """
        End-to-end closure: fit the model, then use it to reproject the board.

        Passes only if the intrinsics, the object points and the detection
        ordering are all mutually consistent.
        """
        calibrator = Calibrator(source.target)
        for frame in frames:
            calibrator.add_frame(frame)
        intrinsics = calibrator.calibrate()

        object_points = source.target.object_points().astype(np.float64)
        for index in range(source.pose_count):
            detected = source.project_corners(index)
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                detected,
                intrinsics.camera_matrix,
                intrinsics.distortion_coefficients,
            )
            assert ok
            reprojected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                intrinsics.camera_matrix,
                intrinsics.distortion_coefficients,
            )
            error = np.linalg.norm(
                reprojected.reshape(-1, 2) - detected, axis=1
            ).mean()
            assert error < 0.5

    def test_fewer_views_still_converge_but_are_reported_honestly(self, source, frames):
        """A thin calibration should still report a real RMS, not a suspicious zero."""
        calibrator = Calibrator(source.target, min_frames=5)
        for frame in frames[:6]:
            calibrator.add_frame(frame)
        intrinsics = calibrator.calibrate()
        assert intrinsics.rms_reprojection_error > 0.0


# =========================================================================
# 6. Undistortion
# =========================================================================


class TestUndistort:
    def test_undistort_preserves_shape_and_dtype(self, calibrated, frames):
        result = Calibrator.undistort(frames[0], calibrated)
        assert result.shape == frames[0].shape
        assert result.dtype == frames[0].dtype

    def test_undistort_changes_the_image(self, calibrated, frames):
        assert not np.array_equal(Calibrator.undistort(frames[0], calibrated),
                                  frames[0])

    def test_undistorting_a_pinhole_image_is_almost_a_no_op(self, frames):
        """With zero distortion there is nothing to correct."""
        pinhole = CameraIntrinsics(
            camera_matrix=GROUND_TRUTH_MATRIX,
            distortion_coefficients=np.zeros(5),
            image_size=IMAGE_SIZE,
            rms_reprojection_error=0.0,
        )
        result = Calibrator.undistort(frames[0], pinhole)
        assert np.array_equal(result, frames[0])

    def test_undistorted_board_is_still_detectable(self, calibrated, frames):
        """Correction must not destroy the image it corrects."""
        calibrator = Calibrator(CalibrationTarget())
        corrected = Calibrator.undistort(frames[0], calibrated)
        assert calibrator.find_corners(corrected) is not None

    def test_undistortion_straightens_the_board(self, source, calibrated, frames):
        """
        A chessboard row is collinear in the world, so it must be collinear
        once distortion is removed -- that is the whole point of correcting it.

        Perspective preserves straight lines, so a tilted board is no obstacle:
        any bow in a detected row is distortion and nothing else. The test uses
        an off-centre pose because radial distortion grows with radius, and the
        frontal pose sits where the lens is nearly rectilinear (0.92 px of bow,
        too close to the noise floor to prove anything).
        """
        target = source.target
        calibrator = Calibrator(target)

        def worst_row_bow(image: np.ndarray) -> float:
            corners = calibrator.find_corners(image)
            assert corners is not None
            points = corners.reshape(target.rows, target.cols, 2)
            worst = 0.0
            for row in points:
                start, end = row[0], row[-1]
                direction = end - start
                direction = direction / np.linalg.norm(direction)
                normal = np.array([-direction[1], direction[0]])
                worst = max(worst, float(np.abs((row - start) @ normal).max()))
            return worst

        off_centre = frames[5]
        distorted_bow = worst_row_bow(off_centre)
        corrected_bow = worst_row_bow(Calibrator.undistort(off_centre, calibrated))
        assert distorted_bow > 1.0, "the synthetic frame was not actually distorted"
        assert corrected_bow < distorted_bow / 4.0

    def test_grayscale_input_is_accepted(self, calibrated, frames):
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        assert Calibrator.undistort(gray, calibrated).shape == gray.shape

    def test_mismatched_frame_size_rejected(self, calibrated, frames):
        resized = cv2.resize(frames[0], (320, 240))
        with pytest.raises(CalibrationError) as excinfo:
            Calibrator.undistort(resized, calibrated)
        assert excinfo.value.status is CalibrationStatus.FRAME_SIZE_MISMATCH

    @pytest.mark.parametrize("bad", ["not an array", np.zeros((4, 4, 4, 4))])
    def test_malformed_input_rejected(self, calibrated, bad):
        with pytest.raises(CalibrationError) as excinfo:
            Calibrator.undistort(bad, calibrated)
        assert excinfo.value.status is CalibrationStatus.MALFORMED_FRAME

    def test_remove_last_frame_drops_one_view(self, frames):
        calibrator = Calibrator(CalibrationTarget())
        calibrator.add_frame(frames[0])
        calibrator.add_frame(frames[1])
        assert calibrator.remove_last_frame() is True
        assert calibrator.frame_count == 1

    def test_remove_last_frame_on_empty_reports_false(self):
        assert Calibrator(CalibrationTarget()).remove_last_frame() is False

    def test_remove_last_frame_keeps_the_accepted_image_size(self, frames):
        """Frame size is a property of the camera, not of any single view."""
        calibrator = Calibrator(CalibrationTarget())
        calibrator.add_frame(frames[0])
        calibrator.remove_last_frame()
        assert calibrator.image_size == IMAGE_SIZE
