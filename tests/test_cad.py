"""
Test suite for the Session D.1 parametric CAD.

Test areas
----------
1. Hardware specifications in src.geometry (validation, derived values,
   the UNVERIFIED-field bookkeeping)
2. Pedestal parameter derivation -- every dimension traceable to geometry.py
3. Design rule checks -- each violation raises with the right DesignStatus
4. Solid construction -- the part builds and has positive volume
5. STL export -- valid binary STL, closed watertight mesh, nonzero volume

The mesh checks parse the exported STL directly rather than trusting the
kernel's own report, because "the STL is watertight" is a property of the
tessellated output a slicer will actually read, not of the B-rep it came from.
"""

from __future__ import annotations

import collections
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.geometry import (
    DEFAULT_ARM,
    DEFAULT_HARDWARE,
    ArmGeometry,
    BaseStack,
    BearingSpec,
    FastenerSpec,
    HardwareSpec,
    ServoSpec,
)

build123d = pytest.importorskip(
    "build123d", reason="build123d is required for the CAD suite (Python >= 3.10)"
)

from cad.base_pedestal import (  # noqa: E402  - must follow the importorskip
    DesignStatus,
    PedestalDesignError,
    PedestalParameters,
    build_pedestal,
    export_pedestal,
)


# =========================================================================
# Mesh helpers
# =========================================================================


class Mesh:
    """A triangle soup welded into an indexed mesh, for topology assertions."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces

    @classmethod
    def from_binary_stl(cls, path: Path, decimals: int = 5) -> "Mesh":
        """
        Parse a binary STL and weld coincident vertices.

        STL stores every triangle's vertices independently, so shared corners
        arrive as distinct float triples. Rounding to ``decimals`` before
        deduplicating is what turns the soup into a mesh whose edges can be
        counted.
        """
        raw = path.read_bytes()
        if len(raw) < 84:
            raise ValueError(f"{path} is too short to be a binary STL.")

        triangle_count = struct.unpack("<I", raw[80:84])[0]
        expected_size = 84 + 50 * triangle_count
        if len(raw) != expected_size:
            raise ValueError(
                f"{path}: header declares {triangle_count} triangles "
                f"({expected_size} bytes) but the file is {len(raw)} bytes."
            )

        record = np.dtype([("normal", "<3f4"), ("verts", "<3,3f4"), ("attr", "<u2")])
        records = np.frombuffer(raw[84:expected_size], dtype=record)
        soup = records["verts"].reshape(-1, 3).astype(np.float64)

        unique, inverse = np.unique(np.round(soup, decimals), axis=0, return_inverse=True)
        return cls(unique, inverse.reshape(-1, 3))

    def edge_use_counts(self) -> collections.Counter:
        """How many faces use each *undirected* edge. Closed surface => all 2."""
        counts: collections.Counter = collections.Counter()
        for a, b, c in self.faces:
            for edge in ((a, b), (b, c), (c, a)):
                counts[tuple(sorted(edge))] += 1
        return counts

    def directed_edge_counts(self) -> collections.Counter:
        """How many faces traverse each *directed* edge. Oriented => all 1."""
        counts: collections.Counter = collections.Counter()
        for a, b, c in self.faces:
            for edge in ((a, b), (b, c), (c, a)):
                counts[edge] += 1
        return counts

    def signed_volume_mm3(self) -> float:
        """
        Enclosed volume via the divergence theorem.

        Sum of signed tetrahedron volumes from the origin to each triangle.
        Positive for a closed surface with outward-facing normals; a
        near-zero result means the surface does not enclose anything.
        """
        corners = self.vertices[self.faces]
        return float(
            np.einsum(
                "ij,ij->i", corners[:, 0], np.cross(corners[:, 1], corners[:, 2])
            ).sum()
            / 6.0
        )

    def bounding_box(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


@pytest.fixture(scope="module")
def params() -> PedestalParameters:
    """Default parameters, derived from the geometry singletons."""
    return PedestalParameters.from_geometry()


@pytest.fixture(scope="module")
def pedestal(params):
    """The built solid. Module-scoped: the kernel work is not free."""
    return build_pedestal(params)


@pytest.fixture(scope="module")
def stl_path(tmp_path_factory, params) -> Path:
    """An exported STL in a temp directory, built once for the whole module."""
    destination = tmp_path_factory.mktemp("cad") / "base_pedestal.stl"
    return export_pedestal(destination, params)


@pytest.fixture(scope="module")
def mesh(stl_path) -> Mesh:
    return Mesh.from_binary_stl(stl_path)


# =========================================================================
# 1. Hardware specifications in src.geometry
# =========================================================================


class TestHardwareSpecs:
    def test_ds3218_body_matches_datasheet(self):
        """40 x 20 x 40.5 mm, 60 g -- DSServo datasheet, section 2 Mechanical."""
        servo = DEFAULT_HARDWARE.base_yaw_servo
        assert servo.name == "DS3218"
        assert (servo.body_length_mm, servo.body_width_mm, servo.body_height_mm) == (
            40.0, 20.0, 40.5,
        )
        assert servo.mass_g == pytest.approx(60.0)

    def test_unverified_fields_are_declared_and_real(self):
        """Every name in UNVERIFIED_FIELDS must actually exist on the spec."""
        servo = ServoSpec()
        assert servo.UNVERIFIED_FIELDS, "the verification warning must not be empty"
        for field_name in servo.UNVERIFIED_FIELDS:
            assert hasattr(servo, field_name), field_name

    def test_datasheet_confirmed_fields_are_not_flagged(self):
        """The body envelope is confirmed; it must not appear in the warning."""
        servo = ServoSpec()
        for field_name in ("body_length_mm", "body_width_mm", "body_height_mm"):
            assert field_name not in servo.UNVERIFIED_FIELDS

    def test_unverified_report_lists_every_flagged_field(self):
        report = ServoSpec().unverified_report()
        for field_name in ServoSpec.UNVERIFIED_FIELDS:
            assert field_name in report

    def test_body_offset_places_shaft_on_the_axis(self):
        servo = ServoSpec(body_length_mm=40.0, shaft_offset_from_body_end_mm=10.0)
        assert servo.body_offset_from_shaft_axis_mm == pytest.approx(10.0)

    def test_centred_shaft_gives_zero_offset(self):
        servo = ServoSpec(body_length_mm=40.0, shaft_offset_from_body_end_mm=20.0)
        assert servo.body_offset_from_shaft_axis_mm == pytest.approx(0.0)

    def test_travel_converts_to_radians(self):
        assert ServoSpec(travel_deg=270.0).travel_rad == pytest.approx(
            3.0 * np.pi / 2.0
        )

    def test_shaft_offset_beyond_body_rejected(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            ServoSpec(body_length_mm=40.0, shaft_offset_from_body_end_mm=45.0)

    def test_flange_narrower_than_body_rejected(self):
        with pytest.raises(ValueError, match="mounting ears extend beyond"):
            ServoSpec(body_length_mm=40.0, flange_span_mm=30.0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"body_length_mm": 0.0}, {"body_width_mm": -1.0}, {"body_height_mm": 0.0}],
    )
    def test_non_positive_servo_dimensions_rejected(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            ServoSpec(**kwargs)

    def test_608zz_is_the_iso_standard_triple(self):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert (
            bearing.bore_diameter_mm,
            bearing.outer_diameter_mm,
            bearing.width_mm,
        ) == (8.0, 22.0, 7.0)

    def test_bearing_seat_is_undersized_for_a_press_fit(self):
        bearing = BearingSpec(outer_diameter_mm=22.0, press_fit_interference_mm=0.1)
        assert bearing.seat_diameter_mm == pytest.approx(21.9)
        assert bearing.seat_diameter_mm < bearing.outer_diameter_mm

    def test_bore_not_smaller_than_od_rejected(self):
        with pytest.raises(ValueError, match="0 < bore"):
            BearingSpec(bore_diameter_mm=25.0, outer_diameter_mm=22.0)

    def test_non_positive_bearing_width_rejected(self):
        with pytest.raises(ValueError, match="width_mm must be positive"):
            BearingSpec(width_mm=0.0)

    def test_m4_clearance_hole_exceeds_nominal(self):
        fastener = DEFAULT_HARDWARE.mounting_fastener
        assert fastener.clearance_hole_diameter_mm > fastener.nominal_diameter_mm

    def test_clearance_hole_below_nominal_rejected(self):
        with pytest.raises(ValueError, match="at least the"):
            FastenerSpec(nominal_diameter_mm=4.0, clearance_hole_diameter_mm=3.5)

    def test_head_smaller_than_hole_rejected(self):
        with pytest.raises(ValueError, match="pulls through"):
            FastenerSpec(clearance_hole_diameter_mm=4.5, head_diameter_mm=4.0)

    def test_base_stack_allowance_is_the_sum_of_its_parts(self):
        stack = BaseStack(
            turntable_plate_thickness_mm=6.0, shoulder_bracket_rise_mm=24.0
        )
        assert stack.allowance_mm == pytest.approx(30.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"turntable_plate_thickness_mm": 0.0},
            {"shoulder_bracket_rise_mm": -1.0},
        ],
    )
    def test_non_positive_stack_component_rejected(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            BaseStack(**kwargs)

    def test_pedestal_height_is_base_height_minus_allowance(self):
        expected = DEFAULT_ARM.base_height_mm - DEFAULT_HARDWARE.base_stack.allowance_mm
        assert DEFAULT_HARDWARE.pedestal_height_mm() == pytest.approx(expected)
        assert DEFAULT_HARDWARE.pedestal_height_mm() == pytest.approx(70.0)

    def test_pedestal_height_tracks_arm_geometry(self):
        """Raising base_height_mm must lengthen the pedestal, not something else."""
        taller = ArmGeometry(base_height_mm=150.0)
        assert DEFAULT_HARDWARE.pedestal_height_mm(taller) == pytest.approx(
            150.0 - DEFAULT_HARDWARE.base_stack.allowance_mm
        )

    def test_stack_allowance_exceeding_base_height_rejected(self):
        greedy = HardwareSpec(
            base_stack=BaseStack(
                turntable_plate_thickness_mm=60.0, shoulder_bracket_rise_mm=60.0
            )
        )
        with pytest.raises(ValueError, match="non-positive height"):
            greedy.pedestal_height_mm()

    def test_too_few_mounting_bolts_rejected(self):
        with pytest.raises(ValueError, match="at least 3 mounting bolts"):
            HardwareSpec(mount_bolt_count=2)

    def test_hardware_report_surfaces_the_verification_warning(self):
        report = DEFAULT_HARDWARE.hardware_report()
        assert "UNVERIFIED" in report
        assert "608ZZ" in report and "DS3218" in report


# =========================================================================
# 2. Pedestal parameter derivation
# =========================================================================


class TestParameterDerivation:
    def test_default_parameters_validate(self, params):
        assert params.validate() is DesignStatus.OK

    def test_height_comes_from_the_geometry_singletons(self, params):
        assert params.total_height_mm == pytest.approx(
            DEFAULT_HARDWARE.pedestal_height_mm(DEFAULT_ARM)
        )

    def test_taller_base_height_gives_a_taller_pedestal(self):
        taller = PedestalParameters.from_geometry(arm=ArmGeometry(base_height_mm=150.0))
        assert taller.total_height_mm == pytest.approx(120.0)

    def test_cavity_is_the_servo_body_plus_clearance(self, params):
        servo = DEFAULT_HARDWARE.base_yaw_servo
        clearance = DEFAULT_HARDWARE.print_clearance_mm
        assert params.cavity_body_length_mm == pytest.approx(
            servo.body_length_mm + 2 * clearance
        )
        assert params.cavity_width_mm == pytest.approx(
            servo.body_width_mm + 2 * clearance
        )

    def test_ear_slot_is_wider_than_the_body_pocket(self, params):
        """Without this step there is no shelf for the servo's ears."""
        assert params.cavity_ear_length_mm > params.cavity_body_length_mm

    def test_cavity_offset_puts_the_shaft_on_the_yaw_axis(self, params):
        servo = DEFAULT_HARDWARE.base_yaw_servo
        assert params.cavity_offset_x_mm == pytest.approx(
            -servo.body_offset_from_shaft_axis_mm
        )

    def test_bearing_seat_matches_the_bearing_spec(self, params):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert params.bearing_seat_diameter_mm == pytest.approx(
            bearing.seat_diameter_mm
        )

    def test_bearing_stands_proud_of_the_top_face(self, params):
        """The turntable must ride the inner race, not the printed face."""
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert params.bearing_seat_depth_mm < bearing.width_mm

    def test_bolt_circle_matches_the_hardware_spec(self, params):
        assert 2 * params.bolt_circle_radius_mm == pytest.approx(
            DEFAULT_HARDWARE.mount_bolt_circle_diameter_mm
        )
        assert params.bolt_count == DEFAULT_HARDWARE.mount_bolt_count

    def test_bolt_positions_are_evenly_spaced_on_the_bolt_circle(self, params):
        positions = params.bolt_positions
        assert len(positions) == params.bolt_count
        for x, y in positions:
            assert np.hypot(x, y) == pytest.approx(params.bolt_circle_radius_mm)

    def test_bolt_positions_clear_the_servo_cavity(self, params):
        """The whole reason the bolt pattern is rotated 45 degrees."""
        half_ear = params.cavity_ear_length_mm / 2.0
        half_width = params.cavity_width_mm / 2.0
        needed = params.bolt_channel_diameter_mm / 2.0 + params.min_wall_thickness_mm
        for x, y in params.bolt_positions:
            gap = params._rect_distance_to_point(half_ear, half_width, x, y)
            assert gap >= needed, f"bolt at ({x:.2f}, {y:.2f}) is only {gap:.2f} mm away"

    def test_servo_screws_land_in_shelf_material(self, params):
        """Each M3 hole must be outside the upper pocket but inside the ear slot."""
        half_body = params.cavity_body_length_mm / 2.0
        half_ear = params.cavity_ear_length_mm / 2.0
        for x, _ in params.servo_screw_positions:
            offset = abs(x - params.cavity_offset_x_mm)
            assert offset > half_body, "screw would open into the upper pocket"
            assert offset < half_ear, "screw would miss the ear slot entirely"

    def test_body_radius_is_derived_not_assumed(self):
        """A larger servo must grow the body, not silently thin the wall."""
        bigger_servo = ServoSpec(
            body_length_mm=44.0, body_width_mm=22.0, flange_span_mm=58.0
        )
        bigger = PedestalParameters.from_geometry(
            hardware=HardwareSpec(base_yaw_servo=bigger_servo)
        )
        default = PedestalParameters.from_geometry()
        assert bigger.body_radius_mm > default.body_radius_mm
        # The wall is preserved, not consumed, by the larger part.
        assert bigger.validate() is DesignStatus.OK

    def test_flange_always_overhangs_the_body(self, params):
        assert params.flange_radius_mm > params.body_radius_mm

    def test_report_names_the_key_dimensions(self, params):
        report = params.report()
        for heading in ("Total height", "Body outer diameter", "Bearing seat"):
            assert heading in report


# =========================================================================
# 3. Design rule checks
# =========================================================================


class TestDesignRuleChecks:
    def test_stack_allowance_exceeding_base_height_reports_negative_height(self):
        greedy = HardwareSpec(
            base_stack=BaseStack(
                turntable_plate_thickness_mm=60.0, shoulder_bracket_rise_mm=60.0
            )
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(hardware=greedy)
        assert excinfo.value.status is DesignStatus.NEGATIVE_HEIGHT

    def test_servo_too_tall_for_the_pedestal_is_caught(self):
        """A short base height cannot swallow a full-height servo."""
        squat = ArmGeometry(base_height_mm=45.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(arm=squat)
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_servo_too_large_for_the_bolt_circle_is_rejected(self):
        """
        The 60 mm bolt circle is a real constraint on servo size.

        Because the bolt access channels run up through the body wall, a servo
        whose ear slot grows past them has nowhere for those channels to go.
        A 60 x 30 mm servo (76 mm ear span) crosses that boundary, and the
        design rule check must say so rather than emitting a part with a
        channel opening into the servo pocket.
        """
        oversized = HardwareSpec(
            base_yaw_servo=ServoSpec(
                body_length_mm=60.0, body_width_mm=30.0, flange_span_mm=76.0
            )
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(hardware=oversized)
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION
        assert "bolt circle" in str(excinfo.value)

    def test_bolt_pattern_aligned_with_the_cavity_is_rejected(self, params):
        """At 0 degrees, one bolt channel drives straight into the ear slot."""
        aligned = replace(params, bolt_azimuth_offset_deg=0.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            aligned.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_thin_wall_is_rejected(self, params):
        pinched = replace(params, body_radius_mm=params.body_radius_mm - 4.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            pinched.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_flange_smaller_than_body_is_rejected(self, params):
        inverted = replace(params, flange_radius_mm=params.body_radius_mm - 1.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            inverted.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_flange_thicker_than_the_part_is_rejected(self, params):
        absurd = replace(params, flange_thickness_mm=params.total_height_mm + 1.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            absurd.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_shaft_bore_wider_than_the_seat_is_rejected(self, params):
        bored_out = replace(
            params, shaft_bore_diameter_mm=params.bearing_seat_diameter_mm + 1.0
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            bored_out.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_screw_holes_breaking_through_the_shelf_are_rejected(self, params):
        deep = replace(params, servo_screw_hole_depth_mm=50.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            deep.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_oversized_cable_slot_is_rejected(self, params):
        wide = replace(params, cable_slot_width_mm=params.cavity_width_mm + 1.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            wide.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_bearing_proud_beyond_its_width_is_rejected(self):
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(bearing_proud_mm=99.0)
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"flange_thickness_mm": 0.0},
            {"flange_lip_mm": -1.0},
            {"bolt_channel_diameter_mm": 0.0},
            {"cable_slot_width_mm": -2.0},
        ],
    )
    def test_non_positive_construction_arguments_rejected(self, kwargs):
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(**kwargs)
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_design_error_message_carries_the_status_name(self, params):
        with pytest.raises(PedestalDesignError, match=r"\[WALL_TOO_THIN\]"):
            replace(params, body_radius_mm=params.body_radius_mm - 4.0).validate()

    def test_build_rejects_invalid_parameters(self, params):
        pinched = replace(params, body_radius_mm=params.body_radius_mm - 4.0)
        with pytest.raises(PedestalDesignError):
            build_pedestal(pinched)


# =========================================================================
# 4. Solid construction
# =========================================================================


class TestSolidConstruction:
    def test_module_builds_without_error(self, pedestal):
        assert pedestal is not None

    def test_part_is_a_single_solid(self, pedestal):
        assert len(pedestal.solids()) == 1, "the pedestal must not be fragmented"

    def test_part_volume_is_positive(self, pedestal):
        assert pedestal.volume > 0.0

    def test_part_is_smaller_than_its_bounding_cylinder(self, pedestal, params):
        """Sanity check that the internal pockets were actually subtracted."""
        solid_cylinder = (
            np.pi * params.flange_radius_mm**2 * params.total_height_mm
        )
        assert pedestal.volume < solid_cylinder

    def test_bounding_box_matches_the_designed_envelope(self, pedestal, params):
        box = pedestal.bounding_box()
        assert box.max.Z - box.min.Z == pytest.approx(params.total_height_mm, abs=1e-6)
        assert box.max.X - box.min.X == pytest.approx(
            2 * params.flange_radius_mm, abs=1e-6
        )

    def test_part_sits_on_the_z_origin(self, pedestal):
        """The underside must be at z = 0 so it drops onto a print bed."""
        assert pedestal.bounding_box().min.Z == pytest.approx(0.0, abs=1e-6)

    def test_build_uses_the_geometry_singletons_by_default(self):
        default_built = build_pedestal()
        assert default_built.bounding_box().max.Z == pytest.approx(
            DEFAULT_HARDWARE.pedestal_height_mm(DEFAULT_ARM), abs=1e-6
        )


# =========================================================================
# 5. STL export and mesh integrity
# =========================================================================


class TestStlExport:
    def test_export_creates_the_file(self, stl_path):
        assert stl_path.exists()
        assert stl_path.stat().st_size > 0

    def test_export_creates_missing_parent_directories(self, tmp_path, params):
        nested = tmp_path / "does" / "not" / "exist" / "pedestal.stl"
        assert export_pedestal(nested, params).exists()

    def test_file_is_a_well_formed_binary_stl(self, stl_path):
        """The declared triangle count must match the actual file length."""
        raw = stl_path.read_bytes()
        triangle_count = struct.unpack("<I", raw[80:84])[0]
        assert triangle_count > 0
        assert len(raw) == 84 + 50 * triangle_count

    def test_mesh_has_triangles(self, mesh):
        assert len(mesh.faces) > 0
        assert len(mesh.vertices) > 0

    def test_mesh_is_watertight(self, mesh):
        """
        Every undirected edge is used by exactly two faces.

        This is the definition of a closed surface: an edge used once is a
        boundary (a hole in the mesh), and an edge used three or more times is
        a non-manifold junction. Either makes the solid unprintable.
        """
        offenders = {
            edge: count for edge, count in mesh.edge_use_counts().items() if count != 2
        }
        assert not offenders, (
            f"{len(offenders)} non-manifold or boundary edge(s); "
            f"a closed mesh uses every edge exactly twice."
        )

    def test_mesh_normals_are_consistently_oriented(self, mesh):
        """
        Every directed edge is traversed exactly once.

        Two adjacent faces of a consistently-oriented surface walk their
        shared edge in opposite directions. A directed edge seen twice means
        one of the two faces is wound backwards, which flips its normal and
        confuses slicers about which side is solid.
        """
        offenders = {
            edge: count
            for edge, count in mesh.directed_edge_counts().items()
            if count != 1
        }
        assert not offenders, f"{len(offenders)} inconsistently-wound edge(s)."

    def test_mesh_volume_is_nonzero_and_positive(self, mesh):
        volume = mesh.signed_volume_mm3()
        assert volume > 0.0, (
            "non-positive enclosed volume means the mesh either encloses "
            "nothing or has inward-facing normals"
        )

    def test_mesh_volume_agrees_with_the_kernel(self, mesh, pedestal):
        """
        Tessellated volume must track the B-rep volume.

        A flat-faceted mesh under-fills curved surfaces, so the mesh volume is
        slightly *lower* than the exact solid. A 1% band catches a genuinely
        wrong export while tolerating that expected chord error.
        """
        assert mesh.signed_volume_mm3() == pytest.approx(pedestal.volume, rel=0.01)

    def test_mesh_bounding_box_matches_the_design(self, mesh, params):
        low, high = mesh.bounding_box()
        assert high[2] - low[2] == pytest.approx(params.total_height_mm, abs=1e-3)
        assert high[0] - low[0] == pytest.approx(
            2 * params.flange_radius_mm, rel=1e-3
        )

    def test_mesh_sits_on_the_z_origin(self, mesh):
        assert mesh.bounding_box()[0][2] == pytest.approx(0.0, abs=1e-6)
