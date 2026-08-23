"""
Test suite for the parametric CAD (Sessions D.1, D.1b and D.1c).

Test areas
----------
1. Hardware specifications in src.geometry (validation, derived values,
   the UNVERIFIED-field bookkeeping)
2. DeskClampSpec -- fastener standards, derived stack-up, clamp physics
3. U-clamp parameter derivation -- every dimension traceable to geometry.py
4. Design rule checks -- each violation raises with the right DesignStatus
5. Pressure foot and knob parameter derivation and design rule checks
6. Solid construction -- every part builds and has positive volume
7. STL export -- valid binary STL, closed watertight mesh, nonzero volume,
   checked for all three printable parts

The mesh checks parse the exported STL directly rather than trusting the
kernel's own report, because "the STL is watertight" is a property of the
tessellated output a slicer will actually read, not of the B-rep it came from.
"""

from __future__ import annotations

import collections
import math
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
    DeskClampSpec,
    FastenerSpec,
    HardwareSpec,
    ServoSpec,
)

build123d = pytest.importorskip(
    "build123d", reason="build123d is required for the CAD suite (Python >= 3.10)"
)

from cad._design import DesignStatus  # noqa: E402  - must follow the importorskip
from cad._primitives import hex_prism, right_triangle_prism  # noqa: E402
from cad.base_pedestal import (  # noqa: E402
    PedestalDesignError,
    PedestalParameters,
    build_pedestal,
    export_pedestal,
)
from cad.desk_clamp_knob import (  # noqa: E402
    KnobDesignError,
    KnobParameters,
    build_knob,
    export_knob,
)
from cad.assembly_preview import (  # noqa: E402
    build_assembly,
    export_assembly_stl,
    render_assembly_png,
)
from cad.desk_clamp_pressure_foot import (  # noqa: E402
    PressureFootDesignError,
    PressureFootParameters,
    build_pressure_foot,
    export_pressure_foot,
)

#: Every printable part, so the mesh-integrity checks cover all of them.
PART_EXPORTERS = {
    "base_pedestal": export_pedestal,
    "desk_clamp_knob": export_knob,
    "desk_clamp_pressure_foot": export_pressure_foot,
}
PART_BUILDERS = {
    "base_pedestal": build_pedestal,
    "desk_clamp_knob": build_knob,
    "desk_clamp_pressure_foot": build_pressure_foot,
}


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

        unique, inverse = np.unique(
            np.round(soup, decimals), axis=0, return_inverse=True
        )
        faces = inverse.reshape(-1, 3)

        # Drop zero-area triangles before any topology analysis. OpenCASCADE
        # emits two of them at each pole when it tessellates a sphere -- a
        # triangle whose two corners weld to the same vertex. They carry no
        # surface, but they contribute phantom edges that would otherwise
        # register as non-manifold. Discarding degenerate faces is standard
        # mesh cleanup and leaves the enclosed volume unchanged.
        non_degenerate = np.array(
            [len(set(face.tolist())) == 3 for face in faces], dtype=bool
        )
        return cls(unique, faces[non_degenerate])

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
    """Default pedestal parameters, derived from the geometry singletons."""
    return PedestalParameters.from_geometry()


@pytest.fixture(scope="module")
def pedestal(params):
    """The built pedestal solid. Module-scoped: the kernel work is not free."""
    return build_pedestal(params)


@pytest.fixture(scope="module")
def stl_paths(tmp_path_factory) -> dict:
    """Every part exported once into a temp directory, keyed by part name."""
    destination = tmp_path_factory.mktemp("cad")
    return {
        name: exporter(destination / f"{name}.stl")
        for name, exporter in PART_EXPORTERS.items()
    }


@pytest.fixture(scope="module")
def meshes(stl_paths) -> dict:
    return {name: Mesh.from_binary_stl(path) for name, path in stl_paths.items()}


@pytest.fixture(scope="module")
def foot() -> PressureFootParameters:
    """Default pressure-foot parameters, derived from the geometry singletons."""
    return PressureFootParameters.from_geometry()


@pytest.fixture(scope="module")
def knob() -> KnobParameters:
    """Default knob parameters, derived from the geometry singletons."""
    return KnobParameters.from_geometry()


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
        """FastenerSpec survives D.1b as a generic spec, though nothing uses M4."""
        fastener = FastenerSpec()
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

    def test_pedestal_height_is_base_height_minus_everything_above_it(self):
        expected = (
            DEFAULT_ARM.base_height_mm
            - DEFAULT_HARDWARE.above_pedestal_allowance_mm
        )
        assert DEFAULT_HARDWARE.pedestal_height_mm() == pytest.approx(expected)
        assert DEFAULT_HARDWARE.pedestal_height_mm() == pytest.approx(69.5)

    def test_base_stack_accounts_for_the_bearings_proud_height(self):
        """
        Session D.1d fix: the turntable rides the bearing's inner race, which
        stands above the turret's top face rather than flush with it. Leaving
        that 0.5 mm out of the budget had put the shoulder pivot 0.5 mm high.
        """
        hardware = DEFAULT_HARDWARE
        assert hardware.above_pedestal_allowance_mm == pytest.approx(
            hardware.thrust_bearing.proud_mm + hardware.base_stack.allowance_mm
        )
        assert hardware.above_pedestal_allowance_mm > hardware.base_stack.allowance_mm

    def test_the_base_stack_closes_exactly_on_the_shoulder_pivot(self):
        """
        The whole point of budgeting the stack: turret top plus everything
        above it must land on base_height_mm to the millimetre, or FK's
        shoulder pivot is wrong by the difference.
        """
        hardware = DEFAULT_HARDWARE
        total = (
            hardware.pedestal_height_mm()
            + hardware.thrust_bearing.proud_mm
            + hardware.base_stack.turntable_plate_thickness_mm
            + hardware.base_stack.shoulder_bracket_rise_mm
        )
        assert total == pytest.approx(DEFAULT_ARM.base_height_mm, abs=1e-9)
        assert total == pytest.approx(hardware.shoulder_pivot_z_mm(), abs=1e-9)

    def test_bearing_seat_depth_leaves_the_bearing_proud(self):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert bearing.seat_depth_mm == pytest.approx(
            bearing.width_mm - bearing.proud_mm
        )
        assert bearing.seat_depth_mm < bearing.width_mm

    @pytest.mark.parametrize("proud", [0.0, -0.5, 7.0, 9.0])
    def test_invalid_bearing_proud_rejected(self, proud):
        with pytest.raises(ValueError, match="proud_mm"):
            BearingSpec(proud_mm=proud)

    def test_pedestal_height_tracks_arm_geometry(self):
        """Raising base_height_mm must lengthen the pedestal, not something else."""
        taller = ArmGeometry(base_height_mm=150.0)
        assert DEFAULT_HARDWARE.pedestal_height_mm(taller) == pytest.approx(
            150.0 - DEFAULT_HARDWARE.above_pedestal_allowance_mm
        )

    def test_stack_allowance_exceeding_base_height_rejected(self):
        greedy = HardwareSpec(
            base_stack=BaseStack(
                turntable_plate_thickness_mm=60.0, shoulder_bracket_rise_mm=60.0
            )
        )
        with pytest.raises(ValueError, match="non-positive height"):
            greedy.pedestal_height_mm()

    def test_hardware_report_surfaces_the_verification_warning(self):
        report = DEFAULT_HARDWARE.hardware_report()
        assert "UNVERIFIED" in report
        assert "608ZZ" in report and "DS3218" in report

    def test_bill_of_materials_lists_the_clamp_fasteners(self):
        """The M4 desk screws are gone; the clamp hardware replaced them."""
        lines = "\n".join(DEFAULT_HARDWARE.bill_of_materials())
        assert "M8" in lines
        assert "hex nut" in lines
        assert "rubber" in lines
        assert "M4" not in lines

    def test_arm_tipping_moment_uses_mass_and_reach(self):
        arm = ArmGeometry(estimated_arm_mass_kg=0.625, max_payload_kg=0.100)
        expected = 0.725 * (arm.total_reach_mm / 1000.0) * 9.81
        assert arm.tipping_moment_nm() == pytest.approx(expected)

    def test_tipping_moment_scales_with_reach(self):
        short = ArmGeometry(l1_upper_arm_mm=200.0)
        assert short.tipping_moment_nm() < DEFAULT_ARM.tipping_moment_nm()


# =========================================================================
# 2. DeskClampSpec -- standards, stack-up, and clamp physics
# =========================================================================


class TestDeskClampSpec:
    def test_m8_fastener_dimensions_match_standards(self):
        """
        DIN 933 hex head and ISO 273 clearance for M8.

        Head 13.00 mm across flats / 5.30 mm tall, coarse pitch 1.25 mm,
        medium-series clearance hole 9.0 mm.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bolt_thread == "M8"
        assert clamp.bolt_nominal_diameter_mm == pytest.approx(8.0)
        assert clamp.bolt_thread_pitch_mm == pytest.approx(1.25)
        assert clamp.bolt_head_across_flats_mm == pytest.approx(13.0)
        assert clamp.bolt_head_height_mm == pytest.approx(5.30)
        assert clamp.bolt_clearance_hole_diameter_mm == pytest.approx(9.0)

    def test_m8_nut_across_flats_is_the_standard_13mm(self):
        assert DEFAULT_HARDWARE.desk_clamp.nut_across_flats_mm == pytest.approx(13.0)

    def test_nut_pocket_uses_the_max_not_nominal_thickness(self):
        """
        A pocket must fit the largest nut it might receive.

        Legacy DIN 934 gives m = 6.5 for M8; DIN EN ISO 4032 gives 6.80 max.
        Both ship as "DIN 934", so cutting to 6.5 would fail to seat a modern
        nut. The pocket derives from the larger figure.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.nut_thickness_nominal_mm == pytest.approx(6.50)
        assert clamp.nut_thickness_max_mm == pytest.approx(6.80)
        assert clamp.nut_pocket_depth_mm == pytest.approx(clamp.nut_thickness_max_mm)
        assert clamp.nut_pocket_depth_mm > clamp.nut_thickness_nominal_mm

    def test_nut_across_corners_is_the_hex_relation(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.nut_across_corners_mm == pytest.approx(
            clamp.nut_across_flats_mm * 2.0 / math.sqrt(3.0)
        )

    def test_max_nut_thickness_below_nominal_rejected(self):
        with pytest.raises(ValueError, match="must not be less than"):
            DeskClampSpec(nut_thickness_nominal_mm=7.0, nut_thickness_max_mm=6.8)

    # ---- U-profile ---------------------------------------------------------

    def test_clamp_throat_opens_wider_than_desk_thickness(self):
        """
        A U-clamp's throat is fixed by the printed geometry.

        Unlike a screw-adjusted jaw it cannot be opened further, so the gap
        has to exceed the thickest supported desk outright or the clamp will
        not go on at all.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.throat_max_opening_mm > clamp.max_desk_thickness_mm
        assert clamp.max_desk_thickness_mm == pytest.approx(35.0)
        assert clamp.throat_max_opening_mm == pytest.approx(45.0)
        assert clamp.desk_removal_clearance_mm == pytest.approx(10.0)

    def test_throat_not_exceeding_desk_thickness_rejected(self):
        with pytest.raises(ValueError, match="must exceed the thickest"):
            DeskClampSpec(
                desk_thickness_range_mm=(15.0, 35.0), throat_max_opening_mm=30.0
            )

    def test_desk_thickness_range_must_increase(self):
        with pytest.raises(ValueError, match="increasing"):
            DeskClampSpec(desk_thickness_range_mm=(35.0, 15.0))

    def test_u_profile_dimensions_are_carried(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.top_arm_depth_mm == pytest.approx(60.0)
        assert clamp.bottom_arm_depth_mm == pytest.approx(60.0)
        assert clamp.top_arm_thickness_mm == pytest.approx(15.0)
        assert clamp.bottom_arm_thickness_mm == pytest.approx(15.0)
        assert clamp.spine_thickness_mm == pytest.approx(15.0)
        assert clamp.servo_shaft_offset_from_edge_mm == pytest.approx(30.0)

    def test_shaft_offset_must_land_on_the_top_arm(self):
        with pytest.raises(ValueError, match="fall off the end"):
            DeskClampSpec(
                servo_shaft_offset_from_edge_mm=70.0, top_arm_depth_mm=60.0
            )

    def test_bottom_arm_keeps_a_floor_above_the_nut_pocket(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bottom_arm_floor_mm == pytest.approx(
            clamp.bottom_arm_thickness_mm - clamp.nut_pocket_depth_mm
        )
        assert clamp.bottom_arm_floor_mm >= DEFAULT_HARDWARE.min_wall_thickness_mm

    def test_nut_pocket_deeper_than_the_bottom_arm_rejected(self):
        with pytest.raises(ValueError, match="no floor"):
            DeskClampSpec(bottom_arm_thickness_mm=6.0)

    def test_pad_thicker_than_its_recess_rejected(self):
        with pytest.raises(ValueError, match="stand proud"):
            DeskClampSpec(pad_recess_depth_mm=2.0, pad_thickness_mm=3.0)

    # ---- Pressure foot -----------------------------------------------------

    def test_pressure_foot_height_is_the_sum_of_its_layers(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.pressure_foot_height_mm == pytest.approx(
            clamp.pressure_foot_bore_depth_mm
            + clamp.pressure_foot_web_mm
            + clamp.pad_recess_depth_mm
        )

    def test_pressure_foot_rise_shortens_the_screw_reach_needed(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.pressure_foot_rise_above_tip_mm == pytest.approx(
            clamp.pressure_foot_web_mm + clamp.pad_recess_depth_mm
        )
        assert clamp.max_screw_protrusion_mm == pytest.approx(
            clamp.throat_max_opening_mm
            - clamp.min_desk_thickness_mm
            - clamp.pressure_foot_rise_above_tip_mm
        )

    def test_pressure_foot_fits_the_throat_at_the_thickest_desk(self):
        """The tightest case: a thick desk leaves the least room below it."""
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.pressure_foot_height_mm <= clamp.desk_removal_clearance_mm

    def test_pressure_foot_pad_must_fit_inside_the_foot(self):
        with pytest.raises(ValueError, match="smaller than"):
            DeskClampSpec(
                pressure_foot_diameter_mm=20.0, pressure_foot_pad_diameter_mm=20.0
            )

    def test_pressure_foot_bore_must_fit_inside_its_pad(self):
        with pytest.raises(ValueError, match="bore must be smaller"):
            DeskClampSpec(pressure_foot_bore_diameter_mm=20.0)

    # ---- Screw length ------------------------------------------------------

    def test_specified_bolt_is_long_enough_for_the_stack(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bolt_length_mm >= clamp.required_bolt_length_mm
        assert clamp.bolt_length_mm == pytest.approx(70.0)

    def test_required_bolt_length_is_set_by_the_thinnest_desk(self):
        """
        A thin desk sits high in the throat, so the screw must reach furthest.

        Raising the minimum supported desk thickness therefore *shortens* the
        screw needed, which is the opposite of the intuitive direction.
        """
        thin = DeskClampSpec(desk_thickness_range_mm=(10.0, 35.0))
        thick = DeskClampSpec(desk_thickness_range_mm=(25.0, 35.0))
        assert thin.required_bolt_length_mm > thick.required_bolt_length_mm

    def test_boss_not_exceeding_bolt_hole_rejected(self):
        with pytest.raises(ValueError, match="must exceed the bolt"):
            DeskClampSpec(knob_boss_diameter_mm=9.0)

    # ---- Physics -----------------------------------------------------------

    def test_preload_rises_with_torque(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bolt_preload_n(5.0) > clamp.bolt_preload_n(1.0) > 0.0

    def test_preload_and_torque_round_trip(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.preload_to_torque_nm(clamp.bolt_preload_n(3.0)) == pytest.approx(
            3.0
        )

    def test_coarser_thread_gives_less_preload_for_the_same_torque(self):
        """
        The pitch really is in the model.

        A coarser thread advances further per turn, so more of the applied
        torque goes into travel and less into tension.
        """
        fine = DeskClampSpec(bolt_thread_pitch_mm=1.00)
        coarse = DeskClampSpec(bolt_thread_pitch_mm=1.50)
        assert coarse.bolt_preload_n(5.0) < fine.bolt_preload_n(5.0)

    def test_larger_boss_wastes_torque_on_collar_friction(self):
        small = DeskClampSpec(knob_boss_diameter_mm=18.0)
        large = DeskClampSpec(knob_boss_diameter_mm=40.0)
        assert large.bolt_preload_n(5.0) < small.bolt_preload_n(5.0)

    def test_negative_torque_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            DEFAULT_HARDWARE.desk_clamp.bolt_preload_n(-1.0)

    def test_hand_torque_scales_with_knob_radius(self):
        """
        The whole point of choosing a knob size.

        Torque a hand can apply is grip force times radius, so knob diameter
        caps the achievable clamp force physically rather than by warning.
        """
        big = DeskClampSpec(knob_diameter_mm=50.0)
        small = DeskClampSpec(knob_diameter_mm=30.0)
        assert big.hand_torque_limit_nm() == pytest.approx(
            40.0 * 0.025
        )
        assert small.hand_torque_limit_nm() < big.hand_torque_limit_nm()

    def test_negative_grip_force_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            DEFAULT_HARDWARE.desk_clamp.hand_torque_limit_nm(-1.0)

    def test_pad_friction_uses_the_stated_coefficient(self):
        clamp = DeskClampSpec(pad_friction_coefficient=0.4)
        assert clamp.pad_friction_force_n(1000.0) == pytest.approx(400.0)

    def test_resisting_moment_requires_a_positive_lever(self):
        with pytest.raises(ValueError, match="lever_arm_mm must be positive"):
            DEFAULT_HARDWARE.desk_clamp.clamp_resisting_moment_nm(5.0, 0.0)

    def test_bottom_arm_allowable_preload_falls_with_overhang(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bottom_arm_allowable_preload_n(
            82.5, 40.0
        ) < clamp.bottom_arm_allowable_preload_n(82.5, 10.0)

    @pytest.mark.parametrize("args", [(0.0, 10.0), (82.5, 0.0)])
    def test_non_positive_section_arguments_rejected(self, args):
        with pytest.raises(ValueError, match="must be positive"):
            DEFAULT_HARDWARE.desk_clamp.bottom_arm_allowable_preload_n(*args)


# =========================================================================
# 3. Clamp capacity against the arm's real loads
# =========================================================================


class TestClampCapacity:
    def test_clamp_grip_force_exceeds_arm_tipping_torque(self, params):
        """
        The headline requirement: a tightened clamp holds the arm down.

        Grip force comes from the M8 thread pitch, thread and collar friction,
        and applied torque. It resists the arm's worst-case tipping moment
        across the lever from the top arm's inboard edge -- the pivot the
        assembly would rotate about -- to the clamp screw.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        grip_force_n = clamp.bolt_preload_n(5.0)
        resisting_nm = clamp.clamp_resisting_moment_nm(
            5.0, params.tipping_lever_arm_mm
        )
        tipping_nm = DEFAULT_ARM.tipping_moment_nm()

        assert grip_force_n > 0.0
        assert tipping_nm > 0.0
        assert resisting_nm > tipping_nm, (
            f"clamp resists {resisting_nm:.2f} N.m but the arm tips at "
            f"{tipping_nm:.2f} N.m"
        )
        assert resisting_nm > 5.0 * tipping_nm

    def test_hand_achievable_grip_exceeds_tipping(self, params):
        """
        The version that matters, using torque a hand can actually apply.

        5 N.m is not reachable on a knob this size, so the test above proves
        the physics but not the product. This one uses
        ``hand_torque_limit_nm`` and still has to clear the arm's tipping
        moment -- against the deliberately pessimistic figure, with the whole
        arm mass at full reach.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        hand_torque = clamp.hand_torque_limit_nm()
        resisting_nm = clamp.clamp_resisting_moment_nm(
            hand_torque, params.tipping_lever_arm_mm
        )
        assert resisting_nm > DEFAULT_ARM.tipping_moment_nm()

    def test_required_preload_is_within_the_bottom_arms_structural_limit(
        self, params
    ):
        """The preload needed to hold the arm must sit inside what the U can carry."""
        clamp = DEFAULT_HARDWARE.desk_clamp
        required_n = DEFAULT_ARM.tipping_moment_nm() / (
            params.tipping_lever_arm_mm / 1000.0
        )
        allowable_n = clamp.bottom_arm_allowable_preload_n(
            params.clamp_width_mm, params.bottom_arm_overhang_mm
        )
        assert required_n < allowable_n
        assert allowable_n / required_n > 5.0

    def test_hand_cannot_overstress_the_u_clamp(self, params):
        """
        The U-profile inverts the old design's weakness.

        The D.1b side wing yielded at 3.9 N.m, below what a hand could apply,
        so the README had to carry a torque warning. A 15 mm bottom arm
        spanning the full clamp width cannot be over-stressed by hand at all,
        which is a structural guarantee rather than an instruction.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        structural_limit = clamp.max_tightening_torque_nm(
            params.clamp_width_mm, params.bottom_arm_overhang_mm
        )
        assert structural_limit > clamp.hand_torque_limit_nm()

    def test_pad_friction_resists_lateral_load(self, params):
        """
        Sliding check, using the PETG-on-wood coefficient.

        Compared against a deliberately harsh proxy for lateral load: the
        entire arm mass accelerated sideways at 1 g.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        preload_n = DEFAULT_ARM.tipping_moment_nm() / (
            params.tipping_lever_arm_mm / 1000.0
        )
        friction_n = clamp.pad_friction_force_n(preload_n)
        lateral_n = (
            DEFAULT_ARM.estimated_arm_mass_kg + DEFAULT_ARM.max_payload_kg
        ) * 9.81
        assert friction_n > lateral_n


# =========================================================================
# 4. U-clamp parameter derivation
# =========================================================================


class TestParameterDerivation:
    def test_default_parameters_validate(self, params):
        assert params.validate() is DesignStatus.OK

    def test_turret_top_comes_from_the_geometry_singletons(self, params):
        assert params.turret_top_z_mm == pytest.approx(
            DEFAULT_HARDWARE.pedestal_height_mm(DEFAULT_ARM)
        )
        assert params.turret_top_z_mm == pytest.approx(69.5)

    def test_taller_base_height_gives_a_taller_turret(self):
        taller = PedestalParameters.from_geometry(arm=ArmGeometry(base_height_mm=150.0))
        assert taller.turret_top_z_mm == pytest.approx(119.5)

    def test_desk_seating_plane_matches_the_shaft_offset(self, params):
        """The desk seats one shaft-offset outboard of the yaw axis."""
        assert params.desk_seat_x_mm == pytest.approx(
            -DEFAULT_HARDWARE.desk_clamp.servo_shaft_offset_from_edge_mm
        )

    def test_the_gusset_defines_where_the_desk_stops(self, params):
        """
        The upper gusset hangs into the throat, so the desk's edge comes to
        rest on it, not on the spine. The spine therefore sits one gusset
        further out -- otherwise the yaw axis would end up
        gusset_size_mm closer to the edge than specified.
        """
        assert params.spine_inner_x_mm == pytest.approx(
            params.desk_seat_x_mm - params.gusset_size_mm
        )
        assert params.spine_inner_x_mm < params.desk_seat_x_mm

    def test_yaw_axis_offset_agrees_with_arm_geometry(self):
        """
        The clamp's shaft offset and ArmGeometry's base_y are the same
        measurement in two frames, so they must not drift apart.
        """
        assert DEFAULT_ARM.base_y_on_desk_mm == pytest.approx(
            DEFAULT_HARDWARE.desk_clamp.servo_shaft_offset_from_edge_mm
        )

    def test_u_profile_spans_match_the_clamp_spec(self, params):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert params.top_arm_depth_mm == pytest.approx(clamp.top_arm_depth_mm)
        assert params.spine_outer_x_mm == pytest.approx(
            params.spine_inner_x_mm - clamp.spine_thickness_mm
        )
        assert params.throat_bottom_z_mm == pytest.approx(
            -clamp.throat_max_opening_mm
        )
        assert params.bottom_arm_bottom_z_mm == pytest.approx(
            -clamp.throat_max_opening_mm - clamp.bottom_arm_thickness_mm
        )

    def test_overall_height_spans_turret_to_bottom_arm(self, params):
        assert params.overall_height_mm == pytest.approx(
            params.turret_top_z_mm - params.bottom_arm_bottom_z_mm
        )
        assert params.overall_height_mm == pytest.approx(129.5)

    def test_cavity_is_the_servo_body_plus_clearance(self, params):
        servo = DEFAULT_HARDWARE.base_yaw_servo
        clearance = DEFAULT_HARDWARE.print_clearance_mm
        assert params.cavity_x_span_mm == pytest.approx(
            servo.body_width_mm + 2 * clearance
        )
        assert params.cavity_body_span_y_mm == pytest.approx(
            servo.body_length_mm + 2 * clearance
        )

    def test_servo_long_axis_runs_across_the_clamp(self, params):
        """
        Along the arm it would not fit.

        The ear slot is 54.5 mm and the body sits 10 mm off the shaft, so
        along X it would reach 37.25 mm from the axis and break out past the
        spine 30 mm away. Across Y the arm only has to be 20.5 mm deep.
        """
        assert params.cavity_ear_span_y_mm > params.cavity_x_span_mm
        assert params.cavity_x_span_mm < params.top_arm_depth_mm

    def test_ear_slot_is_wider_than_the_body_pocket(self, params):
        """Without this step there is no shelf for the servo's ears."""
        assert params.cavity_ear_span_y_mm > params.cavity_body_span_y_mm

    def test_cavity_offset_puts_the_shaft_on_the_yaw_axis(self, params):
        servo = DEFAULT_HARDWARE.base_yaw_servo
        assert params.cavity_offset_y_mm == pytest.approx(
            -servo.body_offset_from_shaft_axis_mm
        )

    def test_turret_encloses_the_cavity_with_a_wall(self, params):
        wall = params.min_wall_thickness_mm
        assert params.turret_x_max_mm - params.cavity_x_span_mm / 2.0 >= wall
        cavity_y_max = (
            params.cavity_offset_y_mm + params.cavity_ear_span_y_mm / 2.0
        )
        cavity_y_min = (
            params.cavity_offset_y_mm - params.cavity_ear_span_y_mm / 2.0
        )
        assert params.turret_y_max_mm - cavity_y_max >= wall
        assert cavity_y_min - params.turret_y_min_mm >= wall

    def test_turret_encloses_the_bearing_seat_with_a_wall(self, params):
        """
        The seat is wider than the cavity in X, so it is what sizes the turret
        there -- a turret sized only from the cavity would break out.
        """
        seat_radius = params.bearing_seat_diameter_mm / 2.0
        assert params.turret_x_max_mm - seat_radius >= params.min_wall_thickness_mm
        assert seat_radius > params.cavity_x_span_mm / 2.0

    def test_turret_sits_within_the_top_arm(self, params):
        assert params.turret_x_min_mm >= params.spine_inner_x_mm
        assert params.turret_x_max_mm <= params.top_arm_inner_x_mm

    def test_clamp_width_covers_the_turret(self, params):
        half = params.clamp_width_mm / 2.0
        assert params.turret_y_min_mm >= -half
        assert params.turret_y_max_mm <= half
        assert params.clamp_width_mm == pytest.approx(82.5)

    def test_bearing_seat_matches_the_bearing_spec(self, params):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert params.bearing_seat_diameter_mm == pytest.approx(
            bearing.seat_diameter_mm
        )

    def test_servo_shaft_output_is_the_bearing_seat_floor(self, params):
        """
        Where the yaw drive actually emerges. Exposed as a property so the
        preview and the tests read the number the solid was cut from, rather
        than each recomputing it and drifting apart.
        """
        assert params.servo_shaft_output_z_mm == pytest.approx(
            params.turret_top_z_mm - params.bearing_seat_depth_mm
        )
        assert params.servo_shaft_output_z_mm == pytest.approx(63.0)

    def test_shaft_output_sits_below_the_shoulder_pivot(self, params):
        """
        The gap between them is the base stack, not an error.

        The brief for D.1d assumed the shaft output was above 100 mm and that
        base_height_mm needed raising to match. It is at 63 mm: the 37 mm to
        the pivot is the bearing, the yaw turntable and the shoulder bracket,
        none of which is designed yet.
        """
        assert params.servo_shaft_output_z_mm < DEFAULT_ARM.base_height_mm
        assert params.turret_top_z_mm < DEFAULT_ARM.base_height_mm

    def test_bearing_top_is_the_turntable_datum(self, params):
        """The turntable rides the inner race, so the bearing's top is the datum."""
        assert params.bearing_top_z_mm == pytest.approx(
            params.turret_top_z_mm + params.bearing_proud_mm
        )
        assert params.bearing_top_z_mm == pytest.approx(70.0)
        assert params.bearing_top_z_mm + DEFAULT_HARDWARE.base_stack.allowance_mm == (
            pytest.approx(DEFAULT_ARM.base_height_mm)
        )

    def test_bearing_stands_proud_of_the_turret_top(self, params):
        """The turntable must ride the inner race, not the printed face."""
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert params.bearing_seat_depth_mm < bearing.width_mm

    def test_pedestal_internals_survived_the_u_clamp_rewrite(self, params):
        """
        D.1c changed the mounting, not the servo/bearing geometry.

        Those dimensions were settled in D.1 and must not drift while the
        surrounding body is rewritten.
        """
        assert params.cavity_body_span_y_mm == pytest.approx(40.5)
        assert params.cavity_ear_span_y_mm == pytest.approx(54.5)
        assert params.cavity_x_span_mm == pytest.approx(20.5)
        assert params.ear_shelf_z_mm == pytest.approx(49.0)
        assert params.cavity_top_z_mm == pytest.approx(59.0)
        assert params.bearing_seat_diameter_mm == pytest.approx(21.9)

    def test_pads_flank_the_cavity_opening(self, params):
        """
        The cavity opens in the middle of the top arm's underside, so a single
        pad cannot sit clear of it. Two strips straddle it instead.
        """
        assert len(params.pad_recesses) == 2
        cavity_half = params.cavity_x_span_mm / 2.0
        outboard, inboard = params.pad_recesses
        assert outboard[1] <= -cavity_half
        assert inboard[0] >= cavity_half

    def test_pads_start_at_the_desk_seating_plane(self, params):
        """No point putting pad outboard of where the desk actually reaches."""
        outboard = params.pad_recesses[0]
        assert outboard[0] >= params.desk_seat_x_mm

    def test_pads_stay_inside_the_top_arm(self, params):
        for x_min, x_max, y_min, y_max in params.pad_recesses:
            assert x_min >= params.spine_inner_x_mm
            assert x_max <= params.top_arm_inner_x_mm
            assert y_min >= -params.clamp_width_mm / 2.0
            assert y_max <= params.clamp_width_mm / 2.0

    def test_pad_area_beats_the_single_forty_square_pad(self, params):
        """Two strips give more contact than the 40 x 40 they replaced."""
        assert params.pad_area_mm2 > 40.0 * 40.0

    def test_screw_sits_as_far_outboard_as_the_foot_allows(self, params):
        """
        Outboard placement is not cosmetic: it shortens the bottom arm's
        cantilever and lengthens the tipping lever at the same time.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert params.bolt_axis_x_mm == pytest.approx(
            params.desk_seat_x_mm
            + clamp.pad_edge_margin_mm
            + params.pressure_foot_diameter_mm / 2.0
        )
        assert params.bolt_axis_x_mm < 0.0

    def test_pressure_foot_stays_under_the_desk(self, params):
        """Its outer edge must not stray past the desk edge into free air."""
        foot_outer = params.bolt_axis_x_mm - params.pressure_foot_diameter_mm / 2.0
        assert foot_outer >= params.desk_seat_x_mm

    def test_bottom_arm_overhang_is_measured_from_the_spine(self, params):
        assert params.bottom_arm_overhang_mm == pytest.approx(
            params.bolt_axis_x_mm - params.spine_inner_x_mm
        )
        assert params.bottom_arm_overhang_mm > 0.0

    def test_tipping_lever_is_measured_from_the_top_arms_inboard_edge(self, params):
        assert params.tipping_lever_arm_mm == pytest.approx(
            params.top_arm_inner_x_mm - params.bolt_axis_x_mm
        )
        assert params.tipping_lever_arm_mm > params.bottom_arm_overhang_mm

    def test_servo_screws_land_in_shelf_material(self, params):
        """Each M3 hole must be outside the upper pocket but inside the ear slot."""
        half_body = params.cavity_body_span_y_mm / 2.0
        half_ear = params.cavity_ear_span_y_mm / 2.0
        for _, y in params.servo_screw_positions:
            offset = abs(y - params.cavity_offset_y_mm)
            assert offset > half_body, "screw would open into the upper pocket"
            assert offset < half_ear, "screw would miss the ear slot entirely"

    def test_turret_is_derived_not_assumed(self):
        """A larger servo must grow the turret, not silently thin its wall."""
        bigger = PedestalParameters.from_geometry(
            hardware=HardwareSpec(
                base_yaw_servo=ServoSpec(
                    body_length_mm=44.0, body_width_mm=22.0, flange_span_mm=58.0
                )
            )
        )
        default = PedestalParameters.from_geometry()
        assert bigger.turret_x_max_mm >= default.turret_x_max_mm
        assert bigger.clamp_width_mm > default.clamp_width_mm
        assert bigger.validate() is DesignStatus.OK

    def test_report_names_the_key_dimensions(self, params):
        report = params.report()
        for heading in (
            "Overall envelope", "Servo turret", "Bearing seat", "Throat",
            "Anti-slip pads", "grip margin",
        ):
            assert heading in report


# =========================================================================
# 5. Design rule checks
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

    def test_servo_too_tall_for_the_turret_is_caught(self):
        """A short base height cannot swallow a full-height servo."""
        squat = ArmGeometry(base_height_mm=45.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(arm=squat)
        assert excinfo.value.status in (
            DesignStatus.FEATURE_COLLISION,
            DesignStatus.NEGATIVE_HEIGHT,
        )

    def test_short_clamp_screw_is_rejected(self):
        """
        The screw has to span knob, nut, bottom arm and the throat.

        A 40 mm M8 cannot reach the thinnest supported desk, so the check must
        catch it rather than ship a clamp that cannot touch the desk.
        """
        short = HardwareSpec(desk_clamp=DeskClampSpec(bolt_length_mm=40.0))
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(hardware=short)
        assert excinfo.value.status is DesignStatus.FASTENER_TOO_SHORT

    def test_turret_wall_too_thin_is_rejected(self, params):
        pinched = replace(
            params, turret_x_max_mm=params.cavity_x_span_mm / 2.0 + 1.0
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            pinched.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_turret_overhanging_the_top_arm_is_rejected(self, params):
        overhanging = replace(
            params, turret_x_max_mm=params.top_arm_inner_x_mm + 5.0
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            overhanging.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_pad_recess_deeper_than_the_top_arm_is_rejected(self, params):
        cut_through = replace(
            params, pad_recess_depth_mm=params.top_arm_thickness_mm
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            cut_through.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_oversized_gussets_are_rejected(self, params):
        """Two gussets meeting in the middle would close the throat."""
        fat = replace(params, gusset_size_mm=params.throat_opening_mm / 2.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            fat.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_screw_outboard_of_the_spine_is_rejected(self, params):
        """Its pressure foot would press on air."""
        adrift = replace(params, bolt_axis_x_mm=params.spine_inner_x_mm - 5.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            adrift.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_pressure_foot_fouling_the_gusset_is_rejected(self, params):
        huge_foot = replace(params, pressure_foot_diameter_mm=60.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            huge_foot.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_thin_nut_floor_is_rejected(self, params):
        squashed = replace(
            params, bottom_arm_thickness_mm=params.nut_pocket_depth_mm + 1.0
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            squashed.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_bolt_hole_wider_than_the_nut_pocket_is_rejected(self, params):
        bored = replace(
            params, bolt_hole_diameter_mm=params.nut_pocket_across_flats_mm + 1.0
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            bored.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

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
        wide = replace(
            params, cable_slot_width_mm=params.cavity_ear_span_y_mm + 1.0
        )
        with pytest.raises(PedestalDesignError) as excinfo:
            wide.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"cable_slot_width_mm": -2.0},
            {"servo_screw_hole_depth_mm": 0.0},
            {"ear_top_offset_from_body_top_mm": -1.0},
        ],
    )
    def test_non_positive_construction_arguments_rejected(self, kwargs):
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(**kwargs)
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_design_error_message_carries_the_status_name(self, params):
        with pytest.raises(PedestalDesignError, match=r"\[WALL_TOO_THIN\]"):
            replace(
                params, turret_x_max_mm=params.cavity_x_span_mm / 2.0 + 1.0
            ).validate()

    def test_build_rejects_invalid_parameters(self, params):
        pinched = replace(
            params, turret_x_max_mm=params.cavity_x_span_mm / 2.0 + 1.0
        )
        with pytest.raises(PedestalDesignError):
            build_pedestal(pinched)


# =========================================================================
# 6. Pressure foot
# =========================================================================


class TestPressureFoot:
    def test_default_parameters_validate(self, foot):
        assert foot.validate() is DesignStatus.OK

    def test_dimensions_come_from_the_clamp_spec(self, foot):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert foot.diameter_mm == pytest.approx(clamp.pressure_foot_diameter_mm)
        assert foot.height_mm == pytest.approx(clamp.pressure_foot_height_mm)
        assert foot.pad_diameter_mm == pytest.approx(
            clamp.pressure_foot_pad_diameter_mm
        )

    def test_height_is_the_sum_of_its_layers(self, foot):
        assert foot.height_mm == pytest.approx(
            foot.bore_depth_mm + foot.web_thickness_mm + foot.pad_recess_depth_mm
        )

    def test_bore_is_an_m8_tapping_hole(self, foot):
        """Smaller than the screw, so the tip cuts its own thread in PETG."""
        assert foot.bore_diameter_mm < DEFAULT_HARDWARE.desk_clamp.bolt_nominal_diameter_mm

    def test_bore_fits_inside_the_pad_recess(self, foot):
        assert foot.bore_diameter_mm < foot.pad_diameter_mm

    def test_foot_spreads_the_load_over_a_bare_tip(self, foot):
        """The reason the part exists at all."""
        assert foot.bearing_area_ratio > 5.0
        assert foot.pad_area_mm2 > 250.0

    def test_foot_fits_the_throat_at_the_thickest_desk(self, foot):
        assert foot.height_mm <= foot.throat_clearance_mm

    def test_foot_too_tall_for_the_throat_is_rejected(self, foot):
        tall = replace(foot, height_mm=foot.throat_clearance_mm + 5.0,
                       bore_depth_mm=foot.bore_depth_mm + 5.0)
        with pytest.raises(PressureFootDesignError) as excinfo:
            tall.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_height_inconsistent_with_its_layers_is_rejected(self, foot):
        inconsistent = replace(foot, height_mm=foot.height_mm + 3.0)
        with pytest.raises(PressureFootDesignError) as excinfo:
            inconsistent.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_pad_wider_than_the_foot_is_rejected(self, foot):
        wide = replace(foot, pad_diameter_mm=foot.diameter_mm)
        with pytest.raises(PressureFootDesignError) as excinfo:
            wide.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_bore_wider_than_the_pad_is_rejected(self, foot):
        bored = replace(foot, bore_diameter_mm=foot.pad_diameter_mm + 1.0)
        with pytest.raises(PressureFootDesignError) as excinfo:
            bored.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    @pytest.mark.parametrize(
        "kwargs", [{"diameter_mm": 0.0}, {"bore_depth_mm": -1.0},
                   {"web_thickness_mm": 0.0}]
    )
    def test_non_positive_dimensions_rejected(self, foot, kwargs):
        with pytest.raises(PressureFootDesignError) as excinfo:
            replace(foot, **kwargs).validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_report_names_the_key_dimensions(self, foot):
        report = foot.report()
        for heading in ("Screw bore", "Pad recess", "Contact area"):
            assert heading in report


# =========================================================================
# 7. Knob
# =========================================================================


class TestKnob:
    def test_default_parameters_validate(self, knob):
        assert knob.validate() is DesignStatus.OK

    def test_body_matches_the_clamp_spec(self, knob):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert knob.body_diameter_mm == pytest.approx(clamp.knob_diameter_mm)
        assert knob.body_thickness_mm == pytest.approx(clamp.knob_thickness_mm)

    def test_hex_socket_takes_the_bolt_head(self, knob):
        """Press fit: much tighter than a general printed-pocket clearance."""
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert knob.socket_across_flats_mm > clamp.bolt_head_across_flats_mm
        slack = knob.socket_across_flats_mm - clamp.bolt_head_across_flats_mm
        assert slack < DEFAULT_HARDWARE.print_clearance_mm

    def test_socket_is_deep_enough_for_the_head(self, knob):
        assert knob.socket_depth_mm > DEFAULT_HARDWARE.desk_clamp.bolt_head_height_mm

    def test_socket_does_not_pierce_the_knob(self, knob):
        assert knob.socket_depth_mm < knob.total_height_mm
        assert knob.shank_bore_length_mm > 0.0

    def test_total_height_includes_the_boss(self, knob):
        assert knob.total_height_mm == pytest.approx(
            knob.body_thickness_mm + knob.boss_height_mm
        )

    def test_boss_leaves_a_wall_around_the_bore(self, knob):
        wall = (knob.boss_diameter_mm - knob.bolt_hole_diameter_mm) / 2.0
        assert wall >= knob.min_wall_thickness_mm

    def test_flutes_are_evenly_spaced_on_the_rim(self, knob):
        positions = knob.flute_positions
        assert len(positions) == knob.flute_count
        for x, y in positions:
            assert math.hypot(x, y) == pytest.approx(knob.body_diameter_mm / 2.0)

    def test_flutes_bite_into_the_rim_without_reaching_the_boss(self, knob):
        assert knob.grip_min_radius_mm < knob.body_diameter_mm / 2.0
        assert knob.grip_min_radius_mm > knob.boss_diameter_mm / 2.0

    def test_socket_piercing_the_knob_is_rejected(self, knob):
        pierced = replace(knob, socket_depth_mm=knob.total_height_mm)
        with pytest.raises(KnobDesignError) as excinfo:
            pierced.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_bore_wider_than_the_socket_is_rejected(self, knob):
        bored = replace(
            knob, bolt_hole_diameter_mm=knob.socket_across_flats_mm + 1.0
        )
        with pytest.raises(KnobDesignError) as excinfo:
            bored.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_thin_boss_is_rejected(self, knob):
        thin = replace(knob, boss_diameter_mm=knob.bolt_hole_diameter_mm + 1.0)
        with pytest.raises(KnobDesignError) as excinfo:
            thin.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_flutes_cutting_into_the_socket_are_rejected(self, knob):
        deep = replace(knob, flute_radius_mm=knob.body_diameter_mm / 2.0 - 6.0)
        with pytest.raises(KnobDesignError) as excinfo:
            deep.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    @pytest.mark.parametrize(
        "kwargs",
        [{"press_fit_clearance_mm": -0.1}, {"flute_depth_fraction": 0.0},
         {"flute_depth_fraction": 0.8}],
    )
    def test_invalid_construction_arguments_rejected(self, kwargs):
        with pytest.raises(KnobDesignError) as excinfo:
            KnobParameters.from_geometry(**kwargs)
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_report_names_the_key_dimensions(self, knob):
        report = knob.report()
        for heading in ("Hex socket", "Bearing boss", "Grip flutes"):
            assert heading in report


# =========================================================================
# 8. Shared primitives
# =========================================================================


class TestHexPrism:
    def test_across_flats_is_honoured(self):
        prism = hex_prism(13.5, 7.0)
        box = prism.bounding_box()
        assert box.max.Y - box.min.Y == pytest.approx(13.5, abs=1e-6)

    def test_across_corners_follows_the_hex_relation(self):
        prism = hex_prism(13.5, 7.0)
        box = prism.bounding_box()
        assert box.max.X - box.min.X == pytest.approx(
            13.5 * 2.0 / math.sqrt(3.0), abs=1e-6
        )

    def test_height_is_honoured_and_sits_on_the_origin(self):
        prism = hex_prism(13.5, 7.0)
        box = prism.bounding_box()
        assert box.min.Z == pytest.approx(0.0, abs=1e-6)
        assert box.max.Z == pytest.approx(7.0, abs=1e-6)

    def test_volume_matches_the_closed_form(self):
        """A regular hexagon of across-flats a has area a^2 * sqrt(3) / 2."""
        across_flats, height = 13.5, 7.0
        expected = (across_flats**2) * math.sqrt(3.0) / 2.0 * height
        assert hex_prism(across_flats, height).volume == pytest.approx(expected)

    @pytest.mark.parametrize("args", [(0.0, 7.0), (13.5, 0.0), (-1.0, 7.0)])
    def test_non_positive_dimensions_rejected(self, args):
        with pytest.raises(ValueError, match="must be positive"):
            hex_prism(*args)


class TestRightTrianglePrism:
    def test_legs_are_honoured(self):
        box = right_triangle_prism(5.0, 7.0, 80.0).bounding_box()
        assert box.min.X == pytest.approx(0.0, abs=1e-6)
        assert box.max.X == pytest.approx(5.0, abs=1e-6)
        assert box.min.Z == pytest.approx(0.0, abs=1e-6)
        assert box.max.Z == pytest.approx(7.0, abs=1e-6)

    def test_extrusion_is_centred_on_the_xz_plane(self):
        """Callers position gussets by their corner, not by a centroid."""
        box = right_triangle_prism(5.0, 5.0, 80.0).bounding_box()
        assert box.min.Y == pytest.approx(-40.0, abs=1e-6)
        assert box.max.Y == pytest.approx(40.0, abs=1e-6)

    def test_volume_is_half_the_enclosing_box(self):
        prism = right_triangle_prism(5.0, 7.0, 80.0)
        assert prism.volume == pytest.approx(0.5 * 5.0 * 7.0 * 80.0)

    @pytest.mark.parametrize(
        "args", [(0.0, 5.0, 80.0), (5.0, 0.0, 80.0), (5.0, 5.0, -1.0)]
    )
    def test_non_positive_dimensions_rejected(self, args):
        with pytest.raises(ValueError, match="must be positive"):
            right_triangle_prism(*args)


# =========================================================================
# 9. Solid construction
# =========================================================================


class TestSolidConstruction:
    @pytest.mark.parametrize("part_name", sorted(PART_BUILDERS))
    def test_part_builds_as_a_single_solid(self, part_name):
        part = PART_BUILDERS[part_name]()
        assert len(part.solids()) == 1, f"{part_name} must not be fragmented"
        assert part.volume > 0.0

    def test_pedestal_is_smaller_than_its_bounding_box(self, pedestal):
        """Sanity check that the internal pockets were actually subtracted."""
        box = pedestal.bounding_box()
        envelope = (
            (box.max.X - box.min.X) * (box.max.Y - box.min.Y) * (box.max.Z - box.min.Z)
        )
        assert pedestal.volume < envelope

    def test_pedestal_bounding_box_spans_the_whole_u(self, pedestal, params):
        box = pedestal.bounding_box()
        assert box.min.X == pytest.approx(params.spine_outer_x_mm, abs=1e-6)
        assert box.max.X == pytest.approx(params.top_arm_inner_x_mm, abs=1e-6)
        assert box.max.Y - box.min.Y == pytest.approx(
            params.clamp_width_mm, abs=1e-6
        )
        assert box.max.Z - box.min.Z == pytest.approx(
            params.overall_height_mm, abs=1e-6
        )

    def test_pedestal_straddles_the_desk_plane(self, pedestal, params):
        """
        The origin is the desk's top surface, not the print bed.

        Keeping that datum is what lets the clamp, ArmGeometry's base height
        and the assembly preview share one coordinate system without a
        conversion step; slicers drop the part to the bed on import.
        """
        box = pedestal.bounding_box()
        assert box.min.Z == pytest.approx(params.bottom_arm_bottom_z_mm, abs=1e-6)
        assert box.max.Z == pytest.approx(params.turret_top_z_mm, abs=1e-6)
        assert box.min.Z < 0.0 < box.max.Z

    def test_pedestal_uses_the_geometry_singletons_by_default(self):
        default_built = build_pedestal()
        assert default_built.bounding_box().max.Z == pytest.approx(
            DEFAULT_HARDWARE.pedestal_height_mm(DEFAULT_ARM), abs=1e-6
        )

    def test_pressure_foot_bounding_box_matches_its_parameters(self, foot):
        box = build_pressure_foot(foot).bounding_box()
        assert box.max.Z - box.min.Z == pytest.approx(foot.height_mm, abs=1e-6)
        assert box.min.Z == pytest.approx(0.0, abs=1e-6)
        assert box.max.X - box.min.X == pytest.approx(foot.diameter_mm, rel=1e-3)

    def test_pressure_foot_is_hollowed_by_its_bore_and_recess(self, foot):
        solid_disc = math.pi * (foot.diameter_mm / 2.0) ** 2 * foot.height_mm
        assert build_pressure_foot(foot).volume < solid_disc

    def test_knob_bounding_box_matches_its_parameters(self):
        knob = KnobParameters.from_geometry()
        box = build_knob(knob).bounding_box()
        assert box.max.Z - box.min.Z == pytest.approx(knob.total_height_mm, abs=1e-6)
        # Flutes bite into the rim, so the widest span is under the nominal
        # diameter but never beyond it.
        span = box.max.X - box.min.X
        assert span <= knob.body_diameter_mm + 1e-6
        assert span > 2 * knob.grip_min_radius_mm

    def test_knob_flutes_actually_remove_material(self):
        """A fluted knob must be lighter than the plain disc it came from."""
        knob = KnobParameters.from_geometry()
        plain_disc = (
            math.pi * (knob.body_diameter_mm / 2.0) ** 2 * knob.body_thickness_mm
        )
        assert build_knob(knob).volume < plain_disc


# =========================================================================
# 10. STL export and mesh integrity
# =========================================================================


class TestStlExport:
    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_export_creates_the_file(self, part_name, stl_paths):
        path = stl_paths[part_name]
        assert path.exists()
        assert path.stat().st_size > 0

    def test_export_creates_missing_parent_directories(self, tmp_path):
        nested = tmp_path / "does" / "not" / "exist" / "pedestal.stl"
        assert export_pedestal(nested).exists()

    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_file_is_a_well_formed_binary_stl(self, part_name, stl_paths):
        """The declared triangle count must match the actual file length."""
        raw = stl_paths[part_name].read_bytes()
        triangle_count = struct.unpack("<I", raw[80:84])[0]
        assert triangle_count > 0
        assert len(raw) == 84 + 50 * triangle_count

    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_mesh_has_triangles(self, part_name, meshes):
        mesh = meshes[part_name]
        assert len(mesh.faces) > 0
        assert len(mesh.vertices) > 0

    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_mesh_is_watertight(self, part_name, meshes):
        """
        Every undirected edge is used by exactly two faces.

        This is the definition of a closed surface: an edge used once is a
        boundary (a hole in the mesh), and an edge used three or more times is
        a non-manifold junction. Either makes the solid unprintable.
        """
        offenders = {
            edge: count
            for edge, count in meshes[part_name].edge_use_counts().items()
            if count != 2
        }
        assert not offenders, (
            f"{part_name}: {len(offenders)} non-manifold or boundary edge(s); "
            f"a closed mesh uses every edge exactly twice."
        )

    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_mesh_normals_are_consistently_oriented(self, part_name, meshes):
        """
        Every directed edge is traversed exactly once.

        Two adjacent faces of a consistently-oriented surface walk their
        shared edge in opposite directions. A directed edge seen twice means
        one of the two faces is wound backwards, which flips its normal and
        confuses slicers about which side is solid.
        """
        offenders = {
            edge: count
            for edge, count in meshes[part_name].directed_edge_counts().items()
            if count != 1
        }
        assert not offenders, (
            f"{part_name}: {len(offenders)} inconsistently-wound edge(s)."
        )

    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_mesh_volume_is_nonzero_and_positive(self, part_name, meshes):
        volume = meshes[part_name].signed_volume_mm3()
        assert volume > 0.0, (
            f"{part_name}: non-positive enclosed volume means the mesh either "
            "encloses nothing or has inward-facing normals"
        )

    @pytest.mark.parametrize("part_name", sorted(PART_EXPORTERS))
    def test_mesh_volume_agrees_with_the_kernel(self, part_name, meshes):
        """
        Tessellated volume must track the B-rep volume.

        A flat-faceted mesh under-fills curved surfaces, so the mesh volume is
        slightly *lower* than the exact solid. A 1% band catches a genuinely
        wrong export while tolerating that expected chord error.
        """
        kernel_volume = PART_BUILDERS[part_name]().volume
        assert meshes[part_name].signed_volume_mm3() == pytest.approx(
            kernel_volume, rel=0.01
        )

    def test_pedestal_mesh_bounding_box_matches_the_design(self, meshes, params):
        low, high = meshes["base_pedestal"].bounding_box()
        assert high[2] - low[2] == pytest.approx(params.overall_height_mm, abs=1e-3)
        assert low[0] == pytest.approx(params.spine_outer_x_mm, abs=1e-3)
        assert high[0] == pytest.approx(params.top_arm_inner_x_mm, abs=1e-3)

    def test_pedestal_mesh_keeps_the_desk_plane_datum(self, meshes, params):
        low, high = meshes["base_pedestal"].bounding_box()
        assert low[2] == pytest.approx(params.bottom_arm_bottom_z_mm, abs=1e-3)
        assert high[2] == pytest.approx(params.turret_top_z_mm, abs=1e-3)


# =========================================================================
# 11. Assembly preview
# =========================================================================


@pytest.fixture(scope="module")
def assembly():
    """The whole scene placed on a nominal desk."""
    return build_assembly()


class TestAssemblyPreview:
    def test_assembly_preview_generates_stl(self, assembly, tmp_path):
        """Exists, is non-empty, and every shell in it is closed."""
        path = export_assembly_stl(tmp_path / "assembly_preview.stl", assembly)
        assert path.exists()
        assert path.stat().st_size > 0

        mesh = Mesh.from_binary_stl(path)
        offenders = {
            edge: count
            for edge, count in mesh.edge_use_counts().items()
            if count != 2
        }
        assert not offenders, (
            f"{len(offenders)} non-manifold or boundary edge(s) in the assembly"
        )
        assert mesh.signed_volume_mm3() > 0.0

    def test_assembly_stl_is_a_well_formed_binary_stl(self, assembly, tmp_path):
        path = export_assembly_stl(tmp_path / "assembly.stl", assembly)
        raw = path.read_bytes()
        triangle_count = struct.unpack("<I", raw[80:84])[0]
        assert triangle_count > 0
        assert len(raw) == 84 + 50 * triangle_count

    def test_assembly_placement_matches_arm_geometry(self, assembly):
        """
        The yaw axis must land exactly where ArmGeometry says the base is.

        Checked through the placed solid's bounding box rather than by reading
        back a stored number: the clamp is modelled with its own +X pointing
        inward and is rotated 90 degrees into the desk frame, so this catches a
        wrong rotation or a missed translation, which echoing a field would not.
        """
        placed = assembly.by_name("base_pedestal").solid
        params = PedestalParameters.from_geometry()
        box = placed.bounding_box()
        base_x = DEFAULT_ARM.base_x_on_desk_mm
        base_y = DEFAULT_ARM.base_y_on_desk_mm

        # Clamp +X becomes desk +Y, so its width now spans desk X about the base.
        assert box.min.X == pytest.approx(
            base_x - params.clamp_width_mm / 2.0, abs=1e-6
        )
        assert box.max.X == pytest.approx(
            base_x + params.clamp_width_mm / 2.0, abs=1e-6
        )
        # ... and its profile spans desk Y from the spine to the arm's inner end.
        assert box.min.Y == pytest.approx(
            base_y + params.spine_outer_x_mm, abs=1e-6
        )
        assert box.max.Y == pytest.approx(
            base_y + params.top_arm_inner_x_mm, abs=1e-6
        )

    def test_desk_edge_lands_on_the_clamps_seating_plane(self, assembly):
        """
        The desk's edge is at y = 0, and the clamp seats it exactly there.

        This is the check that keeps ArmGeometry.base_y_on_desk_mm and
        DeskClampSpec.servo_shaft_offset_from_edge_mm honest against each
        other: if either drifts, the desk edge stops landing on the gussets.
        """
        params = PedestalParameters.from_geometry()
        seat_y = DEFAULT_ARM.base_y_on_desk_mm + params.desk_seat_x_mm
        assert seat_y == pytest.approx(0.0, abs=1e-9)

    def test_clamp_spine_hangs_off_the_desk_edge(self, assembly):
        """The spine wraps the edge, so part of the clamp is beyond the desk."""
        box = assembly.by_name("base_pedestal").solid.bounding_box()
        assert box.min.Y < 0.0
        assert box.max.Y > 0.0

    def test_pedestal_does_not_intersect_placeholder_arm_links(self, assembly):
        """
        At the zero pose the links run horizontally at the shoulder height, and
        the turret must stay clear beneath them. A collision here would mean
        the base height budget is wrong.
        """
        pedestal = assembly.by_name("base_pedestal").solid
        for name in ("L1 upper arm", "L2 forearm", "L3 wrist-to-TCP"):
            link = assembly.by_name(name).solid
            overlap = pedestal.intersect(link)
            volume = 0.0 if overlap is None else float(overlap.volume)
            assert volume == pytest.approx(0.0, abs=1e-6), (
                f"{name} overlaps the pedestal by {volume:.3f} mm3"
            )

    def test_base_stack_placeholders_fill_the_gap_to_the_shoulder(self, assembly):
        """
        The D.1c preview showed L1 apparently floating above the turret. It was
        not a base-frame error: the 30.5 mm between the turret's top face and
        the shoulder pivot is the bearing, the yaw turntable and the shoulder
        bracket, none of which existed to be drawn. With placeholders in the
        scene the stack is continuous, and this test keeps it that way.
        """
        params = PedestalParameters.from_geometry()
        turntable = assembly.by_name("yaw turntable (D.2)").solid.bounding_box()
        bracket = assembly.by_name("shoulder bracket (D.3)").solid.bounding_box()

        # Turntable rides the bearing's upper face, not the turret's.
        assert turntable.min.Z == pytest.approx(params.bearing_top_z_mm, abs=1e-6)
        # Bracket stands on the turntable, with no gap between them.
        assert bracket.min.Z == pytest.approx(turntable.max.Z, abs=1e-6)
        # And its top lands exactly on the shoulder pivot.
        assert bracket.max.Z == pytest.approx(DEFAULT_ARM.base_height_mm, abs=1e-6)

    def test_nothing_floats_between_the_turret_and_the_shoulder(self, assembly):
        """Every interface in the base stack touches the next, to the micron."""
        params = PedestalParameters.from_geometry()
        boundaries = [
            params.turret_top_z_mm,
            params.bearing_top_z_mm,
            assembly.by_name("yaw turntable (D.2)").solid.bounding_box().max.Z,
            assembly.by_name("shoulder bracket (D.3)").solid.bounding_box().max.Z,
        ]
        assert boundaries == sorted(boundaries)
        assert boundaries[-1] == pytest.approx(assembly.shoulder_pivot_z_mm, abs=1e-6)

    def test_stack_placeholders_sit_on_the_turret_not_over_it(self, assembly):
        """
        Sized from the turret they stand on. They stand in for parts D.2 and
        D.3 have yet to design, so inventing dimensions would imply decisions
        nobody has made.
        """
        params = PedestalParameters.from_geometry()
        turret_span = params.turret_x_max_mm - params.turret_x_min_mm
        turntable = assembly.by_name("yaw turntable (D.2)").solid.bounding_box()
        assert turntable.max.X - turntable.min.X <= turret_span + 1e-6

    def test_assembly_reports_the_shaft_output_height(self, assembly):
        params = PedestalParameters.from_geometry()
        assert assembly.servo_shaft_output_z_mm == pytest.approx(
            params.servo_shaft_output_z_mm
        )
        assert assembly.shoulder_pivot_z_mm == pytest.approx(
            DEFAULT_ARM.base_height_mm
        )

    def test_arm_links_clear_the_turret_vertically(self, assembly):
        """The margin, not just the absence of contact."""
        pedestal_top = assembly.by_name("base_pedestal").solid.bounding_box().max.Z
        link_bottom = assembly.by_name("L1 upper arm").solid.bounding_box().min.Z
        assert link_bottom > pedestal_top
        assert link_bottom - pedestal_top > 5.0

    def test_arm_links_stay_above_the_desk(self, assembly):
        """Nothing may pass through the desk surface at the zero pose."""
        for name in ("L1 upper arm", "L2 forearm", "L3 wrist-to-TCP", "TCP marker"):
            assert assembly.by_name(name).solid.bounding_box().min.Z > 0.0

    def test_links_start_at_the_shoulder_and_run_inward(self, assembly):
        """Zero pose is horizontal along the base frame's +X, i.e. desk +Y."""
        base_y = DEFAULT_ARM.base_y_on_desk_mm
        link1 = assembly.by_name("L1 upper arm").solid.bounding_box()
        assert link1.min.Y == pytest.approx(base_y, abs=1e-6)
        assert link1.max.Y == pytest.approx(
            base_y + DEFAULT_ARM.l1_upper_arm_mm, abs=1e-6
        )

    def test_tcp_marker_sits_at_full_reach(self, assembly):
        """FK's zero pose puts the TCP at total_reach along +X from the base."""
        box = assembly.by_name("TCP marker").solid.bounding_box()
        centre_y = (box.min.Y + box.max.Y) / 2.0
        assert centre_y == pytest.approx(
            DEFAULT_ARM.base_y_on_desk_mm + DEFAULT_ARM.total_reach_mm, abs=1e-6
        )
        centre_z = (box.min.Z + box.max.Z) / 2.0
        assert centre_z == pytest.approx(DEFAULT_ARM.base_height_mm, abs=1e-6)

    def test_pressure_foot_bears_on_the_desk_underside(self, assembly):
        """Its contact face must reach the desk, or the clamp grips nothing."""
        box = assembly.by_name("pressure_foot").solid.bounding_box()
        assert box.max.Z == pytest.approx(-assembly.desk_thickness_mm, abs=1e-6)

    def test_knob_hangs_clear_below_the_bottom_arm(self, assembly):
        """
        The knob is a handle, not a bearing surface.

        It has to stay clear of the bottom arm: if it touched, further
        tightening would jam against the clamp instead of loading the desk.
        """
        params = PedestalParameters.from_geometry()
        knob_top = assembly.by_name("knob").solid.bounding_box().max.Z
        assert knob_top < params.bottom_arm_bottom_z_mm
        assert assembly.knob_drop_below_arm_mm > 0.0

    def test_desk_slab_matches_arm_geometry(self, assembly):
        box = assembly.by_name("desk").solid.bounding_box()
        assert box.max.X - box.min.X == pytest.approx(DEFAULT_ARM.desk_width_mm)
        assert box.max.Y - box.min.Y == pytest.approx(DEFAULT_ARM.desk_depth_mm)
        assert box.max.Z == pytest.approx(0.0, abs=1e-6)

    def test_scenery_is_excluded_from_the_print_estimate(self, assembly):
        """The desk and the arm placeholders are not parts we make."""
        printed = {part.name for part in assembly.printed_parts}
        # The D.2/D.3 placeholders are excluded too: they will be printed, but
        # their real volume is unknown, so counting them would invent a figure.
        assert printed == {"base_pedestal", "pressure_foot", "knob"}
        assert assembly.estimated_print_mass_g > 0.0

    @pytest.mark.parametrize("thickness", [15.0, 25.0, 35.0])
    def test_every_supported_desk_thickness_assembles(self, thickness):
        """The clamp has to work across its whole stated range, not just one desk."""
        scene = build_assembly(desk_thickness_mm=thickness)
        foot = scene.by_name("pressure_foot").solid.bounding_box()
        assert foot.max.Z == pytest.approx(-thickness, abs=1e-6)
        assert scene.knob_drop_below_arm_mm > 0.0

    @pytest.mark.parametrize("thickness", [5.0, 50.0])
    def test_unsupported_desk_thickness_rejected(self, thickness):
        with pytest.raises(ValueError, match="outside the clamp"):
            build_assembly(desk_thickness_mm=thickness)

    def test_render_writes_a_png(self, assembly, tmp_path):
        path = render_assembly_png(tmp_path / "preview.png", assembly)
        assert path.exists()
        assert path.stat().st_size > 0
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_summary_names_the_scene_numbers(self, assembly):
        summary = assembly.summary()
        for heading in ("Desk", "Yaw axis on desk", "Clamp footprint",
                        "Est. filament mass"):
            assert heading in summary
