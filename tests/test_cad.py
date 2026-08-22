"""
Test suite for the parametric CAD (Sessions D.1 and D.1b).

Test areas
----------
1. Hardware specifications in src.geometry (validation, derived values,
   the UNVERIFIED-field bookkeeping)
2. DeskClampSpec -- fastener standards, derived stack-up, clamp physics
3. Pedestal parameter derivation -- every dimension traceable to geometry.py
4. Design rule checks -- each violation raises with the right DesignStatus
5. Lower jaw and knob parameter derivation and design rule checks
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
from cad._primitives import hex_prism  # noqa: E402
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
from cad.desk_clamp_lower_jaw import (  # noqa: E402
    LowerJawDesignError,
    LowerJawParameters,
    build_lower_jaw,
    export_lower_jaw,
)

#: Every printable part, so the mesh-integrity checks cover all of them.
PART_EXPORTERS = {
    "base_pedestal": export_pedestal,
    "desk_clamp_lower_jaw": export_lower_jaw,
    "desk_clamp_knob": export_knob,
}
PART_BUILDERS = {
    "base_pedestal": build_pedestal,
    "desk_clamp_lower_jaw": build_lower_jaw,
    "desk_clamp_knob": build_knob,
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
def jaw() -> LowerJawParameters:
    """Default lower-jaw parameters, derived from the geometry singletons."""
    return LowerJawParameters.from_geometry()


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
        # DIN 934 tabulates e min = 14.38 mm for M8.
        assert clamp.nut_across_corners_mm == pytest.approx(15.01, abs=0.02)

    def test_max_nut_thickness_below_nominal_rejected(self):
        with pytest.raises(ValueError, match="must not be less than"):
            DeskClampSpec(nut_thickness_nominal_mm=7.0, nut_thickness_max_mm=6.8)

    def test_clamp_throat_opens_wider_than_desk_thickness(self):
        """
        Replaces the D.1 bolt-circle test.

        The throat must exceed the thickest supported desk, or the clamp
        cannot be slid on and off without fully unthreading the nut.
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

    def test_jaw_totals_add_their_recesses_to_the_structural_thickness(self):
        """
        At 10 mm total, a 2 mm pad recess plus a 6.8 mm nut pocket would leave
        1.2 mm of web, so recesses are additional rather than subtractive.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.upper_jaw_total_thickness_mm == pytest.approx(12.0)
        assert clamp.lower_jaw_total_thickness_mm == pytest.approx(18.8)
        web = (
            clamp.lower_jaw_total_thickness_mm
            - clamp.nut_pocket_depth_mm
            - clamp.pad_recess_depth_mm
        )
        assert web == pytest.approx(clamp.jaw_thickness_mm)

    def test_pad_thicker_than_its_recess_rejected(self):
        with pytest.raises(ValueError, match="stand proud"):
            DeskClampSpec(pad_recess_depth_mm=2.0, pad_thickness_mm=3.0)

    def test_specified_bolt_is_long_enough_for_the_stack(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bolt_length_mm >= clamp.required_bolt_length_mm
        assert clamp.bolt_length_mm == pytest.approx(90.0)

    def test_required_bolt_length_tracks_the_throat(self):
        """Opening the throat further must demand a longer screw."""
        wide = DeskClampSpec(throat_max_opening_mm=60.0, bolt_length_mm=110.0)
        assert wide.required_bolt_length_mm > (
            DEFAULT_HARDWARE.desk_clamp.required_bolt_length_mm
        )

    def test_boss_not_exceeding_bolt_hole_rejected(self):
        with pytest.raises(ValueError, match="must exceed the bolt"):
            DeskClampSpec(knob_boss_diameter_mm=9.0)

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

    def test_pad_friction_uses_the_stated_coefficient(self):
        clamp = DeskClampSpec(pad_friction_coefficient=0.4)
        assert clamp.pad_friction_force_n(1000.0) == pytest.approx(400.0)

    def test_resisting_moment_requires_a_positive_lever(self):
        with pytest.raises(ValueError, match="lever_arm_mm must be positive"):
            DEFAULT_HARDWARE.desk_clamp.clamp_resisting_moment_nm(5.0, 0.0)

    def test_jaw_allowable_preload_falls_with_overhang(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.jaw_allowable_preload_n(30.0) < clamp.jaw_allowable_preload_n(10.0)


# =========================================================================
# 3. Clamp capacity against the arm's real loads
# =========================================================================


class TestClampCapacity:
    def test_clamp_grip_force_exceeds_arm_tipping_torque(self, params):
        """
        The headline requirement: a hand-tightened clamp holds the arm down.

        Grip force comes from the M8 thread pitch, thread and collar friction,
        and 5 N.m of hand torque. It resists the arm's worst-case tipping
        moment across the lever from the pad's inboard edge (the pivot the
        assembly would rotate about) to the clamp screw.
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
        # Not merely exceeding it -- comfortably so.
        assert resisting_nm > 5.0 * tipping_nm

    def test_required_preload_is_within_the_jaws_structural_limit(self, params):
        """
        The real binding constraint, and the reason for the torque warning.

        The preload actually needed to resist tipping must sit inside what the
        printed jaw can carry in bending. It does, with wide margin -- but the
        allowable is far below the 5 N.m the test above assumes, which is why
        cad/README.md says hand-tight only.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        required_n = DEFAULT_ARM.tipping_moment_nm() / (
            params.tipping_lever_arm_mm / 1000.0
        )
        allowable_n = clamp.jaw_allowable_preload_n(params.jaw_overhang_mm)

        assert required_n < allowable_n
        assert allowable_n / required_n > 5.0

    def test_max_safe_torque_is_documented_and_below_five_newton_metres(self, params):
        """
        Pins the finding that a 5 N.m tighten would over-stress the jaw.

        If a future change makes the jaw strong enough for 5 N.m, this test
        fails and the README warning should be revisited.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        max_torque = clamp.max_tightening_torque_nm(params.jaw_overhang_mm)
        assert 0.0 < max_torque < 5.0
        # Comfortably above what is actually needed to hold the arm.
        needed_torque = clamp.preload_to_torque_nm(
            DEFAULT_ARM.tipping_moment_nm() / (params.tipping_lever_arm_mm / 1000.0)
        )
        assert max_torque > 5.0 * needed_torque

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
# 4. Pedestal parameter derivation
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

    def test_pedestal_internals_survived_the_clamp_redesign(self, params):
        """
        D.1b changed only the mounting. The servo/bearing geometry is good and
        must not have drifted while the flange was being replaced.
        """
        assert params.total_height_mm == pytest.approx(70.0)
        assert 2 * params.body_radius_mm == pytest.approx(85.27, abs=0.01)
        assert params.cavity_body_length_mm == pytest.approx(40.5)
        assert params.cavity_ear_length_mm == pytest.approx(54.5)
        assert params.ear_shelf_z_mm == pytest.approx(49.5)
        assert params.cavity_top_z_mm == pytest.approx(59.5)
        assert params.bearing_seat_diameter_mm == pytest.approx(21.9)

    def test_jaw_thickness_comes_from_the_clamp_spec(self, params):
        assert params.jaw_thickness_mm == pytest.approx(
            DEFAULT_HARDWARE.desk_clamp.upper_jaw_total_thickness_mm
        )

    def test_pad_recess_matches_the_clamp_contact_area(self, params):
        length, width = DEFAULT_HARDWARE.desk_clamp.upper_jaw_contact_mm
        assert params.pad_length_mm == pytest.approx(length)
        assert params.pad_width_mm == pytest.approx(width)

    def test_pad_starts_clear_of_the_servo_cavity(self, params):
        """The recess must not open into the pocket the servo slides through."""
        cavity_outer_x = (
            params.cavity_offset_x_mm + params.cavity_ear_length_mm / 2.0
        )
        assert params.pad_inner_x_mm - cavity_outer_x >= params.min_wall_thickness_mm

    def test_clamp_screw_hole_is_outboard_of_the_pad(self, params):
        """The desk edge sits between them, so the order matters."""
        hole_inner_x = params.bolt_axis_x_mm - params.bolt_hole_diameter_mm / 2.0
        assert hole_inner_x > params.pad_outer_x_mm
        assert hole_inner_x - params.pad_outer_x_mm == pytest.approx(
            params.desk_edge_window_mm
        )

    def test_clamp_screw_hole_matches_the_clamp_spec(self, params):
        assert params.bolt_hole_diameter_mm == pytest.approx(
            DEFAULT_HARDWARE.desk_clamp.bolt_clearance_hole_diameter_mm
        )

    def test_jaw_is_wide_enough_for_the_knob(self, params):
        assert params.jaw_width_mm >= DEFAULT_HARDWARE.desk_clamp.knob_diameter_mm

    def test_knob_clears_the_pedestal_body(self, params):
        assert params.bolt_axis_x_mm - params.knob_diameter_mm / 2.0 >= (
            params.body_radius_mm
        )

    def test_jaw_overhang_is_the_worst_case_desk_edge_position(self, params):
        assert params.jaw_overhang_mm == pytest.approx(
            params.bolt_axis_x_mm - params.pad_outer_x_mm
        )
        assert params.jaw_overhang_mm > 0.0

    def test_tipping_lever_is_measured_from_the_pad_inner_edge(self, params):
        assert params.tipping_lever_arm_mm == pytest.approx(
            params.bolt_axis_x_mm - params.pad_inner_x_mm
        )
        assert params.tipping_lever_arm_mm > params.jaw_overhang_mm

    def test_wider_desk_edge_window_lengthens_the_jaw(self):
        narrow = PedestalParameters.from_geometry(desk_edge_window_mm=5.0)
        wide = PedestalParameters.from_geometry(desk_edge_window_mm=20.0)
        assert wide.jaw_reach_mm > narrow.jaw_reach_mm
        assert wide.jaw_overhang_mm > narrow.jaw_overhang_mm

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
        assert bigger.validate() is DesignStatus.OK

    def test_report_names_the_key_dimensions(self, params):
        report = params.report()
        for heading in (
            "Total height", "Body outer diameter", "Bearing seat",
            "Clamp upper jaw", "MAX safe knob torque",
        ):
            assert heading in report


# =========================================================================
# 5. Pedestal design rule checks
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

    def test_short_clamp_screw_is_rejected(self):
        """
        The screw has to span knob, jaw, throat, jaw and nut.

        An M8 x 80 -- the length originally specified -- falls 8 mm short at
        the full 45 mm throat, so the check must catch it rather than ship a
        clamp that cannot reach its nut.
        """
        short = HardwareSpec(desk_clamp=DeskClampSpec(bolt_length_mm=80.0))
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(hardware=short)
        assert excinfo.value.status is DesignStatus.FASTENER_TOO_SHORT
        assert "80.0" in str(excinfo.value)

    def test_thin_wall_is_rejected(self, params):
        pinched = replace(params, body_radius_mm=params.body_radius_mm - 4.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            pinched.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_pad_recess_deeper_than_the_jaw_is_rejected(self, params):
        cut_through = replace(params, pad_recess_depth_mm=params.jaw_thickness_mm)
        with pytest.raises(PedestalDesignError) as excinfo:
            cut_through.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_jaw_thicker_than_the_pedestal_is_rejected(self, params):
        absurd = replace(params, jaw_thickness_mm=params.total_height_mm + 1.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            absurd.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_pad_overlapping_the_servo_cavity_is_rejected(self, params):
        overlapping = replace(params, pad_inner_x_mm=0.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            overlapping.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_screw_hole_running_off_the_jaw_tip_is_rejected(self, params):
        stubby = replace(params, jaw_reach_mm=params.bolt_axis_x_mm)
        with pytest.raises(PedestalDesignError) as excinfo:
            stubby.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_screw_hole_too_close_to_the_jaw_edge_is_rejected(self, params):
        narrow = replace(params, jaw_width_mm=params.bolt_hole_diameter_mm + 1.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            narrow.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_knob_fouling_the_pedestal_body_is_rejected(self, params):
        huge_knob = replace(params, knob_diameter_mm=200.0)
        with pytest.raises(PedestalDesignError) as excinfo:
            huge_knob.validate()
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
            {"desk_edge_window_mm": 0.0},
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
            replace(params, body_radius_mm=params.body_radius_mm - 4.0).validate()

    def test_build_rejects_invalid_parameters(self, params):
        pinched = replace(params, body_radius_mm=params.body_radius_mm - 4.0)
        with pytest.raises(PedestalDesignError):
            build_pedestal(pinched)


# =========================================================================
# 6. Lower jaw
# =========================================================================


class TestLowerJaw:
    def test_default_parameters_validate(self, jaw):
        assert jaw.validate() is DesignStatus.OK

    def test_nut_pocket_fits_a_standard_m8_nut(self, jaw):
        """
        Across flats must clear the 13.00 mm nut with print clearance, and the
        pocket must be deep enough for the thickest DIN EN ISO 4032 nut.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        expected_af = clamp.nut_across_flats_mm + 2 * DEFAULT_HARDWARE.print_clearance_mm
        assert jaw.nut_pocket_across_flats_mm == pytest.approx(expected_af)
        assert jaw.nut_pocket_across_flats_mm > clamp.nut_across_flats_mm
        assert jaw.nut_pocket_depth_mm >= clamp.nut_thickness_max_mm

    def test_nut_pocket_across_corners_is_the_hex_relation(self, jaw):
        assert jaw.nut_pocket_across_corners_mm == pytest.approx(
            jaw.nut_pocket_across_flats_mm * 2.0 / math.sqrt(3.0)
        )

    def test_bolt_hole_is_smaller_than_the_pocket(self, jaw):
        """Otherwise the nut has no shoulder to bear against."""
        assert jaw.bolt_hole_diameter_mm < jaw.nut_pocket_across_flats_mm

    def test_load_bearing_web_meets_the_structural_thickness(self, jaw):
        assert jaw.nut_bearing_thickness_mm == pytest.approx(
            jaw.structural_thickness_mm
        )

    def test_total_thickness_comes_from_the_clamp_spec(self, jaw):
        assert jaw.total_thickness_mm == pytest.approx(
            DEFAULT_HARDWARE.desk_clamp.lower_jaw_total_thickness_mm
        )

    def test_pad_matches_the_lower_contact_area(self, jaw):
        length, width = DEFAULT_HARDWARE.desk_clamp.lower_jaw_contact_mm
        assert jaw.pad_length_mm == pytest.approx(length)
        assert jaw.pad_width_mm == pytest.approx(width)

    def test_pad_stands_back_far_enough_to_sit_under_the_desk(self, jaw):
        """
        The pad must clear the widest legal desk-edge offset, or it would hang
        in free air when the user sites the arm at one end of the window.
        """
        pedestal = PedestalParameters.from_geometry()
        assert jaw.pad_gap_from_bolt_mm >= pedestal.jaw_overhang_mm

    def test_pad_clears_the_nut_pocket(self, jaw):
        assert jaw.pad_gap_from_bolt_mm > jaw.nut_pocket_across_corners_mm / 2.0

    def test_plate_length_is_the_sum_of_its_arms(self, jaw):
        assert jaw.plate_length_mm == pytest.approx(
            jaw.inboard_length_mm + jaw.outboard_length_mm
        )

    def test_thin_web_is_rejected(self, jaw):
        squashed = replace(jaw, total_thickness_mm=jaw.total_thickness_mm - 5.0)
        with pytest.raises(LowerJawDesignError) as excinfo:
            squashed.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_pocket_and_recess_meeting_is_rejected(self, jaw):
        thin = replace(
            jaw,
            total_thickness_mm=jaw.nut_pocket_depth_mm + jaw.pad_recess_depth_mm,
        )
        with pytest.raises(LowerJawDesignError) as excinfo:
            thin.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_bolt_hole_wider_than_the_pocket_is_rejected(self, jaw):
        bored = replace(
            jaw, bolt_hole_diameter_mm=jaw.nut_pocket_across_flats_mm + 1.0
        )
        with pytest.raises(LowerJawDesignError) as excinfo:
            bored.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_narrow_plate_is_rejected(self, jaw):
        narrow = replace(jaw, plate_width_mm=jaw.nut_pocket_across_corners_mm)
        with pytest.raises(LowerJawDesignError) as excinfo:
            narrow.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_short_tail_is_rejected(self, jaw):
        clipped = replace(jaw, outboard_length_mm=1.0)
        with pytest.raises(LowerJawDesignError) as excinfo:
            clipped.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_non_positive_window_rejected(self):
        with pytest.raises(LowerJawDesignError) as excinfo:
            LowerJawParameters.from_geometry(desk_edge_window_mm=0.0)
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_report_names_the_key_dimensions(self, jaw):
        report = jaw.report()
        for heading in ("Nut pocket", "Pad recess", "Load-bearing web"):
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


# =========================================================================
# 9. Solid construction
# =========================================================================


class TestSolidConstruction:
    @pytest.mark.parametrize("part_name", sorted(PART_BUILDERS))
    def test_part_builds_as_a_single_solid(self, part_name):
        part = PART_BUILDERS[part_name]()
        assert len(part.solids()) == 1, f"{part_name} must not be fragmented"
        assert part.volume > 0.0

    def test_pedestal_is_smaller_than_its_bounding_box(self, pedestal, params):
        """Sanity check that the internal pockets were actually subtracted."""
        box = pedestal.bounding_box()
        envelope = (
            (box.max.X - box.min.X) * (box.max.Y - box.min.Y) * (box.max.Z - box.min.Z)
        )
        assert pedestal.volume < envelope

    def test_pedestal_bounding_box_spans_body_plus_jaw(self, pedestal, params):
        box = pedestal.bounding_box()
        assert box.max.Z - box.min.Z == pytest.approx(params.total_height_mm, abs=1e-6)
        assert box.min.X == pytest.approx(-params.body_radius_mm, abs=1e-6)
        assert box.max.X == pytest.approx(params.jaw_reach_mm, abs=1e-6)

    def test_pedestal_sits_on_the_z_origin(self, pedestal):
        """The underside must be at z = 0 so it drops onto a print bed."""
        assert pedestal.bounding_box().min.Z == pytest.approx(0.0, abs=1e-6)

    def test_pedestal_uses_the_geometry_singletons_by_default(self):
        default_built = build_pedestal()
        assert default_built.bounding_box().max.Z == pytest.approx(
            DEFAULT_HARDWARE.pedestal_height_mm(DEFAULT_ARM), abs=1e-6
        )

    def test_lower_jaw_bounding_box_matches_its_parameters(self):
        jaw = LowerJawParameters.from_geometry()
        box = build_lower_jaw(jaw).bounding_box()
        assert box.max.Z - box.min.Z == pytest.approx(jaw.total_thickness_mm, abs=1e-6)
        assert box.max.Y - box.min.Y == pytest.approx(jaw.plate_width_mm, abs=1e-6)
        assert box.max.X - box.min.X == pytest.approx(jaw.plate_length_mm, abs=1e-6)

    def test_lower_jaw_extends_inboard_from_the_bolt_axis(self):
        """Origin is the bolt axis; the plate reaches under the desk in -X."""
        jaw = LowerJawParameters.from_geometry()
        box = build_lower_jaw(jaw).bounding_box()
        assert box.min.X == pytest.approx(-jaw.inboard_length_mm, abs=1e-6)
        assert box.max.X == pytest.approx(jaw.outboard_length_mm, abs=1e-6)

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
        assert high[2] - low[2] == pytest.approx(params.total_height_mm, abs=1e-3)
        assert high[0] == pytest.approx(params.jaw_reach_mm, abs=1e-3)

    def test_pedestal_mesh_sits_on_the_z_origin(self, meshes):
        assert meshes["base_pedestal"].bounding_box()[0][2] == pytest.approx(
            0.0, abs=1e-6
        )
