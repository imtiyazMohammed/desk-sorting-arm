"""
Test suite for the parametric CAD (Sessions D.1 through D.2).

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
   checked for every printable part
8. Yaw turntable (D.2a), including the horn/bearing compatibility rules that
   forced the 608ZZ off the yaw axis
9. Shoulder bracket (D.2b), including the stack closing on base_height_mm
10. Upper arm (D.2c), including the bending check and a swept clearance test
    through the shoulder joint's whole travel

The mesh checks parse the exported STL directly rather than trusting the
kernel's own report, because "the STL is watertight" is a property of the
tessellated output a slicer will actually read, not of the B-rep it came from.
"""

from __future__ import annotations

import collections
import itertools
import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.geometry import (
    BEARING_608ZZ,
    DEFAULT_ARM,
    DEFAULT_HARDWARE,
    ArmGeometry,
    BaseStack,
    BearingSpec,
    DeskClampSpec,
    FastenerSpec,
    HardwareSpec,
    LinkSpec,
    PETG_ALLOWABLE_STRESS_MPA,
    PETG_DENSITY_G_CM3,
    PETG_TENSILE_YIELD_MPA,
    STRUCTURAL_SAFETY_FACTOR,
    ServoHornSpec,
    ServoSpec,
    ShoulderBracketSpec,
    UPPER_ARM_LINK,
    YawTurntableSpec,
)

build123d = pytest.importorskip(
    "build123d", reason="build123d is required for the CAD suite (Python >= 3.10)"
)

from build123d import Pos, Rot  # noqa: E402  - must follow the importorskip

from cad._design import DesignStatus  # noqa: E402
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
from cad.shoulder_bracket import (  # noqa: E402
    ShoulderBracketDesignError,
    ShoulderBracketParameters,
    build_idler_plug,
    build_shoulder_bracket,
    export_idler_plug,
    export_shoulder_bracket,
)
from cad.upper_arm import (  # noqa: E402
    UpperArmDesignError,
    UpperArmParameters,
    build_upper_arm,
    export_upper_arm,
)
from cad.yaw_turntable import (  # noqa: E402
    TurntableDesignError,
    TurntableParameters,
    build_turntable,
    export_turntable,
)

#: Every printable part, so the mesh-integrity checks cover all of them.
PART_EXPORTERS = {
    "base_pedestal": export_pedestal,
    "desk_clamp_knob": export_knob,
    "desk_clamp_pressure_foot": export_pressure_foot,
    "yaw_turntable": export_turntable,
    "shoulder_bracket": export_shoulder_bracket,
    "shoulder_idler_plug": export_idler_plug,
    "upper_arm": export_upper_arm,
}
PART_BUILDERS = {
    "base_pedestal": build_pedestal,
    "desk_clamp_knob": build_knob,
    "desk_clamp_pressure_foot": build_pressure_foot,
    "yaw_turntable": build_turntable,
    "shoulder_bracket": build_shoulder_bracket,
    "shoulder_idler_plug": build_idler_plug,
    "upper_arm": build_upper_arm,
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


def _greedy_hardware() -> HardwareSpec:
    """
    A hardware set whose base stack swallows the whole base height.

    The turntable and bracket specs are inflated alongside BaseStack because
    HardwareSpec refuses to hold a budget that disagrees with the parts it is
    budgeting for -- which is the point of that cross-check, so the fixture
    honours it rather than working around it.
    """
    return HardwareSpec(
        base_stack=BaseStack(
            turntable_plate_thickness_mm=60.0, shoulder_bracket_rise_mm=60.0
        ),
        yaw_turntable=YawTurntableSpec(thickness_mm=60.0),
        shoulder_bracket=ShoulderBracketSpec(bracket_height_mm=60.0),
    )


@pytest.fixture(scope="module")
def foot() -> PressureFootParameters:
    """Default pressure-foot parameters, derived from the geometry singletons."""
    return PressureFootParameters.from_geometry()


@pytest.fixture(scope="module")
def knob() -> KnobParameters:
    """Default knob parameters, derived from the geometry singletons."""
    return KnobParameters.from_geometry()


@pytest.fixture(scope="module")
def turntable() -> TurntableParameters:
    """Default yaw-turntable parameters (Session D.2a)."""
    return TurntableParameters.from_geometry()


@pytest.fixture(scope="module")
def bracket() -> ShoulderBracketParameters:
    """Default shoulder-bracket parameters (Session D.2b)."""
    return ShoulderBracketParameters.from_geometry()


@pytest.fixture(scope="module")
def upper_arm() -> UpperArmParameters:
    """Default upper-arm parameters (Session D.2c)."""
    return UpperArmParameters.from_geometry()


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

    def test_yaw_bearing_is_the_standard_6806_triple(self):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert bearing.name == "6806ZZ"
        assert (
            bearing.bore_diameter_mm,
            bearing.outer_diameter_mm,
            bearing.width_mm,
        ) == (30.0, 42.0, 7.0)

    def test_608zz_survives_as_the_shoulder_idler(self):
        """
        D.2a moved the 608 off the yaw axis rather than deleting it.

        Its bore could not pass a servo horn, but it is exactly right for the
        shoulder yoke's undriven pivot, so it stays in the bill of materials.
        """
        idler = DEFAULT_HARDWARE.shoulder_idler_bearing
        assert idler.name == "608ZZ"
        assert (
            idler.bore_diameter_mm,
            idler.outer_diameter_mm,
            idler.width_mm,
        ) == (8.0, 22.0, 7.0)
        assert idler is not DEFAULT_HARDWARE.thrust_bearing

    def test_the_yaw_bearing_kept_the_608s_width(self):
        """
        The swap was chosen to leave the height budget untouched.

        A wider bearing would have eaten into BaseStack and moved the shoulder
        pivot, which is the one number the whole stack is built to hit.
        """
        assert (
            DEFAULT_HARDWARE.thrust_bearing.width_mm
            == DEFAULT_HARDWARE.shoulder_idler_bearing.width_mm
        )

    def test_bearing_seat_is_undersized_for_a_press_fit(self):
        bearing = BearingSpec(
            bore_diameter_mm=8.0,
            outer_diameter_mm=22.0,
            inner_race_outer_diameter_mm=11.5,
            press_fit_interference_mm=0.1,
        )
        assert bearing.seat_diameter_mm == pytest.approx(21.9)
        assert bearing.seat_diameter_mm < bearing.outer_diameter_mm

    def test_inner_race_land_lies_between_bore_and_outer_diameter(self):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert (
            bearing.bore_diameter_mm
            < bearing.inner_race_outer_diameter_mm
            < bearing.outer_diameter_mm
        )
        assert bearing.race_land_width_mm == pytest.approx(1.5)

    @pytest.mark.parametrize("race_od", [30.0, 42.0, 8.0, 50.0])
    def test_inner_race_outside_the_bore_and_od_rejected(self, race_od):
        with pytest.raises(ValueError, match="inner_race_outer_diameter_mm"):
            BearingSpec(inner_race_outer_diameter_mm=race_od)

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
        with pytest.raises(ValueError, match="non-positive height"):
            _greedy_hardware().pedestal_height_mm()

    def test_hardware_report_surfaces_the_verification_warning(self):
        report = DEFAULT_HARDWARE.hardware_report()
        assert "UNVERIFIED" in report
        assert "6806ZZ" in report and "DS3218" in report

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
            clamp.pressure_foot_seat_depth_mm
            + clamp.pressure_foot_web_mm
            + clamp.pad_recess_depth_mm
        )

    def test_pressure_foot_rise_shortens_the_screw_reach_needed(self):
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.pressure_foot_rise_above_tip_mm == pytest.approx(
            clamp.pressure_foot_height_mm - clamp.screw_tip_seat_height_mm
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

    def test_pressure_foot_seat_must_fit_inside_its_pad(self):
        with pytest.raises(ValueError, match="seat must be smaller"):
            DeskClampSpec(pressure_foot_seat_diameter_mm=20.0)

    @pytest.mark.parametrize("tip", [12.0, 1.0])
    def test_screw_tip_outside_the_seat_band_rejected(self, tip):
        """Too wide misses the cone; too narrow bottoms out on the apex flat."""
        with pytest.raises(ValueError, match="between the seat"):
            DeskClampSpec(bolt_tip_diameter_mm=tip)

    def test_seat_apex_wider_than_its_mouth_rejected(self):
        with pytest.raises(ValueError, match="apex flat must be smaller"):
            DeskClampSpec(pressure_foot_seat_apex_diameter_mm=12.0)

    @pytest.mark.parametrize("angle", [0.0, 180.0, 200.0, -30.0])
    def test_invalid_seat_angle_rejected(self, angle):
        with pytest.raises(ValueError, match="included angle"):
            DeskClampSpec(pressure_foot_seat_angle_deg=angle)

    def test_shallower_seat_cone_gives_a_thinner_foot(self):
        """Seat depth follows from the mouth diameter and the cone angle."""
        steep = DeskClampSpec(pressure_foot_seat_angle_deg=90.0)
        shallow = DeskClampSpec(pressure_foot_seat_angle_deg=140.0)
        assert shallow.pressure_foot_seat_depth_mm < steep.pressure_foot_seat_depth_mm
        assert shallow.pressure_foot_height_mm < steep.pressure_foot_height_mm

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

    def test_knob_has_no_bearing_boss(self):
        """
        Session D.1d removed it. The U-clamp's knob hangs free below the
        bottom arm and bears on nothing, so a boss sized for collar friction
        was modelling a contact that does not exist.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert not hasattr(clamp, "knob_boss_diameter_mm")
        assert not hasattr(clamp, "knob_boss_height_mm")

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

    def test_collar_friction_now_acts_at_the_screw_tip(self):
        """
        A wider screw tip rides further out in the cone, so its friction lever
        grows and the same torque yields less preload. This is the term that
        used to model the knob's boss.
        """
        narrow = DeskClampSpec(bolt_tip_diameter_mm=5.0)
        wide = DeskClampSpec(bolt_tip_diameter_mm=8.0)
        assert wide.bolt_preload_n(1.0) < narrow.bolt_preload_n(1.0)

    def test_steeper_seat_cone_wedges_harder_and_costs_preload(self):
        """
        A cone amplifies contact force by 1/sin(half-angle), so a steeper seat
        raises friction. Flat (180 deg) would be the no-wedge limit.
        """
        steep = DeskClampSpec(pressure_foot_seat_angle_deg=60.0)
        shallow = DeskClampSpec(pressure_foot_seat_angle_deg=150.0)
        assert steep.bolt_preload_n(1.0) < shallow.bolt_preload_n(1.0)

    def test_preload_rose_when_the_collar_term_was_corrected(self):
        """
        Session D.1d: the old model charged collar friction at the knob boss's
        radius (about 6.75 mm) for a contact the U-clamp does not have. Moving
        it to the screw tip roughly halves the friction lever.
        """
        clamp = DEFAULT_HARDWARE.desk_clamp
        assert clamp.bolt_preload_n(1.0) == pytest.approx(516.0, rel=0.02)
        assert clamp.bolt_preload_n(1.0) > 350.0

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
        assert params.bearing_seat_diameter_mm == pytest.approx(41.9)

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
        with pytest.raises(PedestalDesignError) as excinfo:
            PedestalParameters.from_geometry(hardware=_greedy_hardware())
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
            foot.seat_depth_mm + foot.web_thickness_mm + foot.pad_recess_depth_mm
        )

    def test_seat_is_a_cone_the_screw_tip_pivots_in(self, foot):
        """
        Session D.1d replaced a threaded bore with this. A threaded foot turns
        with the screw, dragging its pad across the desk and moving the
        friction lever out to the pad's radius.
        """
        assert foot.seat_diameter_mm > 2.0 * foot.tip_contact_radius_mm
        assert 0.0 < foot.seat_angle_deg < 180.0
        assert foot.seat_depth_mm == pytest.approx(
            (foot.seat_diameter_mm / 2.0 - foot.seat_apex_diameter_mm / 2.0)
            / math.tan(math.radians(foot.seat_angle_deg / 2.0))
        )

    def test_seat_apex_is_truncated(self, foot):
        """A true apex is unprintable and tessellates to degenerate triangles."""
        assert 0.0 < foot.seat_apex_diameter_mm < foot.seat_diameter_mm
        assert foot.seat_apex_diameter_mm < 2.0 * foot.tip_contact_radius_mm

    def test_screw_tip_seats_partway_down_the_cone(self, foot):
        """It stops where its edge meets the wall, not at the apex."""
        assert 0.0 < foot.tip_seat_height_mm < foot.seat_depth_mm

    def test_rise_above_tip_accounts_for_the_seat(self, foot):
        assert foot.rise_above_tip_mm == pytest.approx(
            foot.height_mm - foot.tip_seat_height_mm
        )

    def test_seat_fits_inside_the_pad_recess(self, foot):
        assert foot.seat_diameter_mm < foot.pad_diameter_mm

    def test_foot_spreads_the_load_over_a_bare_tip(self, foot):
        """The reason the part exists at all."""
        assert foot.bearing_area_ratio > 5.0
        assert foot.pad_area_mm2 > 250.0

    def test_foot_fits_the_throat_at_the_thickest_desk(self, foot):
        assert foot.height_mm <= foot.throat_clearance_mm

    def test_foot_too_tall_for_the_throat_is_rejected(self, foot):
        # Grow the seat and the height together so the part stays internally
        # consistent and it is genuinely the throat check that fires.
        tall = replace(foot, seat_depth_mm=foot.seat_depth_mm + 6.0,
                       height_mm=foot.height_mm + 6.0)
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

    def test_seat_wider_than_the_pad_is_rejected(self, foot):
        bored = replace(foot, seat_diameter_mm=foot.pad_diameter_mm + 1.0)
        with pytest.raises(PressureFootDesignError) as excinfo:
            bored.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_tip_wider_than_the_seat_mouth_is_rejected(self, foot):
        oversized = replace(foot, tip_contact_radius_mm=foot.seat_diameter_mm)
        with pytest.raises(PressureFootDesignError) as excinfo:
            oversized.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    @pytest.mark.parametrize(
        "kwargs", [{"diameter_mm": 0.0}, {"seat_depth_mm": -1.0},
                   {"web_thickness_mm": 0.0}]
    )
    def test_non_positive_dimensions_rejected(self, foot, kwargs):
        with pytest.raises(PressureFootDesignError) as excinfo:
            replace(foot, **kwargs).validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_report_names_the_key_dimensions(self, foot):
        report = foot.report()
        for heading in ("Swivel seat", "Pad recess", "Contact area"):
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

    def test_total_height_is_the_body_alone(self, knob):
        """Session D.1d removed the bearing boss; the knob is a plain disc."""
        assert knob.total_height_mm == pytest.approx(knob.body_thickness_mm)
        assert not hasattr(knob, "boss_diameter_mm")
        assert not hasattr(knob, "boss_height_mm")

    def test_flutes_are_evenly_spaced_on_the_rim(self, knob):
        positions = knob.flute_positions
        assert len(positions) == knob.flute_count
        for x, y in positions:
            assert math.hypot(x, y) == pytest.approx(knob.body_diameter_mm / 2.0)

    def test_flutes_bite_into_the_rim_without_reaching_the_bore(self, knob):
        assert knob.grip_min_radius_mm < knob.body_diameter_mm / 2.0
        assert knob.grip_min_radius_mm > knob.bolt_hole_diameter_mm / 2.0

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
        for heading in ("Hex socket", "Shank bore", "Grip flutes"):
            assert heading in report


# =========================================================================
# 8. Session D.2a -- servo horn and yaw turntable
# =========================================================================


class TestServoHornSpec:
    """
    The coupling that made D.2a rewrite the yaw bearing.

    Its disc diameter is what an 8 mm bore could not pass, and its height is
    what the turntable's spigot cap has to be left over from.
    """

    def test_spline_diameter_is_the_verified_number(self):
        horn = DEFAULT_HARDWARE.servo_horn
        assert horn.spline_teeth == 25
        assert horn.spline_diameter_mm == pytest.approx(5.9)
        assert "spline_diameter_mm" not in horn.UNVERIFIED_FIELDS

    def test_horn_geometry_is_flagged_unverified(self):
        horn = DEFAULT_HARDWARE.servo_horn
        assert horn.UNVERIFIED_FIELDS
        for field_name in horn.UNVERIFIED_FIELDS:
            assert hasattr(horn, field_name)
        assert "UNVERIFIED" in horn.unverified_report()

    def test_spline_is_far_too_fine_to_print(self):
        """
        The reason a bought horn is in the bill of materials at all.

        25 teeth on a 5.9 mm circle is a 0.74 mm pitch -- under two extrusion
        widths at a 0.4 mm nozzle.
        """
        horn = DEFAULT_HARDWARE.servo_horn
        assert horn.spline_tooth_pitch_mm == pytest.approx(0.741, abs=0.005)
        assert horn.spline_tooth_pitch_mm < 2 * 0.4

    def test_index_resolution_is_the_spline_step(self):
        assert DEFAULT_HARDWARE.servo_horn.index_resolution_deg == pytest.approx(14.4)

    def test_total_height_is_hub_plus_disc(self):
        horn = DEFAULT_HARDWARE.servo_horn
        assert horn.total_height_mm == pytest.approx(
            horn.hub_height_mm + horn.disc_thickness_mm
        )

    def test_bolt_positions_lie_on_the_bolt_circle(self):
        horn = DEFAULT_HARDWARE.servo_horn
        positions = horn.bolt_positions()
        assert len(positions) == horn.bolt_count
        for x, y in positions:
            assert math.hypot(x, y) == pytest.approx(horn.bolt_circle_mm / 2.0)

    def test_bolt_circle_wider_than_the_disc_rejected(self):
        with pytest.raises(ValueError, match="must be smaller than the disc"):
            ServoHornSpec(bolt_circle_mm=30.0)

    def test_rim_too_thin_for_the_bolt_rejected(self):
        with pytest.raises(ValueError, match="of rim outside the bolt circle"):
            ServoHornSpec(bolt_circle_mm=22.0)

    def test_hub_narrower_than_its_spline_rejected(self):
        with pytest.raises(ValueError, match="wider than the spline"):
            ServoHornSpec(hub_diameter_mm=5.0)

    @pytest.mark.parametrize("count", [0, 1, 2])
    def test_too_few_bolts_rejected(self, count):
        with pytest.raises(ValueError, match="at least 3"):
            ServoHornSpec(bolt_count=count)


class TestHornBearingCompatibility:
    """
    The D.2a finding, pinned as an invariant.

    The coupling passes through the bearing's bore, so the bearing has to be
    bigger than the horn in both directions. A 608 fails both tests, which is
    why it is no longer the yaw bearing.
    """

    def test_the_horn_fits_through_the_yaw_bearings_bore(self):
        horn, bearing = DEFAULT_HARDWARE.servo_horn, DEFAULT_HARDWARE.thrust_bearing
        assert horn.disc_diameter_mm < bearing.bore_diameter_mm

    def test_the_horn_is_shorter_than_the_bearing_is_wide(self):
        horn, bearing = DEFAULT_HARDWARE.servo_horn, DEFAULT_HARDWARE.thrust_bearing
        assert horn.total_height_mm < bearing.width_mm

    def test_a_608_yaw_bearing_is_rejected_for_the_horn_it_cannot_pass(self):
        """
        The original design, re-run. It fails loudly instead of quietly.

        This is the whole of Session D.2a's first finding in one assertion: an
        8 mm bore cannot pass a 25 mm horn, so a stack built on a 608 has
        nowhere to put the coupling between the servo and the turntable.
        """
        with pytest.raises(ValueError, match="coupling sits inside the bore"):
            HardwareSpec(thrust_bearing=BEARING_608ZZ)

    def test_a_horn_taller_than_the_bearing_is_rejected(self):
        tall = ServoHornSpec(hub_height_mm=5.0, disc_thickness_mm=5.0)
        with pytest.raises(ValueError, match="ride the horn"):
            HardwareSpec(servo_horn=tall)


class TestYawTurntableParameters:
    def test_default_parameters_validate(self, turntable):
        assert turntable.validate() is DesignStatus.OK

    def test_dimensions_come_from_the_geometry_singletons(self, turntable):
        spec = DEFAULT_HARDWARE.yaw_turntable
        assert turntable.diameter_mm == spec.diameter_mm
        assert turntable.thickness_mm == spec.thickness_mm
        assert turntable.race_relief_depth_mm == spec.bearing_race_recess_mm

    def test_plate_thickness_matches_the_height_budget(self, turntable):
        """The same 6 mm counted once in BaseStack and once in the part."""
        assert turntable.thickness_mm == pytest.approx(
            DEFAULT_HARDWARE.base_stack.turntable_plate_thickness_mm
        )

    def test_turntable_covers_pedestal_bearing(self, turntable):
        """
        The plate has to cover the bearing it turns on -- not the turret.

        D.2a's brief asked for a plate wider than "the pedestal turret top".
        The turret is not round: it is 49.9 x 66.2 mm, an 82.9 mm diagonal, so
        a disc that covered it would overhang the 60 mm-deep top arm it stands
        on. What the plate must actually cover is the bearing, and it does.
        """
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert turntable.diameter_mm > bearing.outer_diameter_mm
        assert turntable.diameter_mm >= (
            turntable.race_relief_outer_diameter_mm
            + 2.0 * DEFAULT_HARDWARE.min_wall_thickness_mm
        )

    def test_turntable_horn_pattern_matches_servo(self, turntable):
        horn = DEFAULT_HARDWARE.servo_horn
        assert turntable.horn_bolt_count == horn.bolt_count
        assert turntable.horn_bolt_circle_mm == horn.bolt_circle_mm
        for (tx, ty), (hx, hy) in zip(
            turntable.horn_bolt_positions, horn.bolt_positions()
        ):
            assert (tx, ty) == pytest.approx((hx, hy))
        assert turntable.horn_pocket_diameter_mm == pytest.approx(
            horn.disc_diameter_mm + 2.0 * DEFAULT_HARDWARE.print_clearance_mm
        )
        assert turntable.horn_pocket_depth_mm == pytest.approx(horn.total_height_mm)

    def test_spigot_is_a_slip_fit_in_the_bore(self, turntable):
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert turntable.spigot_diameter_mm < bearing.bore_diameter_mm
        assert turntable.spigot_diameter_mm == pytest.approx(
            bearing.bore_diameter_mm
            - DEFAULT_HARDWARE.yaw_turntable.spigot_bore_fit_mm
        )
        assert turntable.spigot_depth_mm == pytest.approx(bearing.width_mm)

    def test_relief_clears_the_outer_race_and_lands_on_the_inner(self, turntable):
        """
        The correction to D.2a's ``bearing_race_recess_mm``.

        A plain 1 mm recess over a bearing standing 0.5 mm proud would drop the
        plate onto the printed turret face and leave the bearing carrying
        nothing. It only works as a relief over the *outer* ring, with the land
        inside it bearing on the inner ring.
        """
        bearing = DEFAULT_HARDWARE.thrust_bearing
        assert turntable.race_relief_depth_mm > bearing.proud_mm
        assert turntable.race_relief_inner_diameter_mm == pytest.approx(
            bearing.inner_race_outer_diameter_mm
        )
        assert turntable.race_land_width_mm > 0.0

    def test_spigot_cap_is_what_the_horn_leaves_of_the_bearing_width(self, turntable):
        horn, bearing = DEFAULT_HARDWARE.servo_horn, DEFAULT_HARDWARE.thrust_bearing
        assert turntable.spigot_cap_mm == pytest.approx(
            bearing.width_mm - horn.total_height_mm
        )
        assert turntable.spigot_cap_mm > 0.0

    def test_ring_around_the_horn_pocket_is_the_tightest_feature(self, turntable):
        """
        2 mm, exactly the floor -- worth a test so a change cannot slip past.

        Half the minimum wall is allowed because the ring is confined by the
        steel race over its whole height, the same allowance the pressure
        foot's rim takes.
        """
        floor = DEFAULT_HARDWARE.min_wall_thickness_mm / 2.0
        assert turntable.spigot_ring_wall_mm >= floor
        assert turntable.spigot_ring_wall_mm < DEFAULT_HARDWARE.min_wall_thickness_mm

    def test_bracket_bolts_clear_the_bearing_and_the_rim(self, turntable):
        assert turntable.bracket_bolt_clearance_to_bearing_mm > 0.0
        assert (
            turntable.rim_outside_bracket_bolts_mm
            >= DEFAULT_HARDWARE.min_wall_thickness_mm
        )

    def test_bracket_bolts_are_blind_so_nothing_protrudes_underneath(self, turntable):
        """
        There is only ``BearingSpec.proud_mm`` under the plate.

        A nut or a screw tip poking through would foul the turret half a
        millimetre below, so the holes stop short and keep a floor.
        """
        assert turntable.bracket_bolt_floor_mm > 0.0
        assert (
            turntable.bracket_bolt_depth_mm < turntable.thickness_mm
        )

    def test_report_names_the_key_dimensions(self, turntable):
        report = turntable.report()
        for token in ("Spigot", "Horn pocket", "Outer-race relief", "witness"):
            assert token in report


class TestYawTurntableDesignRules:
    def test_shallow_relief_that_would_miss_the_bearing_is_rejected(self, turntable):
        broken = replace(turntable, race_relief_depth_mm=0.25)
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_relief_starting_inside_the_spigot_is_rejected(self, turntable):
        broken = replace(turntable, race_relief_inner_diameter_mm=20.0)
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_horn_taller_than_the_spigot_is_rejected(self, turntable):
        broken = replace(turntable, horn_pocket_depth_mm=turntable.spigot_depth_mm)
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_thin_ring_around_the_horn_pocket_is_rejected(self, turntable):
        broken = replace(
            turntable, horn_pocket_diameter_mm=turntable.spigot_diameter_mm - 1.0
        )
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_bracket_bolts_over_the_bearing_are_rejected(self, turntable):
        broken = replace(turntable, bracket_bolt_pattern_mm=(30.0, 10.0))
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_bracket_bolts_too_near_the_rim_are_rejected(self, turntable):
        broken = replace(turntable, bracket_bolt_pattern_mm=(65.0, 10.0))
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_bracket_bolts_breaking_through_the_plate_are_rejected(self, turntable):
        broken = replace(turntable, bracket_bolt_depth_mm=turntable.thickness_mm)
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_counterbore_breaking_out_of_the_ring_is_rejected(self, turntable):
        broken = replace(turntable, horn_counterbore_diameter_mm=14.0)
        with pytest.raises(TurntableDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    @pytest.mark.parametrize(
        "kwargs",
        [{"diameter_mm": 0.0}, {"thickness_mm": -1.0}, {"spigot_depth_mm": 0.0}],
    )
    def test_non_positive_dimensions_rejected(self, turntable, kwargs):
        with pytest.raises(TurntableDesignError):
            replace(turntable, **kwargs).validate()

    def test_build_rejects_invalid_parameters(self, turntable):
        with pytest.raises(TurntableDesignError):
            build_turntable(replace(turntable, race_relief_depth_mm=0.1))


class TestYawTurntableSolid:
    def test_bounding_box_spans_plate_and_spigot(self, turntable):
        box = build_turntable(turntable).bounding_box()
        assert box.max.Z == pytest.approx(turntable.thickness_mm)
        assert box.min.Z == pytest.approx(-turntable.spigot_depth_mm)
        assert box.max.X - box.min.X == pytest.approx(turntable.diameter_mm, abs=0.2)

    def test_horn_pocket_actually_removes_material(self, turntable):
        solid = build_turntable(turntable)
        shallow = replace(turntable, horn_pocket_depth_mm=1.0)
        assert solid.volume < build_turntable(shallow).volume


# =========================================================================
# 9. Session D.2b -- shoulder bracket
# =========================================================================


class TestShoulderBracketParameters:
    def test_default_parameters_validate(self, bracket):
        assert bracket.validate() is DesignStatus.OK

    def test_shoulder_shaft_height_matches_base_height(self, bracket):
        """
        The loop closes: pedestal + bearing + turntable + bracket = 100 mm.

        This is what the whole base stack exists to achieve, and it is checked
        against ArmGeometry rather than against a copy of the number.
        """
        assert bracket.shaft_axis_z_in_desk_frame_mm == pytest.approx(
            DEFAULT_ARM.base_height_mm, abs=1e-9
        )
        assert bracket.stack_below_mm == pytest.approx(
            DEFAULT_HARDWARE.pedestal_height_mm()
            + DEFAULT_HARDWARE.thrust_bearing.proud_mm
            + DEFAULT_HARDWARE.yaw_turntable.thickness_mm
        )

    def test_bracket_rise_matches_the_height_budget(self, bracket):
        assert bracket.pitch_axis_z_mm == pytest.approx(
            DEFAULT_HARDWARE.base_stack.shoulder_bracket_rise_mm
        )

    def test_shoulder_servo_fits_in_bracket_cavity(self, bracket):
        servo = DEFAULT_HARDWARE.shoulder_pitch_servo
        clearance = DEFAULT_HARDWARE.print_clearance_mm
        span_x, span_y, span_z = bracket.servo_cavity_span_mm
        assert span_x == pytest.approx(servo.body_length_mm + 2.0 * clearance)
        assert span_z == pytest.approx(servo.body_width_mm + 2.0 * clearance)
        # Across the walls, what must fit is the body behind the ears; the
        # remaining 10 mm passes through the driven wall's slot.
        assert span_y == pytest.approx(
            servo.body_depth_behind_ears_mm + 2.0 * clearance
        )
        assert span_y >= servo.body_depth_behind_ears_mm

    def test_servo_straddles_the_pitch_axis_and_clears_the_plate(self, bracket):
        assert bracket.cavity_z_min_mm < bracket.pitch_axis_z_mm < bracket.cavity_z_max_mm
        assert bracket.cavity_z_min_mm > bracket.plate_thickness_mm

    def test_standing_the_servo_on_end_would_not_fit(self, bracket):
        """
        The measurement behind "the servo lies on its side".

        Its ears are 49.5 mm apart along the body's length. Stood on end at a
        24 mm rise, the lower one lands 0.75 mm *below* the turntable the
        bracket is bolted to -- so the orientation is forced, not chosen.
        """
        servo = DEFAULT_HARDWARE.shoulder_pitch_servo
        lower_ear_z = (
            bracket.pitch_axis_z_mm - servo.flange_hole_spacing_long_mm / 2.0
        )
        assert lower_ear_z == pytest.approx(-0.75)
        assert lower_ear_z < 0.0

    def test_servo_screws_land_in_wall_material_not_the_slot(self, bracket):
        for screw_x, _ in bracket.servo_screw_positions:
            assert not (
                bracket.cavity_x_min_mm <= screw_x <= bracket.cavity_x_max_mm
            )

    def test_servo_screws_are_blind_with_a_floor(self, bracket):
        assert bracket.servo_screw_depth_mm < bracket.wall_thickness_mm

    def test_bracket_base_matches_turntable_top_pattern(self, bracket):
        """The bracket drills exactly where the turntable threads."""
        expected = DEFAULT_HARDWARE.yaw_turntable.bracket_bolt_positions()
        assert sorted(bracket.turntable_bolt_positions) == sorted(expected)
        assert bracket.turntable_bolt_clearance_mm > (
            DEFAULT_HARDWARE.yaw_turntable.bracket_bolt_nominal_diameter_mm
        )

    def test_turntable_screws_are_reachable_between_the_walls(self, bracket):
        for _, bolt_y in bracket.turntable_bolt_positions:
            assert abs(bolt_y) + bracket.turntable_bolt_clearance_mm / 2.0 < (
                bracket.wall_inner_y_mm
            )

    def test_walls_are_symmetric_about_the_mid_plane(self, bracket):
        """
        The retrofit provision: both walls carry the same servo pattern.

        Which one is driven is a matter of which side the servo goes in; the
        other takes the idler plug today and a reduction plate later.
        """
        assert bracket.wall_inner_y_mm > 0.0
        assert bracket.wall_outer_y_mm > bracket.wall_inner_y_mm
        solid = build_shoulder_bracket(bracket)
        box = solid.bounding_box()
        assert box.min.Y == pytest.approx(-box.max.Y, abs=1e-6)

    def test_shaft_direction_matches_the_fk_convention(self, bracket):
        """
        Shoulder pitch is a rotation about the base frame's +Y in
        forward_kinematics.py, so the output shaft points +Y and the servo's
        positive rotation is positive theta_2 with no sign flip.
        """
        assert bracket.shaft_direction == "+Y"
        assert DEFAULT_HARDWARE.shoulder_bracket.shaft_sign == 1.0

    def test_horn_face_is_where_the_upper_arm_bolts(self, bracket):
        servo, horn = (
            DEFAULT_HARDWARE.shoulder_pitch_servo,
            DEFAULT_HARDWARE.servo_horn,
        )
        assert bracket.horn_face_y_mm == pytest.approx(
            bracket.shaft_face_y_mm
            + servo.shaft_boss_height_mm
            + horn.total_height_mm
        )

    def test_cable_slot_leaves_at_the_rear_not_down_the_yaw_axis(self, bracket):
        """
        D.2b's brief routed the shoulder lead down through the turntable.

        It cannot go that way: the yaw servo's shaft and its horn fill the yaw
        axis solid all the way from the turret to the turntable's cap. The slot
        is a notch in the plate's rear edge instead, and the cable needs a
        service loop for the +/-135 degrees of yaw travel.
        """
        assert bracket.cable_slot_x_mm > 0.0
        slot_far_edge = bracket.plate_x_min_mm + bracket.cable_slot_x_mm
        assert slot_far_edge < 0.0
        assert abs(bracket.plate_x_min_mm) > bracket.turntable_radius_mm

    def test_plate_overhangs_the_turntable_by_design(self, bracket):
        """
        Recorded rather than asserted away: the plate is sized by the servo's
        54.5 mm flange, the disc by the bearing. The bolt pattern is well
        inside the disc, so the overhang carries nothing.
        """
        assert bracket.turntable_overhang_mm > 0.0
        for bolt_x, bolt_y in bracket.turntable_bolt_positions:
            assert math.hypot(bolt_x, bolt_y) < bracket.turntable_radius_mm

    def test_bracket_stays_within_reach_of_the_yoke(self, bracket):
        assert 0.0 < bracket.max_radius_from_pitch_axis_mm < 60.0

    def test_report_names_the_key_dimensions(self, bracket):
        report = bracket.report()
        for token in ("Pitch axis", "Horn face", "Idler axle", "Servo slot"):
            assert token in report


class TestShoulderBracketDesignRules:
    def test_a_stack_that_misses_the_base_height_is_rejected(self, bracket):
        broken = replace(bracket, pitch_axis_z_mm=bracket.pitch_axis_z_mm + 1.0)
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER
        assert "base_height_mm" in str(excinfo.value)

    def test_a_budget_that_disagrees_with_the_bracket_is_caught_upstream(self):
        """
        BaseStack and ShoulderBracketSpec state the same rise; HardwareSpec
        refuses to hold two different answers.
        """
        with pytest.raises(ValueError, match="budgets"):
            HardwareSpec(shoulder_bracket=ShoulderBracketSpec(bracket_height_mm=30.0))

    def test_servo_fouling_the_base_plate_is_rejected(self, bracket):
        broken = replace(bracket, plate_thickness_mm=bracket.cavity_z_min_mm + 1.0)
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_screws_breaking_through_the_wall_are_rejected(self, bracket):
        broken = replace(bracket, servo_screw_depth_mm=bracket.wall_thickness_mm)
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_walls_overhanging_the_base_plate_are_rejected(self, bracket):
        broken = replace(bracket, plate_x_max_mm=bracket.wall_x_max_mm - 1.0)
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_turntable_screws_under_a_wall_are_rejected(self, bracket):
        broken = replace(
            bracket,
            turntable_bolt_positions=((0.0, bracket.wall_inner_y_mm + 1.0),),
        )
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION
        assert "driver" in str(excinfo.value)

    def test_a_cable_slot_that_cuts_the_walls_is_rejected(self, bracket):
        broken = replace(bracket, cable_slot_y_mm=2.0 * bracket.wall_inner_y_mm)
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_an_axle_too_short_for_its_bearing_is_rejected(self, bracket):
        broken = replace(bracket, idler_axle_length_mm=3.0)
        with pytest.raises(ShoulderBracketDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER

    def test_build_rejects_invalid_parameters(self, bracket):
        with pytest.raises(ShoulderBracketDesignError):
            build_shoulder_bracket(replace(bracket, pitch_axis_z_mm=30.0))


class TestShoulderBracketSolid:
    def test_bracket_is_one_solid_standing_on_its_plate(self, bracket):
        solid = build_shoulder_bracket(bracket)
        assert len(solid.solids()) == 1
        box = solid.bounding_box()
        assert box.min.Z == pytest.approx(0.0)
        assert box.max.Z == pytest.approx(bracket.wall_top_z_mm)

    def test_servo_slot_removes_material_from_both_walls(self, bracket):
        solid = build_shoulder_bracket(bracket)
        closed = replace(
            bracket,
            cavity_x_min_mm=-1.0,
            cavity_x_max_mm=1.0,
        )
        assert solid.volume < build_shoulder_bracket(closed).volume

    def test_idler_plug_is_one_solid_reaching_its_bearing(self, bracket):
        plug = build_idler_plug(bracket)
        assert len(plug.solids()) == 1
        box = plug.bounding_box()
        assert box.min.Y == pytest.approx(0.0, abs=1e-6)
        assert box.max.Y == pytest.approx(
            bracket.plug_thickness_mm
            + bracket.plug_boss_length_mm
            + bracket.idler_axle_length_mm
        )

    def test_idler_plug_uses_the_servos_own_hole_pattern(self, bracket):
        """
        What makes the plug and a second servo interchangeable in that wall.
        """
        plug = build_idler_plug(bracket)
        solid = replace(bracket, servo_screw_diameter_mm=0.5)
        assert plug.volume < build_idler_plug(solid).volume


# =========================================================================
# 10. Session D.2c -- upper arm (L1)
# =========================================================================


class TestUpperArmLink:
    def test_link_length_comes_from_arm_geometry(self):
        assert UPPER_ARM_LINK.length_mm == DEFAULT_ARM.l1_upper_arm_mm

    def test_section_properties_match_the_closed_form(self):
        link = UPPER_ARM_LINK
        outer_w, outer_h = link.cross_section_width_mm, link.cross_section_height_mm
        inner_w, inner_h = link.inner_width_mm, link.inner_height_mm
        assert link.cross_section_area_mm2 == pytest.approx(
            outer_w * outer_h - inner_w * inner_h
        )
        assert link.second_moment_mm4 == pytest.approx(
            (outer_w * outer_h**3 - inner_w * inner_h**3) / 12.0
        )
        assert link.section_modulus_mm3 == pytest.approx(
            link.second_moment_mm4 / (outer_h / 2.0)
        )

    def test_upper_arm_bending_stress_below_yield(self):
        """
        The structural check D.2c asked for, and it passes with room to spare.

        40 x 25 x 3 mm carries about 1.25 MPa at the shoulder end against a
        25 MPa allowable -- PETG's 50 MPa yield with a safety factor of two.
        No enlargement is needed; the section is stiffness- and packaging-
        driven, not strength-driven.
        """
        moment = DEFAULT_ARM.shoulder_moment_nm()
        stress = UPPER_ARM_LINK.bending_stress_mpa(moment)
        assert stress < PETG_ALLOWABLE_STRESS_MPA
        assert stress < PETG_TENSILE_YIELD_MPA / STRUCTURAL_SAFETY_FACTOR
        assert stress == pytest.approx(1.25, abs=0.05)
        assert PETG_ALLOWABLE_STRESS_MPA / stress > 15.0

    def test_a_shallow_section_would_fail_the_same_check(self):
        """
        The check has teeth: bending capacity goes as the square of depth, so a
        section shallow enough to matter is caught.
        """
        shallow = replace(
            UPPER_ARM_LINK, cross_section_height_mm=8.0, wall_thickness_mm=1.0
        )
        assert shallow.bending_stress_mpa(
            DEFAULT_ARM.shoulder_moment_nm() * 40.0
        ) > PETG_ALLOWABLE_STRESS_MPA

    def test_deflection_stays_within_a_few_millimetres(self):
        assert UPPER_ARM_LINK.tip_deflection_mm(
            DEFAULT_ARM.shoulder_moment_nm()
        ) < 5.0

    def test_beam_mass_is_the_section_times_the_length(self):
        link = UPPER_ARM_LINK
        assert link.estimated_mass_g() == pytest.approx(
            link.cross_section_area_mm2 * link.length_mm / 1000.0 * PETG_DENSITY_G_CM3
        )
        assert link.estimated_mass_g() > 150.0

    def test_negative_moment_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            UPPER_ARM_LINK.bending_stress_mpa(-1.0)

    def test_walls_that_meet_in_the_middle_rejected(self):
        with pytest.raises(ValueError, match="no hollow left"):
            replace(UPPER_ARM_LINK, wall_thickness_mm=13.0)

    def test_a_channel_as_wide_as_the_beam_rejected(self):
        with pytest.raises(ValueError, match="as wide as the beam"):
            replace(UPPER_ARM_LINK, cable_channel_mm=(40.0, 8.0))

    def test_tabs_that_would_close_the_channel_rejected(self):
        with pytest.raises(ValueError, match="close the channel"):
            replace(UPPER_ARM_LINK, strain_relief_tab_width_mm=100.0)


class TestUpperArmParameters:
    def test_default_parameters_validate(self, upper_arm):
        assert upper_arm.validate() is DesignStatus.OK

    def test_upper_arm_length_matches_geometry(self, upper_arm):
        """Shoulder axis at x = 0, elbow axis at exactly l1_upper_arm_mm."""
        assert upper_arm.length_mm == DEFAULT_ARM.l1_upper_arm_mm
        assert upper_arm.elbow_axis_x_mm == pytest.approx(
            DEFAULT_ARM.l1_upper_arm_mm
        )
        servo = DEFAULT_HARDWARE.elbow_pitch_servo
        body_centre = sum(x for x, _ in upper_arm.elbow_screw_positions) / 4.0
        assert body_centre == pytest.approx(
            upper_arm.elbow_axis_x_mm - servo.body_offset_from_shaft_axis_mm
        )

    def test_upper_arm_end_cavity_fits_elbow_servo(self, upper_arm):
        servo = DEFAULT_HARDWARE.elbow_pitch_servo
        clearance = DEFAULT_HARDWARE.print_clearance_mm
        assert upper_arm.elbow_cavity_x_max_mm - upper_arm.elbow_cavity_x_min_mm == (
            pytest.approx(servo.body_length_mm + 2.0 * clearance)
        )
        assert upper_arm.elbow_cavity_z_max_mm - upper_arm.elbow_cavity_z_min_mm == (
            pytest.approx(servo.body_width_mm + 2.0 * clearance)
        )
        assert 2.0 * upper_arm.elbow_wall_inner_y_mm == pytest.approx(
            servo.body_depth_behind_ears_mm + 2.0 * clearance
        )

    def test_the_servo_does_not_fit_inside_the_beam_section(self, upper_arm):
        """
        Why the distal end swells into a housing rather than being a cavity in
        the beam, as D.2c's brief described. The servo's smallest dimension is
        20 mm and its largest is 40.5; a 40 x 25 mm section cannot swallow it
        in any orientation once walls are counted.
        """
        servo = DEFAULT_HARDWARE.elbow_pitch_servo
        assert servo.body_height_mm > upper_arm.beam_width_mm
        assert servo.body_length_mm + 2.0 * upper_arm.wall_thickness_mm > (
            upper_arm.beam_width_mm
        )
        assert upper_arm.housing_half_y_mm > upper_arm.beam_width_mm / 2.0
        assert upper_arm.housing_z_max_mm > upper_arm.beam_height_mm / 2.0

    def test_yoke_keeps_the_beam_on_the_pitch_axis(self, upper_arm):
        """
        The reason for a yoke instead of one flange.

        A single flange would carry the beam 35.5 mm off the axis at best, and
        forward_kinematics.py has no term for a shoulder offset. Two flanges,
        equally spaced, put the beam back on it.
        """
        assert upper_arm.driven_flange_inner_y_mm == pytest.approx(
            -upper_arm.idler_flange_inner_y_mm
        )
        solid = build_upper_arm(upper_arm)
        box = solid.bounding_box()
        # The beam and both flange faces are symmetric about y = 0; only the
        # idler flange's extra depth for its bearing breaks it.
        assert abs(box.min.Y + box.max.Y) <= (
            upper_arm.idler_flange_thickness_mm
            - upper_arm.driven_flange_thickness_mm
            + 1e-6
        )

    def test_the_driven_flange_lands_on_the_brackets_horn(self, upper_arm, bracket):
        assert upper_arm.driven_flange_inner_y_mm == pytest.approx(
            bracket.horn_face_y_mm
        )

    def test_beam_starts_outside_everything_the_bracket_occupies(
        self, upper_arm, bracket
    ):
        assert upper_arm.beam_start_x_mm > bracket.max_radius_from_pitch_axis_mm

    def test_idler_flange_swallows_its_bearing_with_a_floor(self, upper_arm):
        idler = DEFAULT_HARDWARE.shoulder_idler_bearing
        assert upper_arm.idler_seat_diameter_mm == pytest.approx(idler.seat_diameter_mm)
        assert upper_arm.idler_seat_depth_mm == pytest.approx(idler.width_mm)
        assert upper_arm.idler_seat_floor_mm >= (
            DEFAULT_HARDWARE.min_wall_thickness_mm / 2.0
        )

    def test_cable_trough_sits_above_a_closed_box_section(self, upper_arm):
        """
        The correction to D.2c's brief: an 8 mm channel cut into a 3 mm top
        wall breaks into the hollow and turns a closed section into an open
        one, losing most of its torsional stiffness. The trough stands on the
        beam instead.
        """
        assert upper_arm.channel_depth_mm > upper_arm.wall_thickness_mm
        assert upper_arm.channel_top_z_mm > upper_arm.beam_height_mm / 2.0
        assert upper_arm.channel_top_z_mm == pytest.approx(
            upper_arm.beam_height_mm / 2.0 + upper_arm.channel_depth_mm
        )

    def test_strain_relief_tabs_are_spaced_along_the_beam(self, upper_arm):
        positions = upper_arm.strain_relief_positions
        assert len(positions) >= 3
        for first, second in zip(positions, positions[1:]):
            assert second - first == pytest.approx(
                upper_arm.strain_relief_pitch_mm
            )
        assert all(
            upper_arm.beam_start_x_mm < x < upper_arm.housing_x_min_mm
            for x in positions
        )

    def test_stress_margin_is_reported(self, upper_arm):
        assert upper_arm.bending_stress_mpa == pytest.approx(1.25, abs=0.05)
        assert upper_arm.stress_margin > 15.0

    def test_report_names_the_key_dimensions(self, upper_arm):
        report = upper_arm.report()
        for token in ("Shoulder yoke", "Elbow housing", "Cable trough", "bending"):
            assert token in report


class TestUpperArmDesignRules:
    def test_an_overstressed_section_is_rejected(self, upper_arm):
        broken = replace(upper_arm, section_modulus_mm3=50.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.INVALID_PARAMETER
        assert "allowable" in str(excinfo.value)

    def test_a_beam_that_would_strike_the_bracket_is_rejected(self, upper_arm):
        broken = replace(upper_arm, beam_start_x_mm=10.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_flanges_inside_the_beam_width_are_rejected(self, upper_arm):
        broken = replace(upper_arm, driven_flange_inner_y_mm=10.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_a_flange_too_small_for_its_bearing_is_rejected(self, upper_arm):
        broken = replace(upper_arm, flange_diameter_mm=26.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.WALL_TOO_THIN

    def test_a_counterbore_off_the_flange_is_rejected(self, upper_arm):
        broken = replace(upper_arm, horn_counterbore_diameter_mm=30.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_a_housing_overlapping_the_yoke_is_rejected(self, upper_arm):
        broken = replace(upper_arm, housing_x_min_mm=upper_arm.beam_start_x_mm - 1.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_a_trough_wider_than_the_beam_is_rejected(self, upper_arm):
        broken = replace(upper_arm, channel_width_mm=40.0)
        with pytest.raises(UpperArmDesignError) as excinfo:
            broken.validate()
        assert excinfo.value.status is DesignStatus.FEATURE_COLLISION

    def test_build_rejects_invalid_parameters(self, upper_arm):
        with pytest.raises(UpperArmDesignError):
            build_upper_arm(replace(upper_arm, beam_start_x_mm=5.0))


class TestUpperArmSolid:
    def test_upper_arm_is_one_solid(self, upper_arm):
        assert len(build_upper_arm(upper_arm).solids()) == 1

    def test_bounding_box_spans_yoke_to_elbow_housing(self, upper_arm):
        box = build_upper_arm(upper_arm).bounding_box()
        assert box.min.X == pytest.approx(-upper_arm.flange_diameter_mm / 2.0, abs=0.2)
        assert box.max.X == pytest.approx(upper_arm.housing_x_max_mm, abs=0.2)
        assert box.max.Y == pytest.approx(
            upper_arm.driven_flange_inner_y_mm
            + upper_arm.driven_flange_thickness_mm
        )

    def test_the_beam_is_actually_hollow(self, upper_arm):
        solid = build_upper_arm(upper_arm)
        filled = replace(upper_arm, wall_thickness_mm=12.0)
        assert solid.volume < build_upper_arm(filled).volume

    def test_the_yoke_clears_the_bracket_through_the_joints_travel(
        self, upper_arm, bracket
    ):
        """
        Swept, not merely checked at the zero pose.

        The yoke rotates about the pitch axis, so a clearance that holds at
        zero says nothing about 120 degrees up. This walks the joint's whole
        mechanical range and insists on zero intersection at every step.
        """
        arm_solid = build_upper_arm(upper_arm)
        static = Pos(0, 0, -bracket.pitch_axis_z_mm) * build_shoulder_bracket(
            bracket
        )
        limit = DEFAULT_ARM.joint_limits[1]
        for degrees in np.linspace(limit.min_deg, limit.max_deg, 10):
            swung = Rot(0, float(degrees), 0) * arm_solid
            assert (swung & static).volume == pytest.approx(0.0, abs=1e-6), (
                f"upper arm strikes the shoulder bracket at "
                f"theta_2 = {degrees:.1f} deg"
            )


# =========================================================================
# 11. Shared primitives
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

    def test_pedestal_does_not_intersect_the_arm(self, assembly):
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

    def test_assembly_uses_real_upper_stack(self, assembly):
        """
        The turntable, bracket and upper arm are placed parts now, not
        cylinders standing in for them.

        Checked by volume rather than by name: a placeholder disc and a real
        turntable can share a label, but they cannot share a volume, because
        the real part is hollowed by its horn pocket, race relief and holes.
        """
        for name, builder, params in (
            ("yaw turntable", build_turntable, TurntableParameters.from_geometry()),
            (
                "shoulder bracket",
                build_shoulder_bracket,
                ShoulderBracketParameters.from_geometry(),
            ),
            ("L1 upper arm", build_upper_arm, UpperArmParameters.from_geometry()),
        ):
            placed = assembly.by_name(name)
            assert placed.printed, f"{name} should count toward the print estimate"
            assert placed.solid.volume == pytest.approx(
                builder(params).volume, rel=1e-9
            )

    def test_no_collisions_in_upper_stack_at_zero_pose(self, assembly):
        """
        Every pair of solids in the base stack, intersected.

        A positioning error anywhere in pedestal -> bearing -> turntable ->
        bracket -> upper arm shows up here as a non-zero intersection volume.
        """
        names = (
            "base_pedestal",
            "yaw turntable",
            "shoulder bracket",
            "shoulder idler plug",
            "L1 upper arm",
        )
        solids = {name: assembly.by_name(name).solid for name in names}
        for first, second in itertools.combinations(names, 2):
            overlap = (solids[first] & solids[second]).volume
            assert overlap == pytest.approx(0.0, abs=1e-6), (
                f"{first} intersects {second} by {overlap:.3f} mm3 at the "
                "zero pose"
            )

    def test_upper_stack_is_continuous_from_turret_to_elbow(self, assembly):
        """
        Nothing floats. Each part's underside meets the one below it, with the
        bearing's proud height the only deliberate gap.
        """
        pedestal = PedestalParameters.from_geometry()
        turntable = assembly.by_name("yaw turntable").solid.bounding_box()
        bracket = assembly.by_name("shoulder bracket").solid.bounding_box()

        # The spigot reaches down to the bearing seat's floor ...
        assert turntable.min.Z == pytest.approx(pedestal.servo_shaft_output_z_mm)
        # ... and the plate's top face is where the bracket stands.
        assert bracket.min.Z == pytest.approx(
            pedestal.bearing_top_z_mm
            + DEFAULT_HARDWARE.yaw_turntable.thickness_mm
        )
        assert bracket.min.Z - pedestal.turret_top_z_mm == pytest.approx(
            DEFAULT_HARDWARE.thrust_bearing.proud_mm
            + DEFAULT_HARDWARE.yaw_turntable.thickness_mm
        )

    def test_upper_arm_is_centred_on_the_yaw_axis(self, assembly):
        """
        What the yoke bought. A single-flange mount would have put this whole
        box roughly 50 mm to one side of the yaw axis.
        """
        box = assembly.by_name("L1 upper arm").solid.bounding_box()
        centre_x = (box.min.X + box.max.X) / 2.0
        assert centre_x == pytest.approx(DEFAULT_ARM.base_x_on_desk_mm, abs=2.5)

    def test_upper_arm_pivots_on_the_shoulder_axis(self, assembly):
        box = assembly.by_name("L1 upper arm").solid.bounding_box()
        # In the desk frame the link runs along +Y from the shoulder.
        assert box.max.Y == pytest.approx(
            DEFAULT_ARM.base_y_on_desk_mm
            + UpperArmParameters.from_geometry().housing_x_max_mm
        )
        assert assembly.elbow_pivot_xyz_mm == pytest.approx(
            (
                DEFAULT_ARM.base_x_on_desk_mm,
                DEFAULT_ARM.base_y_on_desk_mm + DEFAULT_ARM.l1_upper_arm_mm,
                DEFAULT_ARM.base_height_mm,
            )
        )

    def test_placeholders_are_only_the_links_d3_has_yet_to_design(self, assembly):
        scenery = {part.name for part in assembly.parts if not part.printed}
        assert {"L2 forearm", "L3 wrist-to-TCP"} <= scenery
        assert "L1 upper arm" not in scenery
        assert "yaw turntable" not in scenery
        assert "shoulder bracket" not in scenery

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
        """
        Zero pose is horizontal along the base frame's +X, i.e. desk +Y.

        L1's bounding box now reaches *behind* the shoulder axis, because its
        yoke wraps back around the bracket to reach the servo horn. What has to
        land on the axis is the joint, so that is what is checked -- the elbow
        end, one link length along.
        """
        base_y = DEFAULT_ARM.base_y_on_desk_mm
        params = UpperArmParameters.from_geometry()
        link1 = assembly.by_name("L1 upper arm").solid.bounding_box()
        assert link1.min.Y == pytest.approx(
            base_y - params.flange_diameter_mm / 2.0, abs=0.2
        )
        assert base_y + DEFAULT_ARM.l1_upper_arm_mm < link1.max.Y
        link2 = assembly.by_name("L2 forearm").solid.bounding_box()
        assert link2.min.Y == pytest.approx(
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
        """
        The desk, the fasteners, the bearings and the two undesigned links are
        not parts we make. Everything else in the scene is, since Session D.2.
        """
        printed = {part.name for part in assembly.printed_parts}
        assert printed == {
            "base_pedestal",
            "pressure_foot",
            "knob",
            "yaw turntable",
            "shoulder bracket",
            "shoulder idler plug",
            "L1 upper arm",
        }
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
