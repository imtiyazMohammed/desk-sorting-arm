"""
Arm geometry definition — single source of truth for all downstream modules.

Every kinematics, planning, and visualization module MUST import ArmGeometry
from here rather than hardcoding link lengths. This guarantees that changing
a physical dimension propagates atomically across the entire codebase.

Coordinate convention
---------------------
- All lengths in millimeters (mm).
- All angles in radians internally; conversions to degrees are explicit.
- Arm base frame: origin at the center of the base rotation servo shaft,
  Z-axis pointing up, X-axis pointing "forward" into the desk work area
  when all joint angles are zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class JointLimit:
    """Mechanical joint limit in radians. Frozen so tests can trust invariants."""

    min_rad: float
    max_rad: float
    name: str

    def __post_init__(self) -> None:
        if self.min_rad >= self.max_rad:
            raise ValueError(
                f"Joint {self.name}: min_rad ({self.min_rad}) must be strictly "
                f"less than max_rad ({self.max_rad})."
            )

    def clamp(self, angle_rad: float) -> float:
        """Clamp an angle to the valid mechanical range."""
        return float(np.clip(angle_rad, self.min_rad, self.max_rad))

    def contains(self, angle_rad: float, tolerance_rad: float = 1e-6) -> bool:
        """True iff the angle is within joint limits (with numerical tolerance)."""
        return (
            self.min_rad - tolerance_rad
            <= angle_rad
            <= self.max_rad + tolerance_rad
        )

    @property
    def min_deg(self) -> float:
        return float(np.degrees(self.min_rad))

    @property
    def max_deg(self) -> float:
        return float(np.degrees(self.max_rad))


@dataclass(frozen=True)
class ArmGeometry:
    """
    Physical geometry of the 5-DOF desk-sorting arm.

    The kinematic chain (base to end-effector) is:
        Base yaw (Z) -> Shoulder pitch (Y) -> Elbow pitch (Y)
                     -> Wrist pitch (Y)    -> Wrist roll (X)

    All lengths in mm. All limits in radians.
    """

    # ---- Link lengths (mm) ---------------------------------------------------
    base_height_mm: float = 100.0   # Desk surface -> shoulder pivot
    l1_upper_arm_mm: float = 400.0  # Shoulder pivot -> elbow pivot
    l2_forearm_mm: float = 350.0    # Elbow pivot -> wrist pivot
    l3_wrist_to_tip_mm: float = 200.0  # Wrist pivot -> gripper tip (TCP)

    # ---- Desk workspace (mm) -------------------------------------------------
    desk_width_mm: float = 1200.0
    desk_depth_mm: float = 600.0
    # Base mount position on the desk (center of long edge).
    # Desk frame origin is one corner; base is placed on the near long edge,
    # centered along its length, offset "into" the desk by base_inset_mm.
    base_x_on_desk_mm: float = 600.0
    base_y_on_desk_mm: float = 0.0

    # ---- Joint limits --------------------------------------------------------
    # DS3218 servos physically travel ~270°. We restrict further to avoid
    # mechanical interference with adjacent links.
    # NOTE sign convention (see forward_kinematics.py docstring):
    #   +theta_2 rotates shoulder DOWN from horizontal. Therefore the practical
    #   operating range spans NEGATIVE angles (arm above horizontal) with a
    #   small positive allowance to dip below the desk edge.
    joint_limits: Tuple[JointLimit, ...] = field(
        default_factory=lambda: (
            JointLimit(np.radians(-135.0), np.radians(135.0), "base_yaw"),
            JointLimit(np.radians(-120.0), np.radians(15.0),  "shoulder_pitch"),
            JointLimit(np.radians(-135.0), np.radians(135.0), "elbow_pitch"),
            JointLimit(np.radians(-100.0), np.radians(100.0), "wrist_pitch"),
            JointLimit(np.radians(-135.0), np.radians(135.0), "wrist_roll"),
        )
    )

    # ---- Servo dynamic limits (from DS3218 datasheet) ------------------------
    # The DSServo datasheet gives 0.16 s/60° at 5 V and 0.14 s/60° at 6.8 V.
    # The 0.16 figure is used here (see DS3218_SPEC.speed_s_per_60deg_at_5v):
    # it is the 5 V number, not the 6.8 V number an earlier comment claimed,
    # and keeping the slower of the two is the conservative choice. Changing
    # it would shift every trajectory timing in docs/PROOF_OF_CONCEPT.md.
    #   0.16 s/60°  ->  radians(60) / 0.16  =  6.545 rad/s theoretical.
    # We derate to 60% for realistic loaded motion.
    servo_max_speed_rad_s: float = 6.545 * 0.60
    servo_max_accel_rad_s2: float = 30.0  # Empirical safe accel bound

    # ---- Mass budget (kg) ----------------------------------------------------
    # Estimates carried forward from docs/PROOF_OF_CONCEPT.md section 2.2:
    # 5 servos at 60 g plus roughly 300 g of PETG. Not yet weighed -- these
    # only feed the mount's tipping check, where they are used conservatively.
    estimated_arm_mass_kg: float = 0.625
    max_payload_kg: float = 0.100

    # ---- Derived / convenience ----------------------------------------------
    @property
    def total_reach_mm(self) -> float:
        """Maximum theoretical reach (fully extended arm)."""
        return self.l1_upper_arm_mm + self.l2_forearm_mm + self.l3_wrist_to_tip_mm

    @property
    def safe_reach_mm(self) -> float:
        """Recommended max operating radius (85% of full extension)."""
        return 0.85 * self.total_reach_mm

    @property
    def num_dof(self) -> int:
        return len(self.joint_limits)

    def worst_case_desk_reach_mm(self) -> float:
        """
        Distance from base to the farthest desk corner.
        Used to sanity-check that geometry can cover the entire workspace.
        """
        corners = np.array(
            [
                [0.0, 0.0],
                [self.desk_width_mm, 0.0],
                [0.0, self.desk_depth_mm],
                [self.desk_width_mm, self.desk_depth_mm],
            ]
        )
        base = np.array([self.base_x_on_desk_mm, self.base_y_on_desk_mm])
        distances = np.linalg.norm(corners - base, axis=1)
        return float(distances.max())

    def tipping_moment_nm(self, gravity_m_s2: float = 9.81) -> float:
        """
        Worst-case moment the base mount must resist, in newton-metres.

        Deliberately pessimistic: the entire arm mass plus a full payload is
        treated as concentrated at maximum reach, with the arm horizontal.
        The real centre of mass sits at roughly 40% of the length (see
        docs/PROOF_OF_CONCEPT.md section 2.2, which computes about 3.4 N.m
        on that basis), so this figure is close to double the honest estimate.

        A mount sized against this number has margin against the modelling
        error in the mass estimates themselves, which have not been weighed.
        """
        total_mass_kg = self.estimated_arm_mass_kg + self.max_payload_kg
        return float(total_mass_kg * (self.total_reach_mm / 1000.0) * gravity_m_s2)

    def coverage_report(self) -> str:
        """Human-readable geometry summary printed at startup."""
        worst = self.worst_case_desk_reach_mm()
        reachable = worst <= self.safe_reach_mm
        return (
            f"ArmGeometry summary\n"
            f"-------------------\n"
            f"  Base height          : {self.base_height_mm:.1f} mm\n"
            f"  Upper arm (L1)       : {self.l1_upper_arm_mm:.1f} mm\n"
            f"  Forearm  (L2)        : {self.l2_forearm_mm:.1f} mm\n"
            f"  Wrist->tip (L3)      : {self.l3_wrist_to_tip_mm:.1f} mm\n"
            f"  Total reach          : {self.total_reach_mm:.1f} mm\n"
            f"  Safe reach (85%)     : {self.safe_reach_mm:.1f} mm\n"
            f"  DOF                  : {self.num_dof}\n"
            f"  Worst-case corner    : {worst:.1f} mm from base\n"
            f"  Full-desk reachable? : {reachable}\n"
            f"  Tipping moment       : {self.tipping_moment_nm():.2f} N.m "
            f"(worst case, all mass at full reach)\n"
        )


# Default singleton used across the project.
DEFAULT_ARM = ArmGeometry()


# =========================================================================
# Hardware component specifications
# =========================================================================
#
# These live alongside ArmGeometry rather than in a separate module so that
# there remains exactly one file to read for "what are this robot's numbers".
# The CAD package (cad/) imports from here; it defines no dimensions of its
# own.
#
# VERIFICATION POLICY
# -------------------
# Every dataclass below tags which of its fields came from a manufacturer
# datasheet and which are placeholders awaiting physical measurement. Nothing
# here is a guess presented as fact: unverified values are enumerated in
# UNVERIFIED_FIELDS, surfaced by unverified_report(), and printed by
# `python3 -m src.geometry`. Parts generated from unverified values are
# geometrically valid but must not be printed for final assembly until the
# real component is measured.


@dataclass(frozen=True)
class ServoSpec:
    """
    Physical and electrical specification of one servo.

    The DS3218 defaults are transcribed from the DSServo product datasheet
    (dsservo.com). The body envelope, mass, gear ratio, torque and speed
    figures are datasheet-confirmed. The mounting-flange geometry and output
    shaft placement are NOT: the datasheet's dimensioned drawing is a raster
    image with no extractable numbers, and suppliers ship visually similar
    variants. Those fields carry standard-size-servo placeholders and are
    listed in :attr:`UNVERIFIED_FIELDS`.

    Angles are degrees at this boundary because that is how servo travel is
    universally specified; :attr:`travel_rad` converts for internal use.
    """

    name: str = "DS3218"

    # ---- Body envelope (datasheet-confirmed: "Size 40*20*40.5mm") ---------
    body_length_mm: float = 40.0
    body_width_mm: float = 20.0
    body_height_mm: float = 40.5
    mass_g: float = 60.0

    # ---- Electrical / dynamic (datasheet-confirmed) ----------------------
    voltage_min_v: float = 4.8
    voltage_max_v: float = 6.8
    stall_torque_kgcm_at_5v: float = 19.0
    stall_torque_kgcm_at_6v8: float = 21.5
    speed_s_per_60deg_at_5v: float = 0.16
    speed_s_per_60deg_at_6v8: float = 0.14
    gear_ratio: float = 236.0

    # ---- Mounting flange (PLACEHOLDER - measure before printing) ---------
    # Standard-size-servo values. The flange is what the pedestal's retention
    # shelf is cut to, so these are the fields that most need real numbers.
    flange_span_mm: float = 54.0
    flange_thickness_mm: float = 2.5
    flange_hole_spacing_long_mm: float = 49.5
    flange_hole_spacing_short_mm: float = 10.0
    flange_hole_diameter_mm: float = 3.0

    # ---- Output shaft (PLACEHOLDER - measure before printing) ------------
    # Distance from one end of the body to the output shaft axis. Drives how
    # far the servo body must be offset inside the pedestal for its shaft to
    # land on the yaw axis, so an error here shifts the whole cavity.
    shaft_offset_from_body_end_mm: float = 10.0
    shaft_boss_diameter_mm: float = 13.0
    shaft_boss_height_mm: float = 4.0
    output_shaft_diameter_mm: float = 5.8

    # ---- Travel (PLACEHOLDER - VARIANT-DEPENDENT) ------------------------
    # DS3218 ships in 180-degree and 270-degree variants that are otherwise
    # near-identical. Confirm which one was ordered: it bounds the achievable
    # base yaw range against the +/-135 degree joint limit above.
    travel_deg: float = 270.0

    #: Fields whose values have NOT been confirmed against a datasheet or a
    #: physical measurement. Kept machine-readable so tests can assert that
    #: the warning is present rather than trusting a comment.
    UNVERIFIED_FIELDS: ClassVar[Tuple[str, ...]] = (
        "flange_span_mm",
        "flange_thickness_mm",
        "flange_hole_spacing_long_mm",
        "flange_hole_spacing_short_mm",
        "flange_hole_diameter_mm",
        "shaft_offset_from_body_end_mm",
        "shaft_boss_diameter_mm",
        "shaft_boss_height_mm",
        "output_shaft_diameter_mm",
        "travel_deg",
    )

    def __post_init__(self) -> None:
        for field_name in (
            "body_length_mm", "body_width_mm", "body_height_mm",
            "flange_span_mm", "flange_thickness_mm",
        ):
            if getattr(self, field_name) <= 0.0:
                raise ValueError(
                    f"ServoSpec {self.name}: {field_name} must be positive, "
                    f"got {getattr(self, field_name)}."
                )
        if self.shaft_offset_from_body_end_mm > self.body_length_mm:
            raise ValueError(
                f"ServoSpec {self.name}: shaft_offset_from_body_end_mm "
                f"({self.shaft_offset_from_body_end_mm}) cannot exceed "
                f"body_length_mm ({self.body_length_mm})."
            )
        if self.flange_span_mm < self.body_length_mm:
            raise ValueError(
                f"ServoSpec {self.name}: flange_span_mm ({self.flange_span_mm}) "
                f"must be at least body_length_mm ({self.body_length_mm}); the "
                "mounting ears extend beyond the body, they do not inset."
            )

    @property
    def travel_rad(self) -> float:
        """Total mechanical travel in radians."""
        return float(np.radians(self.travel_deg))

    @property
    def body_offset_from_shaft_axis_mm(self) -> float:
        """
        Lateral offset of the body centre from the output-shaft axis.

        When the servo drives a joint whose rotation axis must be the shaft
        axis, the body sits off-centre by this much. Positive means the body
        centre is displaced toward the far end from the shaft.
        """
        return self.body_length_mm / 2.0 - self.shaft_offset_from_body_end_mm

    def unverified_report(self) -> str:
        """Human-readable list of fields still awaiting physical measurement."""
        lines = [
            f"{self.name}: {len(self.UNVERIFIED_FIELDS)} UNVERIFIED dimension(s) "
            "- measure before printing for final assembly:"
        ]
        for field_name in self.UNVERIFIED_FIELDS:
            lines.append(f"    {field_name:<34} = {getattr(self, field_name)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class BearingSpec:
    """
    A rolling-element bearing, described by its standard bore/OD/width triple.

    The 608ZZ defaults are the ISO 15 dimension series for that designation
    (8 mm bore, 22 mm outer diameter, 7 mm width) - a hard standard, not a
    per-supplier value, so no verification flag is needed.
    """

    name: str = "608ZZ"
    bore_diameter_mm: float = 8.0
    outer_diameter_mm: float = 22.0
    width_mm: float = 7.0

    #: Radial interference for a printed press fit. The seat is cut this much
    #: SMALLER than outer_diameter_mm so the outer race grips.
    press_fit_interference_mm: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.bore_diameter_mm < self.outer_diameter_mm:
            raise ValueError(
                f"BearingSpec {self.name}: require "
                f"0 < bore ({self.bore_diameter_mm}) < OD "
                f"({self.outer_diameter_mm})."
            )
        if self.width_mm <= 0.0:
            raise ValueError(
                f"BearingSpec {self.name}: width_mm must be positive, "
                f"got {self.width_mm}."
            )

    @property
    def seat_diameter_mm(self) -> float:
        """Diameter of the pocket the outer race presses into."""
        return self.outer_diameter_mm - self.press_fit_interference_mm


@dataclass(frozen=True)
class FastenerSpec:
    """
    A metric fastener and the hole geometry it needs.

    The M4 defaults are standards, not supplier-specific values: clearance
    hole from ISO 273 medium series (4.5 mm), head envelope from ISO 4762
    socket head cap screw (7.0 mm across, 4.0 mm tall).
    """

    name: str = "M4"
    nominal_diameter_mm: float = 4.0
    clearance_hole_diameter_mm: float = 4.5
    head_diameter_mm: float = 7.0
    head_height_mm: float = 4.0

    def __post_init__(self) -> None:
        if self.clearance_hole_diameter_mm < self.nominal_diameter_mm:
            raise ValueError(
                f"FastenerSpec {self.name}: clearance hole "
                f"({self.clearance_hole_diameter_mm}) must be at least the "
                f"nominal diameter ({self.nominal_diameter_mm})."
            )
        if self.head_diameter_mm <= self.clearance_hole_diameter_mm:
            raise ValueError(
                f"FastenerSpec {self.name}: head diameter "
                f"({self.head_diameter_mm}) must exceed the clearance hole "
                f"({self.clearance_hole_diameter_mm}), or the head pulls through."
            )


@dataclass(frozen=True)
class BaseStack:
    """
    Vertical budget between the desk surface and the shoulder pivot.

    ``ArmGeometry.base_height_mm`` (100 mm) is defined as desk surface ->
    shoulder pivot. The base pedestal is only the bottom of that stack: a yaw
    turntable plate and a shoulder bracket sit on top of it. Splitting the
    budget explicitly here keeps FK's z = base_height_mm shoulder pivot exact
    by construction, instead of discovering after printing that the real
    pivot ended up 30 mm high.

    Both components are PROVISIONAL. They are refined in Sessions D.2 (yaw
    turntable) and D.3 (shoulder bracket), when those parts actually exist.
    Raising either one shortens the pedestal automatically.
    """

    turntable_plate_thickness_mm: float = 6.0
    shoulder_bracket_rise_mm: float = 24.0

    def __post_init__(self) -> None:
        if self.turntable_plate_thickness_mm <= 0.0:
            raise ValueError(
                "BaseStack: turntable_plate_thickness_mm must be positive, got "
                f"{self.turntable_plate_thickness_mm}."
            )
        if self.shoulder_bracket_rise_mm <= 0.0:
            raise ValueError(
                "BaseStack: shoulder_bracket_rise_mm must be positive, got "
                f"{self.shoulder_bracket_rise_mm}."
            )

    @property
    def allowance_mm(self) -> float:
        """Total height consumed above the pedestal, in mm."""
        return self.turntable_plate_thickness_mm + self.shoulder_bracket_rise_mm


@dataclass(frozen=True)
class DeskClampSpec:
    """
    G-clamp mount that grips the desk edge, replacing the drilled-flange mount.

    The arm is clamped rather than bolted so the desk is never drilled and the
    whole assembly can be repositioned or removed. An upper jaw (part of the
    pedestal) sits on the desk top, a separate lower jaw goes underneath, and
    a captive-nut M8 screw driven by a printed knob pulls them together.

    Fastener dimensions are standards, verified rather than assumed:

    - M8 coarse thread pitch 1.25 mm.
    - Hex head, DIN 933 / ISO 4017: 13.00 mm across flats, 5.30 mm tall.
    - Hex nut across flats, DIN 934: 13.00 mm (max; 12.73 min).
    - Clearance hole, ISO 273 medium series for M8: 9.0 mm.

    .. note::
       **Nut thickness is version-dependent and the pocket uses the larger
       value.** Legacy DIN 934 tables give m = 6.5 mm max for M8; the current
       DIN EN ISO 4032 revision gives m = 6.80 max / 6.44 min. Both are sold
       as "DIN 934". A pocket cut to 6.5 mm would not seat a modern nut, so
       :attr:`nut_pocket_depth_mm` derives from
       :attr:`nut_thickness_max_mm` (6.80). The 6.5 figure is retained as
       :attr:`nut_thickness_nominal_mm` for reference only.
    """

    # ---- Desk compatibility -------------------------------------------------
    #: (min, max) desk thickness the clamp is designed to grip, in mm.
    desk_thickness_range_mm: Tuple[float, float] = (15.0, 35.0)

    #: Maximum jaw gap, in mm. Exceeds the thickest supported desk so the
    #: clamp can be slid on and off without fully unthreading the nut.
    throat_max_opening_mm: float = 45.0

    # ---- Jaw contact geometry -----------------------------------------------
    #: (length, width) of the upper jaw's anti-slip pad area, in mm.
    upper_jaw_contact_mm: Tuple[float, float] = (50.0, 50.0)

    #: (length, width) of the lower jaw's anti-slip pad area, in mm.
    lower_jaw_contact_mm: Tuple[float, float] = (40.0, 40.0)

    #: Structural thickness of each jaw plate, in mm. Recesses and pockets are
    #: cut IN ADDITION to this, so it is the material that actually carries
    #: load -- see :attr:`upper_jaw_total_thickness_mm`.
    jaw_thickness_mm: float = 10.0

    #: Depth of the anti-slip pad recesses, in mm.
    pad_recess_depth_mm: float = 2.0

    #: Thickness of the rubber sheet glued into those recesses, in mm.
    pad_thickness_mm: float = 2.0

    # ---- Clamping screw -----------------------------------------------------
    bolt_thread: str = "M8"
    bolt_nominal_diameter_mm: float = 8.0
    bolt_thread_pitch_mm: float = 1.25
    bolt_length_mm: float = 90.0
    #: ISO 273 medium series clearance hole for M8.
    bolt_clearance_hole_diameter_mm: float = 9.0
    #: DIN 933 / ISO 4017 hex head, across flats (max) and height.
    bolt_head_across_flats_mm: float = 13.0
    bolt_head_height_mm: float = 5.30

    # ---- Captive nut --------------------------------------------------------
    nut_across_flats_mm: float = 13.0
    #: Legacy DIN 934 figure, kept for reference. NOT used for the pocket.
    nut_thickness_nominal_mm: float = 6.50
    #: DIN EN ISO 4032 maximum. The pocket is cut from this.
    nut_thickness_max_mm: float = 6.80

    # ---- Hand knob ----------------------------------------------------------
    knob_diameter_mm: float = 50.0
    knob_thickness_mm: float = 20.0
    #: How far below the knob's top face the bolt head is buried, in mm. Sunk
    #: deliberately: it protects the head and shortens the bolt length needed.
    knob_head_recess_mm: float = 2.50
    #: Number of grip flutes cut around the knob's perimeter.
    knob_flute_count: int = 12
    #: Diameter and height of the small bearing boss under the knob. Keeping
    #: the rubbing contact on a small radius is what stops collar friction
    #: from swallowing most of the hand torque -- see :meth:`bolt_preload_n`.
    knob_boss_diameter_mm: float = 18.0
    knob_boss_height_mm: float = 2.0

    # ---- Friction coefficients ----------------------------------------------
    #: Anti-slip pad against the desk. 0.4 is the PETG-on-wood figure, used
    #: deliberately even though the pad is rubber (which is higher), so the
    #: slip margin is understated rather than overstated.
    pad_friction_coefficient: float = 0.40
    #: Steel bolt thread in a steel nut, dry.
    thread_friction_coefficient: float = 0.15
    #: Printed knob boss bearing on the printed upper jaw.
    collar_friction_coefficient: float = 0.30

    def __post_init__(self) -> None:
        low, high = self.desk_thickness_range_mm
        if not 0.0 < low < high:
            raise ValueError(
                f"DeskClampSpec: desk_thickness_range_mm must be an increasing "
                f"positive pair, got {self.desk_thickness_range_mm}."
            )
        if self.throat_max_opening_mm <= high:
            raise ValueError(
                f"DeskClampSpec: throat_max_opening_mm "
                f"({self.throat_max_opening_mm}) must exceed the thickest "
                f"supported desk ({high}), or the clamp cannot be slid on."
            )
        for name in (
            "jaw_thickness_mm", "pad_recess_depth_mm", "pad_thickness_mm",
            "bolt_nominal_diameter_mm", "bolt_thread_pitch_mm", "bolt_length_mm",
            "knob_diameter_mm", "knob_thickness_mm", "knob_boss_diameter_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"DeskClampSpec: {name} must be positive, got "
                    f"{getattr(self, name)}."
                )
        if self.pad_thickness_mm > self.pad_recess_depth_mm:
            raise ValueError(
                f"DeskClampSpec: pad_thickness_mm ({self.pad_thickness_mm}) "
                f"exceeds pad_recess_depth_mm ({self.pad_recess_depth_mm}); "
                "the pad would stand proud and rock on the desk."
            )
        if self.nut_thickness_max_mm < self.nut_thickness_nominal_mm:
            raise ValueError(
                "DeskClampSpec: nut_thickness_max_mm must not be less than "
                "nut_thickness_nominal_mm."
            )
        if self.bolt_clearance_hole_diameter_mm <= self.bolt_nominal_diameter_mm:
            raise ValueError(
                f"DeskClampSpec: bolt clearance hole "
                f"({self.bolt_clearance_hole_diameter_mm}) must exceed the "
                f"nominal diameter ({self.bolt_nominal_diameter_mm})."
            )
        if self.knob_boss_diameter_mm >= self.knob_diameter_mm:
            raise ValueError(
                "DeskClampSpec: knob_boss_diameter_mm must be smaller than "
                "knob_diameter_mm."
            )
        if self.knob_boss_diameter_mm <= self.bolt_clearance_hole_diameter_mm:
            raise ValueError(
                f"DeskClampSpec: knob_boss_diameter_mm "
                f"({self.knob_boss_diameter_mm}) must exceed the bolt "
                f"clearance hole ({self.bolt_clearance_hole_diameter_mm}), or "
                "the boss is not an annulus."
            )
        if self.knob_flute_count < 3:
            raise ValueError(
                "DeskClampSpec: knob_flute_count must be at least 3, got "
                f"{self.knob_flute_count}."
            )

    # ---- Derived geometry ---------------------------------------------------

    @property
    def min_desk_thickness_mm(self) -> float:
        return self.desk_thickness_range_mm[0]

    @property
    def max_desk_thickness_mm(self) -> float:
        return self.desk_thickness_range_mm[1]

    @property
    def desk_removal_clearance_mm(self) -> float:
        """Extra throat beyond the thickest desk, for sliding the clamp on."""
        return self.throat_max_opening_mm - self.max_desk_thickness_mm

    @property
    def nut_pocket_depth_mm(self) -> float:
        """Pocket depth for the captive nut. Sized from the MAX nut thickness."""
        return self.nut_thickness_max_mm

    @property
    def nut_across_corners_mm(self) -> float:
        """Hexagon across-corners, 2/sqrt(3) times across-flats."""
        return float(self.nut_across_flats_mm * 2.0 / np.sqrt(3.0))

    @property
    def upper_jaw_total_thickness_mm(self) -> float:
        """Structural thickness plus the pad recess cut into its underside."""
        return self.jaw_thickness_mm + self.pad_recess_depth_mm

    @property
    def lower_jaw_total_thickness_mm(self) -> float:
        """
        Structural thickness plus a pad recess on top and a nut pocket below.

        Both recesses are additional to :attr:`jaw_thickness_mm` because at
        10 mm total the 2 mm pad recess and 6.8 mm nut pocket would leave only
        1.2 mm between them.
        """
        return (
            self.jaw_thickness_mm
            + self.pad_recess_depth_mm
            + self.nut_pocket_depth_mm
        )

    @property
    def knob_socket_depth_mm(self) -> float:
        """Depth of the hex socket in the knob's top face."""
        return self.bolt_head_height_mm + self.knob_head_recess_mm

    @property
    def required_bolt_length_mm(self) -> float:
        """
        Shortest bolt that reaches the nut at maximum throat opening.

        Stack, from the underside of the bolt head downward: the knob material
        below the head, the upper jaw, the open throat, the lower jaw above
        its nut pocket, and full engagement in the nut.
        """
        knob_below_head = self.knob_thickness_mm - self.knob_socket_depth_mm
        lower_jaw_above_pocket = (
            self.lower_jaw_total_thickness_mm - self.nut_pocket_depth_mm
        )
        return float(
            knob_below_head
            + self.upper_jaw_total_thickness_mm
            + self.throat_max_opening_mm
            + lower_jaw_above_pocket
            + self.nut_thickness_max_mm
        )

    # ---- Clamping physics ---------------------------------------------------

    def torque_to_preload_factor_m(self) -> float:
        """
        Metres of effective lever between applied torque and bolt preload.

        Preload is ``torque / factor``. Built from the power-screw relation
        rather than a lumped nut factor, so the thread pitch actually appears:

            T = F * [ d2/2 * (p + pi*mu_t*d2*sec(a)) / (pi*d2 - mu_t*p*sec(a))
                      + mu_c * r_c ]

        with ``d2`` the pitch diameter (``d - 0.6495*p`` for ISO metric),
        ``a = 30 deg`` the thread half-angle, and ``r_c`` the mean radius of
        the knob's bearing boss. The collar term dominates, which is why the
        boss is kept small.
        """
        pitch_diameter = (
            self.bolt_nominal_diameter_mm - 0.6495 * self.bolt_thread_pitch_mm
        )
        sec_half_angle = 1.0 / np.cos(np.radians(30.0))
        mu_t = self.thread_friction_coefficient

        numerator = (
            self.bolt_thread_pitch_mm
            + np.pi * mu_t * pitch_diameter * sec_half_angle
        )
        denominator = (
            np.pi * pitch_diameter
            - mu_t * self.bolt_thread_pitch_mm * sec_half_angle
        )
        thread_term_mm = (pitch_diameter / 2.0) * (numerator / denominator)

        collar_radius_mm = (
            self.knob_boss_diameter_mm / 2.0
            + self.bolt_clearance_hole_diameter_mm / 2.0
        ) / 2.0
        collar_term_mm = self.collar_friction_coefficient * collar_radius_mm

        return float((thread_term_mm + collar_term_mm) / 1000.0)

    def bolt_preload_n(self, torque_nm: float) -> float:
        """
        Axial clamping force produced by ``torque_nm`` at the knob, in newtons.

        Raises
        ------
        ValueError
            If ``torque_nm`` is negative.
        """
        if torque_nm < 0.0:
            raise ValueError(f"torque_nm must be non-negative, got {torque_nm}.")
        return float(torque_nm / self.torque_to_preload_factor_m())

    def preload_to_torque_nm(self, preload_n: float) -> float:
        """Inverse of :meth:`bolt_preload_n`: torque needed for a given force."""
        if preload_n < 0.0:
            raise ValueError(f"preload_n must be non-negative, got {preload_n}.")
        return float(preload_n * self.torque_to_preload_factor_m())

    def pad_friction_force_n(self, preload_n: float) -> float:
        """
        Lateral force the pads resist before the clamp slides, in newtons.

        Simple Coulomb friction at the pad/desk interface. This is the
        sliding check; tipping is handled by
        :meth:`clamp_resisting_moment_nm`.
        """
        if preload_n < 0.0:
            raise ValueError(f"preload_n must be non-negative, got {preload_n}.")
        return float(self.pad_friction_coefficient * preload_n)

    def clamp_resisting_moment_nm(
        self, torque_nm: float, lever_arm_mm: float
    ) -> float:
        """
        Overturning moment the clamp resists, in newton-metres.

        The arm's tipping moment tries to lift the clamped side. The bolt
        preload holds it down, acting at ``lever_arm_mm`` from the pivot --
        taken as the inboard edge of the upper jaw's pad, the last line of
        contact the assembly would rotate about.

        Raises
        ------
        ValueError
            If ``lever_arm_mm`` is not positive.
        """
        if lever_arm_mm <= 0.0:
            raise ValueError(
                f"lever_arm_mm must be positive, got {lever_arm_mm}."
            )
        return float(self.bolt_preload_n(torque_nm) * (lever_arm_mm / 1000.0))

    def jaw_allowable_preload_n(
        self,
        overhang_mm: float,
        allowable_stress_mpa: float = 25.0,
    ) -> float:
        """
        Preload the upper jaw can carry before bending failure, in newtons.

        The jaw overhangs the desk edge by ``overhang_mm`` with the bolt load
        at its tip, so it is a cantilever of rectangular section: the section
        modulus is ``width * thickness^2 / 6`` and the allowable load is
        ``stress * modulus / overhang``.

        The 25 MPa default is roughly half of PETG's ~50 MPa tensile yield,
        i.e. a safety factor of 2 against a printed part whose layer adhesion
        is weaker than bulk material.

        This is the real limit on how hard the knob may be tightened, and it
        is well below what a hand can apply -- see ``cad/README.md``.
        """
        if overhang_mm <= 0.0:
            raise ValueError(f"overhang_mm must be positive, got {overhang_mm}.")
        width_mm = self.upper_jaw_contact_mm[1]
        section_modulus_mm3 = width_mm * self.jaw_thickness_mm**2 / 6.0
        return float(allowable_stress_mpa * section_modulus_mm3 / overhang_mm)

    def max_tightening_torque_nm(
        self, overhang_mm: float, allowable_stress_mpa: float = 25.0
    ) -> float:
        """Hand torque at which the upper jaw reaches its allowable stress."""
        return self.preload_to_torque_nm(
            self.jaw_allowable_preload_n(overhang_mm, allowable_stress_mpa)
        )


@dataclass(frozen=True)
class HardwareSpec:
    """
    Every off-the-shelf component the CAD needs, in one place.

    Held separately from :class:`ArmGeometry` because these are bill-of-
    materials facts rather than kinematic parameters, but kept in this module
    so downstream code still has a single import for "the robot's numbers".
    """

    base_yaw_servo: ServoSpec = field(default_factory=ServoSpec)
    thrust_bearing: BearingSpec = field(default_factory=BearingSpec)
    base_stack: BaseStack = field(default_factory=BaseStack)

    #: Desk-edge clamp. Replaced the drilled M4 flange in Session D.1b: the
    #: user does not want holes in the desk, and a clamp is repositionable.
    desk_clamp: DeskClampSpec = field(default_factory=DeskClampSpec)

    #: Per-surface gap added to printed pockets so parts actually fit. FDM
    #: prints slightly undersize on internal features; 0.25 mm per wall is a
    #: conservative starting point for PETG on a well-tuned machine.
    print_clearance_mm: float = 0.25

    #: Minimum printed wall thickness. Four 0.4 mm perimeters plus margin.
    min_wall_thickness_mm: float = 4.0

    def __post_init__(self) -> None:
        if self.print_clearance_mm < 0.0:
            raise ValueError(
                "HardwareSpec: print_clearance_mm must be non-negative, got "
                f"{self.print_clearance_mm}."
            )
        if self.min_wall_thickness_mm <= 0.0:
            raise ValueError(
                "HardwareSpec: min_wall_thickness_mm must be positive, got "
                f"{self.min_wall_thickness_mm}."
            )

    def pedestal_height_mm(self, arm: Optional[ArmGeometry] = None) -> float:
        """
        Height of the base pedestal alone, in mm.

        Derived as ``arm.base_height_mm - base_stack.allowance_mm`` so that
        the pedestal plus everything above it lands the shoulder pivot at
        exactly ``base_height_mm``.

        Raises
        ------
        ValueError
            If the stack allowance consumes the entire base height, which
            would call for a pedestal of zero or negative height.
        """
        arm = DEFAULT_ARM if arm is None else arm
        height = arm.base_height_mm - self.base_stack.allowance_mm
        if height <= 0.0:
            raise ValueError(
                f"Base stack allowance ({self.base_stack.allowance_mm:.1f} mm) "
                f"meets or exceeds base_height_mm ({arm.base_height_mm:.1f} mm); "
                "the pedestal would have non-positive height. Either shorten "
                "the turntable/bracket budget or raise base_height_mm."
            )
        return float(height)

    def bill_of_materials(self) -> Tuple[str, ...]:
        """Off-the-shelf parts to buy, one human-readable line each."""
        clamp = self.desk_clamp
        return (
            f"1 x {self.base_yaw_servo.name} servo (base yaw)",
            f"1 x {self.thrust_bearing.name} bearing "
            f"({self.thrust_bearing.bore_diameter_mm:.0f} x "
            f"{self.thrust_bearing.outer_diameter_mm:.0f} x "
            f"{self.thrust_bearing.width_mm:.0f} mm)",
            f"1 x {clamp.bolt_thread} x {clamp.bolt_length_mm:.0f} mm hex-head "
            f"machine screw (DIN 933)",
            f"1 x {clamp.bolt_thread} hex nut (DIN 934), "
            f"{clamp.nut_across_flats_mm:.0f} mm across flats",
            f"1 x rubber anti-slip sheet, {clamp.pad_thickness_mm:.1f} mm thick "
            f"(cut 2 pads, glued into the jaw recesses)",
            "4 x M3 screws (servo retention)",
        )

    def hardware_report(self, arm: Optional[ArmGeometry] = None) -> str:
        """Human-readable component summary, including verification warnings."""
        arm = DEFAULT_ARM if arm is None else arm
        servo, bearing, clamp = (
            self.base_yaw_servo, self.thrust_bearing, self.desk_clamp
        )
        return (
            f"HardwareSpec summary\n"
            f"--------------------\n"
            f"  Base yaw servo       : {servo.name} "
            f"{servo.body_length_mm:.1f} x {servo.body_width_mm:.1f} x "
            f"{servo.body_height_mm:.1f} mm, {servo.mass_g:.0f} g\n"
            f"  Thrust bearing       : {bearing.name} "
            f"{bearing.bore_diameter_mm:.0f} ID / "
            f"{bearing.outer_diameter_mm:.0f} OD / "
            f"{bearing.width_mm:.0f} W mm\n"
            f"  Desk mount           : {clamp.bolt_thread} desk clamp, "
            f"throat {clamp.throat_max_opening_mm:.0f} mm for "
            f"{clamp.min_desk_thickness_mm:.0f}-"
            f"{clamp.max_desk_thickness_mm:.0f} mm desks\n"
            f"  Clamp screw          : {clamp.bolt_thread} x "
            f"{clamp.bolt_length_mm:.0f} mm (needs "
            f"{clamp.required_bolt_length_mm:.1f} mm)\n"
            f"  Base height budget   : {arm.base_height_mm:.1f} mm total\n"
            f"    turntable plate    : "
            f"{self.base_stack.turntable_plate_thickness_mm:.1f} mm\n"
            f"    shoulder bracket   : "
            f"{self.base_stack.shoulder_bracket_rise_mm:.1f} mm\n"
            f"    -> pedestal        : {self.pedestal_height_mm(arm):.1f} mm\n"
            f"  Print clearance      : {self.print_clearance_mm:.2f} mm/surface\n"
            f"  Min wall thickness   : {self.min_wall_thickness_mm:.1f} mm\n"
            f"\n"
            f"  Tipping moment       : {arm.tipping_moment_nm():.2f} N.m\n"
            f"  Preload at 5 N.m     : {clamp.bolt_preload_n(5.0):.0f} N\n"
            f"\n"
            f"  Bill of materials\n"
            + "".join(f"    - {line}\n" for line in self.bill_of_materials())
            + f"\n"
            f"  !! {servo.unverified_report()}\n"
        )


# Default hardware singleton, paired with DEFAULT_ARM.
DEFAULT_HARDWARE = HardwareSpec()


if __name__ == "__main__":
    print(DEFAULT_ARM.coverage_report())
    print(DEFAULT_HARDWARE.hardware_report())
