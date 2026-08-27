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
    # Desk frame origin is one corner; the base sits on the near long edge,
    # centered along its length. The Y offset is not a free choice: the U-clamp
    # puts the yaw axis DeskClampSpec.servo_shaft_offset_from_edge_mm inward of
    # the desk edge, and this is that same measurement in the desk frame. The
    # two must agree, which tests/test_cad.py asserts.
    base_x_on_desk_mm: float = 600.0
    base_y_on_desk_mm: float = 30.0

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

    #: Fraction of full extension treated as the recommended operating limit.
    #:
    #: Raised from 0.85 to 0.88 in Session D.1d. A serial arm loses reach
    #: rapidly near full extension because the Jacobian goes rank-deficient, so
    #: this is a caution line rather than a hard limit -- but at 0.85 it sat
    #: BELOW the desk's far corners, which is not a caution, it is a
    #: contradiction: the arm was designed to reach them.
    #:
    #: D.1c moved the base 30 mm inward to clear the U-clamp, improving the
    #: worst corner from 848.5 mm to 827.6 mm -- 87.1% of full extension. The
    #: old 0.85 (807.5 mm) could not cover that; 0.88 (836.0 mm) does, with
    #: 8.4 mm to spare, and still sits inside the 0.89 the links were sized
    #: against in docs/PROOF_OF_CONCEPT.md section 2.1.
    SAFE_REACH_FRACTION: ClassVar[float] = 0.88

    @property
    def safe_reach_mm(self) -> float:
        """
        Recommended maximum operating radius, in mm.

        See :attr:`SAFE_REACH_FRACTION` for why the fraction is what it is.
        """
        return self.SAFE_REACH_FRACTION * self.total_reach_mm

    @property
    def num_dof(self) -> int:
        return len(self.joint_limits)

    def worst_case_desk_reach_mm(self) -> float:
        """
        Distance from base to the farthest desk corner.

        Used to sanity-check that geometry can cover the entire workspace.

        Note that this is compared against two different thresholds elsewhere,
        and they disagree. :attr:`safe_reach_mm` is 85% of full extension, a
        conservative round number; docs/PROOF_OF_CONCEPT.md section 2.1 sized
        the arm against 89%, which is the figure the link lengths were actually
        chosen for. The far corners land between the two, so
        :meth:`coverage_report` reports "not reachable" while the arm is in
        fact within its design envelope. Treat the 85% line as a caution, not
        a limit.
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

    #: Where the arm's own centre of mass sits, as a fraction of full reach.
    #: docs/PROOF_OF_CONCEPT.md section 2.2 took it at 400 mm from the
    #: shoulder; 0.40 of the 950 mm reach is 380 mm, the same figure to within
    #: 5%, and it tracks the link lengths instead of restating one of them.
    ARM_COM_FRACTION: ClassVar[float] = 0.40

    def shoulder_moment_nm(self, gravity_m_s2: float = 9.81) -> float:
        """
        Static moment about the shoulder pitch axis, in newton-metres.

        The arm horizontal and fully extended, holding a full payload: the
        payload acting at full reach plus the arm's own mass acting at
        :attr:`ARM_COM_FRACTION` of it. This is the load every structural
        member distal to the shoulder has to carry, and it is also the torque
        the shoulder servo has to hold - see docs/PROOF_OF_CONCEPT.md section
        2.2, which records that it exceeds the DS3218's rating.

        Distinct from :meth:`tipping_moment_nm`, which is deliberately
        pessimistic for sizing the desk mount. This one is the honest estimate,
        because using a doubled load to size a link would just make the arm
        heavier and the shoulder shortfall worse.
        """
        reach_m = self.total_reach_mm / 1000.0
        payload = self.max_payload_kg * reach_m * gravity_m_s2
        own_mass = (
            self.estimated_arm_mass_kg
            * (self.ARM_COM_FRACTION * reach_m)
            * gravity_m_s2
        )
        return float(payload + own_mass)

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
            f"  Safe reach ({self.SAFE_REACH_FRACTION * 100:.0f}%)     : "
            f"{self.safe_reach_mm:.1f} mm\n"
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


#: Tensile stress at yield for PETG, in MPa. Manufacturer data sheets cluster
#: on 50 (Fillamentum and Rigid Ink both publish exactly 50 to ISO 527;
#: 3D4Makers publishes 53), with the spread across the market running roughly
#: 34-53 depending on formulation and print orientation. 50 is the figure the
#: clamp was sized against in Session D.1 and it is kept here so every part
#: quotes one number.
PETG_TENSILE_YIELD_MPA: float = 50.0

#: Safety factor applied to :data:`PETG_TENSILE_YIELD_MPA` for printed
#: structural parts. Two, because layer adhesion is weaker than bulk material
#: and none of these parts has been tested.
STRUCTURAL_SAFETY_FACTOR: float = 2.0

#: Working stress for printed structural parts, in MPa.
PETG_ALLOWABLE_STRESS_MPA: float = PETG_TENSILE_YIELD_MPA / STRUCTURAL_SAFETY_FACTOR

#: Density of PETG, g/cm^3. Used for print-mass estimates.
PETG_DENSITY_G_CM3: float = 1.27


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
    #: How far below the shaft-end face the mounting ears sit, in mm. Every
    #: part that holds a servo needs it: the pedestal cuts its retention shelf
    #: at this depth, and the shoulder bracket sets its wall spacing from it.
    #: It lived as a constructor default in cad/base_pedestal.py until Session
    #: D.2b needed the same number in a second place.
    ear_offset_from_shaft_face_mm: float = 10.0
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
        "ear_offset_from_shaft_face_mm",
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
        if self.ear_offset_from_shaft_face_mm >= self.body_height_mm:
            raise ValueError(
                f"ServoSpec {self.name}: ear_offset_from_shaft_face_mm "
                f"({self.ear_offset_from_shaft_face_mm}) must be less than "
                f"body_height_mm ({self.body_height_mm}); the ears are on the "
                "case, not behind it."
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
    def body_depth_behind_ears_mm(self) -> float:
        """
        Body remaining behind the mounting ears, in mm.

        Measured along the shaft axis: the case is
        :attr:`body_height_mm` deep and the ears sit
        :attr:`ear_offset_from_shaft_face_mm` in from its shaft-end face, so
        this is what a bracket has to leave room for behind its mounting wall.
        """
        return self.body_height_mm - self.ear_offset_from_shaft_face_mm

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

    The 6806ZZ defaults are the standard dimensions for that designation
    (30 mm bore, 42 mm outer diameter, 7 mm width) - a hard standard, not a
    per-supplier value, so the triple needs no verification flag.

    .. note::
       **The yaw bearing was a 608ZZ until Session D.2a.** It could not stay:
       the 608's 8 mm bore is the only path between the servo's output shaft
       and the turntable above it, and no DS3218 horn fits through it. A 25T
       round horn is 19.7-25 mm across, so the coupling had nowhere to live -
       and nothing gripped the 608's inner race, which meant the "bearing" was
       two washers with balls between them.

       The 6806 is the smallest standard bearing whose bore clears the horn.
       It is 7 mm wide, exactly like the 608, so the entire vertical budget in
       :class:`BaseStack` is untouched by the swap. Its mean race diameter is
       36 mm rather than 15 mm, so the arm's overturning moment produces about
       2.4x less force on the races as a side benefit.

       :data:`BEARING_608ZZ` survives as the shoulder yoke's idler.
    """

    name: str = "6806ZZ"
    bore_diameter_mm: float = 30.0
    outer_diameter_mm: float = 42.0
    width_mm: float = 7.0

    #: Radial interference for a printed press fit. The seat is cut this much
    #: SMALLER than outer_diameter_mm so the outer race grips.
    press_fit_interference_mm: float = 0.10

    #: How far the bearing stands above the face its seat is cut into, in mm.
    #: Must be positive: the part above rides on the bearing's inner race, and
    #: a flush bearing would let it scrub the printed face instead. It also
    #: occupies real height in the base stack, so it is kept here rather than
    #: as a modelling constant inside cad/ -- see HardwareSpec.pedestal_height_mm.
    proud_mm: float = 0.50

    #: Outer diameter of the *inner* ring, in mm. The part above rides on this
    #: ring's top face, so it bounds how far out the seating land may reach
    #: before it fouls the outer ring instead.
    #:
    #: UNVERIFIED. Bearing tables publish bore/OD/width but rarely the ring
    #: split, and it varies between makers. The value below is a deliberately
    #: conservative estimate - bore plus a quarter of the radial section, which
    #: predicts 11.5 mm for a 608 whose real inner ring is about 12.1 mm - so a
    #: land sized against it stays on the inner ring even if the estimate is
    #: optimistic by a millimetre.
    inner_race_outer_diameter_mm: float = 33.0

    #: Fields awaiting measurement of a physical bearing. The bore/OD/width
    #: triple is a published standard and is deliberately absent from this list.
    UNVERIFIED_FIELDS: ClassVar[Tuple[str, ...]] = ("inner_race_outer_diameter_mm",)

    def __post_init__(self) -> None:
        if not 0.0 < self.bore_diameter_mm < self.outer_diameter_mm:
            raise ValueError(
                f"BearingSpec {self.name}: require "
                f"0 < bore ({self.bore_diameter_mm}) < OD "
                f"({self.outer_diameter_mm})."
            )
        if not (
            self.bore_diameter_mm
            < self.inner_race_outer_diameter_mm
            < self.outer_diameter_mm
        ):
            raise ValueError(
                f"BearingSpec {self.name}: inner_race_outer_diameter_mm "
                f"({self.inner_race_outer_diameter_mm}) must lie strictly "
                f"between the bore ({self.bore_diameter_mm}) and the OD "
                f"({self.outer_diameter_mm})."
            )
        if self.width_mm <= 0.0:
            raise ValueError(
                f"BearingSpec {self.name}: width_mm must be positive, "
                f"got {self.width_mm}."
            )
        if not 0.0 < self.proud_mm < self.width_mm:
            raise ValueError(
                f"BearingSpec {self.name}: proud_mm ({self.proud_mm}) must be "
                f"positive and less than width_mm ({self.width_mm}); otherwise "
                "there is no seat left to press the outer race into."
            )

    @property
    def seat_diameter_mm(self) -> float:
        """Diameter of the pocket the outer race presses into."""
        return self.outer_diameter_mm - self.press_fit_interference_mm

    @property
    def seat_depth_mm(self) -> float:
        """Depth of that pocket: the bearing's width less what stands proud."""
        return self.width_mm - self.proud_mm

    @property
    def race_land_width_mm(self) -> float:
        """
        Radial width of the inner ring's exposed top face, in mm.

        The annulus a driven part may bear on without touching the outer ring.
        """
        return (self.inner_race_outer_diameter_mm - self.bore_diameter_mm) / 2.0


#: The 608ZZ that used to be the yaw bearing. It remains in the bill of
#: materials as the shoulder yoke's idler pivot - see
#: :attr:`HardwareSpec.shoulder_idler_bearing`.
BEARING_608ZZ = BearingSpec(
    name="608ZZ",
    bore_diameter_mm=8.0,
    outer_diameter_mm=22.0,
    width_mm=7.0,
    inner_race_outer_diameter_mm=11.5,
)


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
    shoulder pivot. The base pedestal is only the bottom of that stack. Above
    its top face sit, in order:

    1. the part of the thrust bearing standing proud of that face
       (``BearingSpec.proud_mm``, accounted for separately because it belongs
       to the bearing, not to a printed part),
    2. the yaw turntable plate, and
    3. the shoulder bracket.

    Splitting the budget explicitly keeps FK's z = base_height_mm shoulder
    pivot exact by construction, instead of discovering after printing that
    the real pivot ended up 30 mm high. The pedestal's height is whatever is
    left over -- see :meth:`HardwareSpec.pedestal_height_mm`.

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
    Monolithic U-clamp that wraps the desk edge, carrying the base yaw joint.

    In side view the part is a C-profile, like a monitor-arm clamp::

                    servo turret
                   +----------+
        z=+70      |          |
                   |          |
        z=+15  +---+----------+---------+   top arm, lies on the desk
               |///|                    |
               |///|   throat (desk)    |   spine hugs the desk edge outside
               |///|                    |
        z=-45  +---+----------+---------+   bottom arm, under the desk
               |///|          |  bolt   |
        z=-60  +---+----------+---------+
                        knob below

    The servo cannot live inside the 15 mm top arm -- a DS3218 is 40.5 mm tall
    and needs 55.5 mm of housing once the bearing seat and ceiling are counted
    -- so it sits in a turret rising from the arm to the full pedestal height.
    The arm stays thin, which is what makes the profile a C rather than a slab.

    Clamping is by a single M8 driven **upward** from the bottom arm: the nut
    is captive in a pocket in that arm's underside, the head and its knob hang
    below, and the screw's tip carries a printed pressure foot that bears on
    the desk's underside. Note this is the only arrangement that can actually
    grip: a screw entering from the top of the bottom arm would press on
    nothing.

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

    #: Vertical gap between the arms, in mm. Fixed by the printed geometry --
    #: unlike a screw-adjusted jaw, a U-clamp's throat cannot open further, so
    #: this must exceed the thickest supported desk outright.
    throat_max_opening_mm: float = 45.0

    # ---- U-profile geometry -------------------------------------------------
    #: How far the top arm reaches inward over the desk from the edge, in mm.
    top_arm_depth_mm: float = 60.0

    #: How far the bottom arm reaches inward under the desk, in mm.
    bottom_arm_depth_mm: float = 60.0

    #: Vertical thickness of the top arm, in mm. This is the plate that lies
    #: on the desk; the servo housing is a turret above it, not inside it.
    top_arm_thickness_mm: float = 15.0

    #: Vertical thickness of the bottom arm, in mm. Must swallow the nut
    #: pocket and still leave a structural floor.
    bottom_arm_thickness_mm: float = 15.0

    #: Thickness of the vertical spine that hugs the desk edge, in mm.
    spine_thickness_mm: float = 15.0

    #: Distance from the desk edge inward to the servo shaft -- which is the
    #: arm's yaw axis. Must match ``ArmGeometry.base_y_on_desk_mm``, since that
    #: is the same measurement expressed in the desk frame.
    servo_shaft_offset_from_edge_mm: float = 30.0

    #: Leg size of the triangular gussets at the two inner corners of the U,
    #: in mm. Gussets rather than fillets: a swept fillet is the most fragile
    #: operation to re-run when upstream dimensions move, and this package's
    #: whole premise is that dimensions do move.
    gusset_size_mm: float = 5.0

    # ---- Anti-slip pads -----------------------------------------------------
    #: Depth of the anti-slip pad recesses, in mm.
    pad_recess_depth_mm: float = 2.0

    #: Thickness of the rubber sheet glued into those recesses, in mm.
    pad_thickness_mm: float = 2.0

    #: Gap left between a pad recess and the feature beside it, in mm. Smaller
    #: than a structural wall: these are surface recesses, not load paths.
    pad_edge_margin_mm: float = 2.0

    # ---- Clamping screw -----------------------------------------------------
    bolt_thread: str = "M8"
    bolt_nominal_diameter_mm: float = 8.0
    bolt_thread_pitch_mm: float = 1.25
    bolt_length_mm: float = 70.0
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

    # ---- Pressure foot ------------------------------------------------------
    #: Printed puck on the screw's tip. A bare M8 tip at a few hundred newtons
    #: would bite into the desk underside; the foot spreads that load and
    #: carries the second anti-slip pad.
    pressure_foot_diameter_mm: float = 24.0
    #: Diameter of the pad recess in the foot's upper face, in mm.
    pressure_foot_pad_diameter_mm: float = 20.0

    #: Conical seat in the foot's underside that the screw's tip pivots in.
    #:
    #: Session D.1d replaced a threaded bore with this. A foot threaded onto a
    #: turning screw rotates with it, dragging its rubber pad across the desk
    #: as you tighten -- which scuffs the finish and, because friction then
    #: acts at the pad's radius rather than the screw's, roughly halves the
    #: preload a given hand torque produces. Seated on a cone the foot is free
    #: to stay still while the screw turns inside it, which is what every
    #: monitor-arm clamp does.
    pressure_foot_seat_diameter_mm: float = 9.0
    #: Flat at the cone's apex, in mm. Truncating the cone keeps a sharp
    #: internal point out of the print -- which no nozzle can resolve anyway --
    #: and avoids the degenerate tessellation a true apex produces in STL. The
    #: screw's tip contacts the cone wall well above it, so it is functionally
    #: free.
    pressure_foot_seat_apex_diameter_mm: float = 2.0
    #: Included angle of that cone, in degrees. A standard countersink angle,
    #: shallow enough to keep the foot thin. It also sets how much the cone
    #: amplifies contact force -- see :meth:`torque_to_preload_factor_m`.
    pressure_foot_seat_angle_deg: float = 120.0
    #: Material between the seat's apex and the pad recess floor, in mm.
    #: Loaded in pure compression, so it does not need a structural thickness.
    pressure_foot_web_mm: float = 2.0

    #: Diameter of the screw's chamfered end, in mm. Sets the radius at which
    #: the tip contacts the cone, and therefore the friction lever. Roughly the
    #: thread's minor diameter for an ISO 4753 chamfer; UNVERIFIED to better
    #: than a few tenths, which moves preload by only a few percent.
    bolt_tip_diameter_mm: float = 6.4

    # ---- Hand knob ----------------------------------------------------------
    knob_diameter_mm: float = 50.0
    knob_thickness_mm: float = 20.0
    #: How far below the knob's top face the bolt head is buried, in mm. Sunk
    #: deliberately: it protects the head and shortens the bolt length needed.
    knob_head_recess_mm: float = 2.50
    #: Number of grip flutes cut around the knob's perimeter.
    knob_flute_count: int = 12

    # ---- Friction coefficients ----------------------------------------------
    #: Anti-slip pad against the desk. 0.4 is the PETG-on-wood figure, used
    #: deliberately even though the pad is rubber (which is higher), so the
    #: slip margin is understated rather than overstated.
    pad_friction_coefficient: float = 0.40
    #: Steel bolt thread in a steel nut, dry.
    thread_friction_coefficient: float = 0.15
    #: Steel screw tip against the printed cone of the pressure foot. Until
    #: Session D.1d this modelled a knob boss bearing on the clamp, a contact
    #: the U-clamp does not have: the knob hangs free below the bottom arm.
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
                f"supported desk ({high}). A U-clamp's throat is fixed by the "
                "printed geometry and cannot be opened further."
            )
        for name in (
            "top_arm_depth_mm", "bottom_arm_depth_mm", "top_arm_thickness_mm",
            "bottom_arm_thickness_mm", "spine_thickness_mm",
            "servo_shaft_offset_from_edge_mm", "gusset_size_mm",
            "pad_recess_depth_mm", "pad_thickness_mm",
            "bolt_nominal_diameter_mm", "bolt_thread_pitch_mm", "bolt_length_mm",
            "pressure_foot_diameter_mm", "pressure_foot_seat_diameter_mm",
            "pressure_foot_web_mm", "bolt_tip_diameter_mm",
            "knob_diameter_mm", "knob_thickness_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"DeskClampSpec: {name} must be positive, got "
                    f"{getattr(self, name)}."
                )
        if self.servo_shaft_offset_from_edge_mm >= self.top_arm_depth_mm:
            raise ValueError(
                f"DeskClampSpec: the shaft sits "
                f"{self.servo_shaft_offset_from_edge_mm} mm inward but the top "
                f"arm only reaches {self.top_arm_depth_mm} mm, so the yaw axis "
                "would fall off the end of the arm."
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
        if self.nut_pocket_depth_mm >= self.bottom_arm_thickness_mm:
            raise ValueError(
                f"DeskClampSpec: the nut pocket "
                f"({self.nut_pocket_depth_mm} mm) is as deep as the bottom arm "
                f"({self.bottom_arm_thickness_mm} mm), leaving no floor for the "
                "nut to bear against."
            )
        if self.bolt_clearance_hole_diameter_mm <= self.bolt_nominal_diameter_mm:
            raise ValueError(
                f"DeskClampSpec: bolt clearance hole "
                f"({self.bolt_clearance_hole_diameter_mm}) must exceed the "
                f"nominal diameter ({self.bolt_nominal_diameter_mm})."
            )
        if self.pressure_foot_pad_diameter_mm >= self.pressure_foot_diameter_mm:
            raise ValueError(
                f"DeskClampSpec: the pressure foot's pad recess "
                f"({self.pressure_foot_pad_diameter_mm}) must be smaller than "
                f"the foot itself ({self.pressure_foot_diameter_mm})."
            )
        if self.pressure_foot_seat_diameter_mm >= self.pressure_foot_pad_diameter_mm:
            raise ValueError(
                "DeskClampSpec: the pressure foot's seat must be smaller than "
                "its pad recess."
            )
        if not 0.0 < self.pressure_foot_seat_angle_deg < 180.0:
            raise ValueError(
                f"DeskClampSpec: pressure_foot_seat_angle_deg must be a real "
                f"included angle in (0, 180), got "
                f"{self.pressure_foot_seat_angle_deg}."
            )
        if self.pressure_foot_seat_apex_diameter_mm >= self.pressure_foot_seat_diameter_mm:
            raise ValueError(
                "DeskClampSpec: the seat's apex flat must be smaller than its "
                "mouth, or the seat is a plain counterbore rather than a cone."
            )
        if not (
            self.pressure_foot_seat_apex_diameter_mm
            < self.bolt_tip_diameter_mm
            < self.pressure_foot_seat_diameter_mm
        ):
            raise ValueError(
                f"DeskClampSpec: the screw tip ({self.bolt_tip_diameter_mm} mm) "
                f"must sit between the seat's apex flat "
                f"({self.pressure_foot_seat_apex_diameter_mm} mm) and its mouth "
                f"({self.pressure_foot_seat_diameter_mm} mm) so it contacts the "
                "cone wall rather than bottoming out or missing the seat."
            )
        if self.knob_flute_count < 3:
            raise ValueError(
                "DeskClampSpec: knob_flute_count must be at least 3, got "
                f"{self.knob_flute_count}."
            )

    # ---- Desk compatibility -------------------------------------------------

    @property
    def min_desk_thickness_mm(self) -> float:
        return self.desk_thickness_range_mm[0]

    @property
    def max_desk_thickness_mm(self) -> float:
        return self.desk_thickness_range_mm[1]

    @property
    def nominal_desk_thickness_mm(self) -> float:
        """
        Mid-range desk thickness, in mm.

        Used wherever a single representative desk is needed -- the assembly
        preview, worked examples -- rather than hardcoding a number that would
        then drift from the supported range.
        """
        return float(sum(self.desk_thickness_range_mm) / 2.0)

    @property
    def desk_removal_clearance_mm(self) -> float:
        """Slack above the thickest desk, for sliding the clamp on and off."""
        return self.throat_max_opening_mm - self.max_desk_thickness_mm

    # ---- Derived geometry ---------------------------------------------------

    @property
    def nut_pocket_depth_mm(self) -> float:
        """Pocket depth for the captive nut. Sized from the MAX nut thickness."""
        return self.nut_thickness_max_mm

    @property
    def nut_across_corners_mm(self) -> float:
        """Hexagon across-corners, 2/sqrt(3) times across-flats."""
        return float(self.nut_across_flats_mm * 2.0 / np.sqrt(3.0))

    @property
    def bottom_arm_floor_mm(self) -> float:
        """Material between the nut pocket's crown and the arm's top face."""
        return self.bottom_arm_thickness_mm - self.nut_pocket_depth_mm

    @property
    def pressure_foot_seat_half_angle_rad(self) -> float:
        """Half the seat cone's included angle, in radians."""
        return float(np.radians(self.pressure_foot_seat_angle_deg / 2.0))

    @property
    def pressure_foot_seat_depth_mm(self) -> float:
        """
        Depth of the conical seat, mouth to truncated apex, in mm.

        Set by the mouth and apex diameters and the cone angle: a shallower
        cone (larger included angle) gives a thinner foot.
        """
        return float(
            (
                self.pressure_foot_seat_diameter_mm / 2.0
                - self.pressure_foot_seat_apex_diameter_mm / 2.0
            )
            / np.tan(self.pressure_foot_seat_half_angle_rad)
        )

    @property
    def pressure_foot_height_mm(self) -> float:
        """Total printed height of the pressure foot, in mm."""
        return (
            self.pressure_foot_seat_depth_mm
            + self.pressure_foot_web_mm
            + self.pad_recess_depth_mm
        )

    @property
    def screw_tip_contact_radius_mm(self) -> float:
        """
        Radius at which the screw's tip touches the cone, in mm.

        The tip's chamfered edge rests against the cone wall, so the contact is
        a circle at the tip's own radius -- not at the seat's mouth.
        """
        return self.bolt_tip_diameter_mm / 2.0

    @property
    def screw_tip_seat_height_mm(self) -> float:
        """
        How far above the foot's underside the seated screw tip sits, in mm.

        The tip stops where its edge meets the cone, part-way down rather than
        at the apex.
        """
        mouth_radius = self.pressure_foot_seat_diameter_mm / 2.0
        apex_radius = self.pressure_foot_seat_apex_diameter_mm / 2.0
        return float(
            self.pressure_foot_seat_depth_mm
            * (mouth_radius - self.screw_tip_contact_radius_mm)
            / (mouth_radius - apex_radius)
        )

    @property
    def pressure_foot_rise_above_tip_mm(self) -> float:
        """How far the foot's contact face stands above the screw's tip."""
        return self.pressure_foot_height_mm - self.screw_tip_seat_height_mm

    @property
    def knob_socket_depth_mm(self) -> float:
        """Depth of the hex socket in the knob's top face."""
        return self.bolt_head_height_mm + self.knob_head_recess_mm

    @property
    def max_screw_protrusion_mm(self) -> float:
        """
        Screw travel above the bottom arm needed for the *thinnest* desk.

        A thin desk sits high in the throat, so its underside is furthest from
        the bottom arm and the screw has to reach hardest. The pressure foot
        makes up part of that distance.
        """
        return (
            self.throat_max_opening_mm
            - self.min_desk_thickness_mm
            - self.pressure_foot_rise_above_tip_mm
        )

    @property
    def required_bolt_length_mm(self) -> float:
        """
        Shortest screw that still reaches the thinnest supported desk.

        Stack, from under the head upward: the knob material below the head,
        full engagement in the captive nut, the bottom arm above it, and the
        protrusion into the throat.
        """
        knob_below_head = self.knob_thickness_mm - self.knob_socket_depth_mm
        return float(
            knob_below_head
            + self.nut_thickness_max_mm
            + self.bottom_arm_thickness_mm
            + self.max_screw_protrusion_mm
        )

    # ---- Clamping physics ---------------------------------------------------

    def torque_to_preload_factor_m(self) -> float:
        """
        Metres of effective lever between applied torque and bolt preload.

        Preload is ``torque / factor``. Built from the power-screw relation
        rather than a lumped nut factor, so the thread pitch actually appears:

            T = F * [ d2/2 * (p + pi*mu_t*d2*sec(a)) / (pi*d2 - mu_t*p*sec(a))
                      + mu_c * r_c ]

        with ``d2`` the pitch diameter (``d - 0.6495*p`` for ISO metric) and
        ``a = 30 deg`` the thread half-angle.

        **The collar term changed in Session D.1d.** It previously modelled a
        boss under the knob bearing on the clamp -- a contact the D.1b jaw had
        but the U-clamp does not, because the knob hangs free below the bottom
        arm. The real rubbing interface is the screw's tip turning in the
        pressure foot's conical seat, so ``r_c`` is the tip's contact radius,
        divided by ``sin`` of the cone's half-angle because a cone wedges the
        contact force above the axial load:

            r_c_effective = r_tip / sin(seat_half_angle)

        A lumped nut factor (``T = K*F*d``, K about 0.2 dry steel-on-steel)
        would be the textbook alternative, but K is calibrated for a flat
        steel bearing face under the head. This joint has neither -- the head
        turns with the knob and touches nothing -- so deriving the two real
        interfaces is both more honest and more transferable.
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

        collar_term_mm = (
            self.collar_friction_coefficient
            * self.screw_tip_contact_radius_mm
            / np.sin(self.pressure_foot_seat_half_angle_rad)
        )

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

    def hand_torque_limit_nm(self, grip_force_n: float = 40.0) -> float:
        """
        Torque a hand can actually apply to this knob, in newton-metres.

        A firm two-finger grip is about 40 N tangential at the rim, so torque
        scales directly with knob radius. This is the point of choosing a
        knob size: shrinking it caps the torque physically, rather than
        relying on the user to read a warning.
        """
        if grip_force_n < 0.0:
            raise ValueError(
                f"grip_force_n must be non-negative, got {grip_force_n}."
            )
        return float(grip_force_n * (self.knob_diameter_mm / 2.0) / 1000.0)

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

        The arm's tipping moment tries to lift the clamped assembly off the
        desk. The screw's preload holds it down, acting at ``lever_arm_mm``
        from the pivot -- taken as the inboard edge of the top arm's pads, the
        last line of contact the assembly would rotate about.

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

    def bottom_arm_allowable_preload_n(
        self,
        width_mm: float,
        overhang_mm: float,
        allowable_stress_mpa: float = 25.0,
    ) -> float:
        """
        Preload the bottom arm can carry before bending failure, in newtons.

        The bottom arm cantilevers from the spine with the screw's reaction at
        ``overhang_mm`` along it, so it is a rectangular-section cantilever:
        section modulus ``width * thickness^2 / 6``, allowable load
        ``stress * modulus / overhang``.

        The 25 MPa default is roughly half of PETG's ~50 MPa tensile yield,
        i.e. a safety factor of 2 against a printed part whose layer adhesion
        is weaker than bulk material.

        This is the weakest link in the U, and what sets the maximum safe
        tightening torque.
        """
        if width_mm <= 0.0:
            raise ValueError(f"width_mm must be positive, got {width_mm}.")
        if overhang_mm <= 0.0:
            raise ValueError(f"overhang_mm must be positive, got {overhang_mm}.")
        section_modulus_mm3 = width_mm * self.bottom_arm_thickness_mm**2 / 6.0
        return float(allowable_stress_mpa * section_modulus_mm3 / overhang_mm)

    def max_tightening_torque_nm(
        self,
        width_mm: float,
        overhang_mm: float,
        allowable_stress_mpa: float = 25.0,
    ) -> float:
        """Hand torque at which the bottom arm reaches its allowable stress."""
        return self.preload_to_torque_nm(
            self.bottom_arm_allowable_preload_n(
                width_mm, overhang_mm, allowable_stress_mpa
            )
        )


@dataclass(frozen=True)
class ServoHornSpec:
    """
    The metal horn that couples a servo's splined output to a printed part.

    Two parts of the arm are driven straight off a servo shaft - the yaw
    turntable and the upper arm's shoulder end - and both use the same
    interface, so it is described once here.

    Why a bought horn rather than a printed spline
    ----------------------------------------------
    The DS3218's output is a 25-tooth spline 5.9 mm across. Tooth pitch is
    therefore ``pi * 5.9 / 25 = 0.74 mm``, which a 0.4 mm nozzle cannot
    resolve into anything that will carry 20 kg.cm. Every printed part in this
    project drives through a bought aluminium horn instead, and bolts to it.

    .. note::
       **The spline diameter is the one verified number here.** DSServo's
       listings and the retailers give 5.9 mm for the DS3218's 25T spline.
       Everything describing the *horn* is a placeholder: round 25T horns are
       sold at 19.7, 24.5 and 25 mm across by different suppliers, and hole
       patterns vary (REV publishes six M3 holes on a 16 mm circle; Power HD
       gives two holes at 15 and 19 mm radius). The defaults below describe a
       25 mm round horn with four M3 holes on a 16 mm circle, which is
       self-consistent - a 25 mm disc leaves 4.5 mm of rim outside a 16 mm
       circle, enough for M3 - but the horn in hand must be measured before
       anything is printed for final assembly.
    """

    name: str = "25T round horn"
    spline_teeth: int = 25

    #: Across the spline, in mm. Datasheet-confirmed for the DS3218.
    spline_diameter_mm: float = 5.9

    # ---- Disc and hub (PLACEHOLDER - measure before printing) ------------
    #: Outside diameter of the horn's disc, in mm. Sets the pocket every driven
    #: part needs, and hence the smallest bearing bore the coupling can pass
    #: through.
    disc_diameter_mm: float = 25.0
    #: Thickness of that disc, in mm.
    disc_thickness_mm: float = 3.0
    #: Diameter of the hub boss on the servo side, in mm.
    hub_diameter_mm: float = 13.0
    #: Height of that boss, in mm. It rests on the servo's shaft boss, so the
    #: horn's disc ends up this far above the shaft's crown.
    hub_height_mm: float = 3.0

    # ---- Mounting pattern (PLACEHOLDER - measure before printing) --------
    bolt_circle_mm: float = 16.0
    bolt_count: int = 4
    #: Nominal thread of the horn's mounting holes, in mm. They are threaded,
    #: so the driven part carries clearance holes and the screws pull into the
    #: horn.
    bolt_nominal_diameter_mm: float = 3.0
    #: ISO 273 medium clearance hole for that thread.
    bolt_clearance_diameter_mm: float = 3.4
    #: Socket-head envelope for the counterbores the driven parts need, so no
    #: screw head stands proud of a mating face.
    bolt_head_diameter_mm: float = 5.5
    bolt_head_height_mm: float = 3.0

    UNVERIFIED_FIELDS: ClassVar[Tuple[str, ...]] = (
        "disc_diameter_mm",
        "disc_thickness_mm",
        "hub_diameter_mm",
        "hub_height_mm",
        "bolt_circle_mm",
        "bolt_count",
    )

    def __post_init__(self) -> None:
        for name in (
            "spline_diameter_mm", "disc_diameter_mm", "disc_thickness_mm",
            "hub_diameter_mm", "hub_height_mm", "bolt_circle_mm",
            "bolt_nominal_diameter_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"ServoHornSpec {self.name}: {name} must be positive, got "
                    f"{getattr(self, name)}."
                )
        if self.bolt_count < 3:
            raise ValueError(
                f"ServoHornSpec {self.name}: bolt_count must be at least 3 to "
                f"locate a part rotationally, got {self.bolt_count}."
            )
        if self.bolt_circle_mm >= self.disc_diameter_mm:
            raise ValueError(
                f"ServoHornSpec {self.name}: the bolt circle "
                f"({self.bolt_circle_mm}) must be smaller than the disc "
                f"({self.disc_diameter_mm}); the holes would fall off it."
            )
        rim = (self.disc_diameter_mm - self.bolt_circle_mm) / 2.0
        if rim <= self.bolt_nominal_diameter_mm:
            raise ValueError(
                f"ServoHornSpec {self.name}: only {rim:.2f} mm of rim outside "
                f"the bolt circle, which cannot hold an "
                f"M{self.bolt_nominal_diameter_mm:.0f} hole."
            )
        if self.hub_diameter_mm >= self.disc_diameter_mm:
            raise ValueError(
                f"ServoHornSpec {self.name}: the hub ({self.hub_diameter_mm}) "
                f"must be smaller than the disc ({self.disc_diameter_mm})."
            )
        if self.spline_diameter_mm >= self.hub_diameter_mm:
            raise ValueError(
                f"ServoHornSpec {self.name}: the hub ({self.hub_diameter_mm}) "
                f"must be wider than the spline it grips "
                f"({self.spline_diameter_mm})."
            )

    @property
    def total_height_mm(self) -> float:
        """
        Hub plus disc, in mm.

        Measured from the servo's shaft boss - which the hub sits on - to the
        horn's outer face, which is the surface a driven part bolts against.
        """
        return self.hub_height_mm + self.disc_thickness_mm

    @property
    def spline_tooth_pitch_mm(self) -> float:
        """Circumferential pitch of one spline tooth, in mm."""
        return float(np.pi * self.spline_diameter_mm / self.spline_teeth)

    @property
    def index_resolution_deg(self) -> float:
        """
        Finest angular step the spline can be re-indexed by, in degrees.

        360 / 25 = 14.4 for a 25T spline. This is the resolution with which a
        joint's zero can be set mechanically; anything finer is trimmed in
        software.
        """
        return 360.0 / self.spline_teeth

    def bolt_positions(self, phase_deg: float = 45.0) -> Tuple[Tuple[float, float], ...]:
        """
        (x, y) centres of the horn's mounting holes, in mm, about its axis.

        ``phase_deg`` rotates the pattern; the default staggers it 45 degrees
        so a four-hole pattern lands on the diagonals of a square part.
        """
        radius = self.bolt_circle_mm / 2.0
        return tuple(
            (
                float(radius * np.cos(np.radians(phase_deg + i * 360.0 / self.bolt_count))),
                float(radius * np.sin(np.radians(phase_deg + i * 360.0 / self.bolt_count))),
            )
            for i in range(self.bolt_count)
        )

    def unverified_report(self) -> str:
        """Human-readable list of fields still awaiting physical measurement."""
        lines = [
            f"{self.name}: {len(self.UNVERIFIED_FIELDS)} UNVERIFIED dimension(s) "
            "- measure the horn in hand before printing:"
        ]
        for field_name in self.UNVERIFIED_FIELDS:
            lines.append(f"    {field_name:<34} = {getattr(self, field_name)}")
        return "\n".join(lines)


@dataclass(frozen=True)
class YawTurntableSpec:
    """
    The plate that turns on the yaw bearing and carries the shoulder bracket.

    It sits on top of the pedestal's thrust bearing, is driven by the base yaw
    servo through a :class:`ServoHornSpec`, and presents a bolt circle for the
    shoulder bracket. Bottom to top it is:

    1. a spigot filling the bearing's bore, with the horn pocketed inside it,
    2. a land bearing on the inner ring's top face,
    3. the plate proper, relieved over the outer ring.

    Why the plate does not simply sit on the bearing
    ------------------------------------------------
    The bearing stands ``BearingSpec.proud_mm`` (0.5 mm) above the turret. A
    plain recess deeper than that would drop the plate onto the printed turret
    face and leave the bearing carrying nothing, so
    :attr:`bearing_race_recess_mm` relieves the **outer** ring only, and the
    plate lands on the inner ring inside it. That is what makes the bearing a
    bearing rather than a spacer.
    """

    #: Plate diameter, in mm. Sized from the bracket's bolt circle outward,
    #: not from the turret: the turret is rectangular and 66 mm across its long
    #: side, so a disc that covered it would have to overhang the top arm.
    #: What the plate must cover is the bearing.
    diameter_mm: float = 68.0

    #: Plate thickness, in mm. Must equal ``BaseStack.turntable_plate_thickness_mm``
    #: - the same 6 mm counted once in the height budget and once here.
    thickness_mm: float = 6.0

    #: Depth of the annular relief over the bearing's outer ring, in mm. Must
    #: exceed ``BearingSpec.proud_mm`` or the relief does not clear.
    bearing_race_recess_mm: float = 1.0

    #: Diametral slip fit of the spigot in the bearing's bore, in mm. Taken off
    #: the bore so the spigot enters without a press.
    spigot_bore_fit_mm: float = 0.50

    # ---- Shoulder bracket interface ---------------------------------------
    #: (X, Y) spacing of the four shoulder-bracket screws, in mm, centred on
    #: the yaw axis.
    #:
    #: A rectangle rather than a bolt circle, which is what the brief for D.2a
    #: proposed. No circle works: the holes must clear the bearing's relief
    #: (radius 21.25 mm) and must also miss the bracket's two walls, which
    #: stand on the plate between y = 15.5 and 21.5 mm. A circle big enough for
    #: the first crosses the second at every phase angle unless its diameter
    #: exceeds 66 mm, at which point it is off the plate. Putting the holes on
    #: a rectangle - well outboard along X, well inboard along Y - satisfies
    #: both, and leaves all four heads reachable from above between the walls.
    bracket_bolt_pattern_mm: Tuple[float, float] = (52.0, 20.0)
    #: Nominal thread of the bracket screws.
    bracket_bolt_nominal_diameter_mm: float = 3.0
    #: Pilot hole the screws cut their own thread in. Blind, not through: a nut
    #: on the underside would foul the bearing and the turret 0.5 mm below.
    bracket_bolt_pilot_diameter_mm: float = 2.5
    bracket_bolt_depth_mm: float = 4.0

    # ---- Yaw-zero witness --------------------------------------------------
    #: A notch in the rim marking the arm-forward direction, so the turntable
    #: goes back on the horn the same way round after a strip-down.
    #:
    #: This is a witness mark, not a key. A true key is not possible against a
    #: round horn whose four holes sit on a square: the pattern is symmetric
    #: under 90 degree rotation, so any mating feature fits four ways. Keying
    #: it would mean filing a flat on a bought horn. The notch lines up with
    #: the matching one on the turret's front face at yaw zero.
    index_notch_width_mm: float = 3.0
    index_notch_depth_mm: float = 1.5
    index_notch_length_mm: float = 6.0

    def __post_init__(self) -> None:
        for name in (
            "diameter_mm", "thickness_mm", "bearing_race_recess_mm",
            "spigot_bore_fit_mm",
            "bracket_bolt_pilot_diameter_mm", "bracket_bolt_depth_mm",
            "index_notch_width_mm", "index_notch_depth_mm",
            "index_notch_length_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"YawTurntableSpec: {name} must be positive, got "
                    f"{getattr(self, name)}."
                )
        if len(self.bracket_bolt_pattern_mm) != 2 or any(
            value <= 0.0 for value in self.bracket_bolt_pattern_mm
        ):
            raise ValueError(
                "YawTurntableSpec: bracket_bolt_pattern_mm must be a pair of "
                f"positive spacings, got {self.bracket_bolt_pattern_mm}."
            )
        if self.bracket_bolt_radius_mm * 2.0 >= self.diameter_mm:
            raise ValueError(
                f"YawTurntableSpec: the bracket bolt pattern reaches "
                f"{self.bracket_bolt_radius_mm:.2f} mm from the axis, off a "
                f"{self.diameter_mm} mm plate."
            )
        if self.bracket_bolt_depth_mm >= self.thickness_mm:
            raise ValueError(
                f"YawTurntableSpec: a {self.bracket_bolt_depth_mm} mm blind "
                f"hole in a {self.thickness_mm} mm plate leaves no floor; the "
                "screw would break through onto the bearing."
            )
        if self.bracket_bolt_pilot_diameter_mm >= self.bracket_bolt_nominal_diameter_mm:
            raise ValueError(
                f"YawTurntableSpec: the pilot hole "
                f"({self.bracket_bolt_pilot_diameter_mm}) must be smaller than "
                f"the screw ({self.bracket_bolt_nominal_diameter_mm}), or there "
                "is no material for it to cut a thread in."
            )
        if self.index_notch_depth_mm >= self.thickness_mm:
            raise ValueError(
                "YawTurntableSpec: the index notch must not cut through the "
                f"plate ({self.index_notch_depth_mm} into "
                f"{self.thickness_mm} mm)."
            )

    @property
    def bracket_bolt_count(self) -> int:
        """Four, by construction: the pattern is a rectangle."""
        return 4

    @property
    def bracket_bolt_radius_mm(self) -> float:
        """How far the outermost bracket screw sits from the yaw axis, in mm."""
        x_span, y_span = self.bracket_bolt_pattern_mm
        return float(np.hypot(x_span / 2.0, y_span / 2.0))

    def bracket_bolt_positions(self) -> Tuple[Tuple[float, float], ...]:
        """(x, y) centres of the shoulder bracket's four mounting holes, in mm."""
        x_span, y_span = self.bracket_bolt_pattern_mm
        return tuple(
            (float(sx * x_span / 2.0), float(sy * y_span / 2.0))
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )


@dataclass(frozen=True)
class ShoulderBracketSpec:
    """
    The clevis that stands on the yaw turntable and carries the shoulder servo.

    Two walls rise from a base plate with the shoulder pitch servo held between
    them. Its output shaft is the shoulder pitch axis, and by construction that
    axis lands at ``ArmGeometry.base_height_mm`` above the desk.

    Sign convention
    ---------------
    ``forward_kinematics.py`` makes shoulder pitch a rotation about the base
    frame's **+Y**, with positive theta pitching the arm *down* from horizontal.
    :attr:`shaft_direction` records that the servo's output points the same way,
    so the servo's positive rotation is positive theta_2 with no sign flip in
    the driver. (Which pulse width produces which physical direction is a
    property of the servo, not of this bracket, and is resolved by a per-joint
    sign in the firmware.)

    Why the servo lies on its side
    ------------------------------
    It has to. Standing the servo on end - body length vertical - puts its lower
    mounting ear 24 - 49.5/2 = -0.75 mm relative to the base plate's underside,
    i.e. below the turntable it is bolted to. Laid down, the body straddles the
    pitch axis from 14 to 34 mm and both ears are reachable. Laying it down also
    keeps every part of the bracket inside a 24 mm radius of the pitch axis, so
    the upper arm's yoke can swing through the joint's full travel.

    Retrofit provision
    ------------------
    docs/PROOF_OF_CONCEPT.md section 2.2 records that this joint needs about
    34.5 kg.cm and a DS3218 supplies 20. The provision baked in here is that
    **the two walls are identical**: the far wall carries the same servo window
    and the same four mounting holes as the driven wall. Today it holds a small
    idler plug with the yoke's pivot bearing; fitting a second DS3218 later is
    a matter of unbolting that plug. A 2:1 reduction is the other option in
    section 2.2 and is *not* designed here - its gear geometry has not been
    chosen - but the same wall pattern is what a reduction plate would bolt to.
    """

    #: Footprint of the base plate on the turntable, (X, Y) in mm. X is the
    #: arm-forward direction. Larger than the wall span because it also has to
    #: carry the turntable's bolt circle with edge distance around it.
    base_plate_mm: Tuple[float, float] = (72.0, 43.0)
    base_plate_thickness_mm: float = 6.0

    #: Wall thickness, in mm. Six rather than the 4 mm minimum because the
    #: servo's retention screws are blind holes driven into the wall's inner
    #: face: 4 mm would leave a 1 mm floor for an M3 to tent or break through.
    wall_thickness_mm: float = 6.0

    #: Depth of those blind pilot holes, in mm.
    servo_screw_depth_mm: float = 4.5

    #: Base plate underside to the shoulder pitch axis, in mm. Must equal
    #: ``BaseStack.shoulder_bracket_rise_mm``: the same rise counted once in
    #: the height budget and once here.
    bracket_height_mm: float = 24.0

    #: Which way along the base frame's Y the servo's output shaft points.
    shaft_direction: str = "+Y"

    #: Leg of the triangular gussets at the wall-to-base junctions, in mm.
    #: Gussets, not fillets, for the reason cad/README.md gives: a swept fillet
    #: is the most fragile operation to re-run when an upstream dimension moves,
    #: and every dimension here is derived from this file.
    gusset_size_mm: float = 6.0

    #: Cable exit in the base plate for the shoulder servo's lead, (X, Y) in mm.
    #: It routes down through the turntable's centre and into the pedestal.
    cable_slot_mm: Tuple[float, float] = (16.0, 8.0)

    # ---- Idler plug (fills the undriven wall) -----------------------------
    #: Thickness of the plate that fills the far wall's servo window, in mm.
    idler_plug_thickness_mm: float = 6.0
    #: Diameter of the boss that carries the yoke's pivot out to meet it, in mm.
    idler_boss_diameter_mm: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "base_plate_thickness_mm", "wall_thickness_mm", "bracket_height_mm",
            "gusset_size_mm", "idler_plug_thickness_mm",
            "idler_boss_diameter_mm", "servo_screw_depth_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"ShoulderBracketSpec: {name} must be positive, got "
                    f"{getattr(self, name)}."
                )
        for label, pair in (
            ("base_plate_mm", self.base_plate_mm),
            ("cable_slot_mm", self.cable_slot_mm),
        ):
            if len(pair) != 2 or any(value <= 0.0 for value in pair):
                raise ValueError(
                    f"ShoulderBracketSpec: {label} must be a pair of positive "
                    f"lengths, got {pair}."
                )
        if self.shaft_direction not in ("+Y", "-Y"):
            raise ValueError(
                f"ShoulderBracketSpec: shaft_direction must be '+Y' or '-Y', "
                f"got {self.shaft_direction!r}. The shoulder pitch axis is the "
                "base frame's Y axis; nothing else is meaningful."
            )
        if self.servo_screw_depth_mm >= self.wall_thickness_mm:
            raise ValueError(
                f"ShoulderBracketSpec: a {self.servo_screw_depth_mm} mm blind "
                f"hole in a {self.wall_thickness_mm} mm wall leaves no floor; "
                "the screw would break through."
            )
        if self.bracket_height_mm <= self.base_plate_thickness_mm:
            raise ValueError(
                f"ShoulderBracketSpec: the pitch axis "
                f"({self.bracket_height_mm} mm) must sit above the base plate "
                f"({self.base_plate_thickness_mm} mm thick), not inside it."
            )

    @property
    def shaft_sign(self) -> float:
        """+1 if the output shaft points along +Y, -1 if along -Y."""
        return 1.0 if self.shaft_direction == "+Y" else -1.0

    @property
    def base_plate_x_mm(self) -> float:
        return self.base_plate_mm[0]

    @property
    def base_plate_y_mm(self) -> float:
        return self.base_plate_mm[1]


@dataclass(frozen=True)
class LinkSpec:
    """
    A structural arm link: a hollow rectangular beam between two joint axes.

    Parameterised so L2 and L3 reuse it in Session D.3 rather than growing
    their own near-copies.

    The section is sized by stiffness and by what has to fit inside it, not by
    strength. :meth:`bending_stress_mpa` on the worst-case shoulder moment
    returns about 1.3 MPa against a 25 MPa allowable - a factor of nineteen -
    because a 400 mm cantilever carrying 3.3 N.m is simply not a demanding
    bending problem. What it costs is mass: see :meth:`estimated_mass_g`.
    """

    name: str
    length_mm: float

    #: Across the beam, perpendicular to its length and to gravity, in mm.
    cross_section_width_mm: float = 40.0
    #: The bending depth: vertical, in the plane the link swings in, in mm.
    cross_section_height_mm: float = 25.0
    #: Wall thickness of the hollow section, in mm.
    #:
    #: 3 mm is below ``HardwareSpec.min_wall_thickness_mm`` (4 mm), and that is
    #: deliberate: the 4 mm floor was set for the clamp's load-bearing walls,
    #: and 3 mm is still seven perimeters at a 0.4 mm nozzle. Taking the walls
    #: to 4 mm would add roughly 29% to the mass of a link that already has a
    #: nineteen-fold stress margin. The exemption is recorded in cad/README.md.
    wall_thickness_mm: float = 3.0

    #: (diameter, thickness) of the flange plate at each end, in mm.
    end_flange_mm: Tuple[float, float] = (36.0, 6.0)

    #: (width, depth) of the open cable channel, in mm.
    cable_channel_mm: Tuple[float, float] = (8.0, 8.0)
    #: Spacing of the strain-relief tabs bridging that channel, in mm.
    strain_relief_pitch_mm: float = 80.0
    #: How much of the channel's length each tab covers, in mm.
    strain_relief_tab_width_mm: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "length_mm", "cross_section_width_mm", "cross_section_height_mm",
            "wall_thickness_mm", "strain_relief_pitch_mm",
            "strain_relief_tab_width_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    f"LinkSpec {self.name}: {name} must be positive, got "
                    f"{getattr(self, name)}."
                )
        for label, pair in (
            ("end_flange_mm", self.end_flange_mm),
            ("cable_channel_mm", self.cable_channel_mm),
        ):
            if len(pair) != 2 or any(value <= 0.0 for value in pair):
                raise ValueError(
                    f"LinkSpec {self.name}: {label} must be a pair of positive "
                    f"lengths, got {pair}."
                )
        if 2.0 * self.wall_thickness_mm >= min(
            self.cross_section_width_mm, self.cross_section_height_mm
        ):
            raise ValueError(
                f"LinkSpec {self.name}: walls of {self.wall_thickness_mm} mm "
                f"meet in the middle of a "
                f"{self.cross_section_width_mm} x "
                f"{self.cross_section_height_mm} mm section; there is no "
                "hollow left."
            )
        if self.cable_channel_mm[0] >= self.cross_section_width_mm:
            raise ValueError(
                f"LinkSpec {self.name}: the cable channel "
                f"({self.cable_channel_mm[0]} mm) is as wide as the beam "
                f"({self.cross_section_width_mm} mm)."
            )
        if self.strain_relief_tab_width_mm >= self.strain_relief_pitch_mm:
            raise ValueError(
                f"LinkSpec {self.name}: strain-relief tabs "
                f"({self.strain_relief_tab_width_mm} mm) spaced every "
                f"{self.strain_relief_pitch_mm} mm would close the channel."
            )

    # ---- Section properties -------------------------------------------------

    @property
    def inner_width_mm(self) -> float:
        return self.cross_section_width_mm - 2.0 * self.wall_thickness_mm

    @property
    def inner_height_mm(self) -> float:
        return self.cross_section_height_mm - 2.0 * self.wall_thickness_mm

    @property
    def cross_section_area_mm2(self) -> float:
        """Material in one cross-section of the beam, in mm^2."""
        return float(
            self.cross_section_width_mm * self.cross_section_height_mm
            - self.inner_width_mm * self.inner_height_mm
        )

    @property
    def second_moment_mm4(self) -> float:
        """
        Second moment of area about the bending axis, in mm^4.

        Bending is in the plane the link swings in, so the section's *height*
        is its depth and the width is the flange breadth.
        """
        return float(
            (
                self.cross_section_width_mm * self.cross_section_height_mm**3
                - self.inner_width_mm * self.inner_height_mm**3
            )
            / 12.0
        )

    @property
    def section_modulus_mm3(self) -> float:
        """Second moment divided by the distance to the extreme fibre."""
        return float(self.second_moment_mm4 / (self.cross_section_height_mm / 2.0))

    def bending_stress_mpa(self, moment_nm: float) -> float:
        """
        Peak bending stress in the section under ``moment_nm``, in MPa.

        Raises
        ------
        ValueError
            If ``moment_nm`` is negative.
        """
        if moment_nm < 0.0:
            raise ValueError(f"moment_nm must be non-negative, got {moment_nm}.")
        return float(moment_nm * 1000.0 / self.section_modulus_mm3)

    def tip_deflection_mm(
        self, moment_nm: float, youngs_modulus_mpa: float = 2000.0
    ) -> float:
        """
        Cantilever tip deflection under an equivalent end load, in mm.

        The moment is converted to the end load that would produce it over this
        link's length, then the standard ``P L^3 / 3 E I`` is applied. A hand
        calculation: it ignores shear, the joint compliance at both ends and
        PETG's anisotropy, so treat it as an order of magnitude.
        """
        if moment_nm < 0.0:
            raise ValueError(f"moment_nm must be non-negative, got {moment_nm}.")
        end_load_n = moment_nm * 1000.0 / self.length_mm
        return float(
            end_load_n
            * self.length_mm**3
            / (3.0 * youngs_modulus_mpa * self.second_moment_mm4)
        )

    def estimated_mass_g(self, density_g_cm3: float = PETG_DENSITY_G_CM3) -> float:
        """
        Mass of the beam section alone, in grams.

        Walls this thick print solid, so the section area is real material.
        Excludes the end flanges, the cable channel and the distal servo
        housing, all of which the CAD adds - this is the floor, not the total.
        """
        return float(
            self.cross_section_area_mm2 * self.length_mm / 1000.0 * density_g_cm3
        )

    @property
    def strain_relief_count(self) -> int:
        """How many tabs bridge the cable channel over this link's length."""
        return max(1, int(self.length_mm // self.strain_relief_pitch_mm))


#: L1, the upper arm: shoulder pitch axis to elbow pitch axis.
UPPER_ARM_LINK = LinkSpec(
    name="L1 upper arm",
    length_mm=DEFAULT_ARM.l1_upper_arm_mm,
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
    shoulder_pitch_servo: ServoSpec = field(default_factory=ServoSpec)
    elbow_pitch_servo: ServoSpec = field(default_factory=ServoSpec)

    #: Yaw thrust bearing, in the pedestal's turret. A 6806ZZ since D.2a - see
    #: the note on :class:`BearingSpec`.
    thrust_bearing: BearingSpec = field(default_factory=BearingSpec)

    #: Pivot bearing in the shoulder yoke's undriven side, opposite the servo.
    #: The 608ZZ the yaw joint used to use, kept in the bill of materials.
    shoulder_idler_bearing: BearingSpec = field(
        default_factory=lambda: BEARING_608ZZ
    )

    #: Coupling between a servo shaft and the part it drives. One spec, used by
    #: both the yaw turntable and the upper arm's shoulder end.
    servo_horn: ServoHornSpec = field(default_factory=ServoHornSpec)

    base_stack: BaseStack = field(default_factory=BaseStack)

    #: The three parts designed in Session D.2, between the pedestal and the
    #: elbow.
    yaw_turntable: YawTurntableSpec = field(default_factory=YawTurntableSpec)
    shoulder_bracket: ShoulderBracketSpec = field(
        default_factory=ShoulderBracketSpec
    )
    upper_arm_link: LinkSpec = field(default_factory=lambda: UPPER_ARM_LINK)

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
        # ---- The height budget is stated twice; make it agree. ------------
        # BaseStack owns the budget, the two part specs own the geometry. If
        # they drift, the shoulder pivot silently stops landing on
        # base_height_mm, which is exactly the class of error Session D.1d
        # spent its time on.
        if (
            abs(
                self.base_stack.turntable_plate_thickness_mm
                - self.yaw_turntable.thickness_mm
            )
            > 1e-9
        ):
            raise ValueError(
                f"HardwareSpec: BaseStack budgets "
                f"{self.base_stack.turntable_plate_thickness_mm} mm for the "
                f"turntable plate but YawTurntableSpec is "
                f"{self.yaw_turntable.thickness_mm} mm thick."
            )
        if (
            abs(
                self.base_stack.shoulder_bracket_rise_mm
                - self.shoulder_bracket.bracket_height_mm
            )
            > 1e-9
        ):
            raise ValueError(
                f"HardwareSpec: BaseStack budgets "
                f"{self.base_stack.shoulder_bracket_rise_mm} mm for the "
                f"shoulder bracket but ShoulderBracketSpec rises "
                f"{self.shoulder_bracket.bracket_height_mm} mm."
            )
        # ---- The horn has to fit the bearing it passes through. -----------
        horn, bearing = self.servo_horn, self.thrust_bearing
        if horn.disc_diameter_mm >= bearing.bore_diameter_mm:
            raise ValueError(
                f"HardwareSpec: the {horn.name} is "
                f"{horn.disc_diameter_mm} mm across but the "
                f"{bearing.name}'s bore is {bearing.bore_diameter_mm} mm. The "
                "coupling sits inside the bore, so it cannot be wider than it. "
                "Fit a smaller horn or a larger-bore bearing."
            )
        if horn.total_height_mm >= bearing.width_mm:
            raise ValueError(
                f"HardwareSpec: the {horn.name} stands "
                f"{horn.total_height_mm} mm above the servo's shaft boss, "
                f"which is the whole width of the {bearing.name} "
                f"({bearing.width_mm} mm). The turntable would ride the horn "
                "instead of the bearing. Fit a thinner horn, or lower the "
                "servo in the turret - there is spare wall below the bearing "
                "seat to spend."
            )

    @property
    def above_pedestal_allowance_mm(self) -> float:
        """
        Everything between the pedestal's top face and the shoulder pivot, in mm.

        The bearing's proud height plus the turntable and bracket budget. Kept
        as one accessor so the pedestal height and the assembly preview cannot
        disagree about what occupies that space.
        """
        return self.thrust_bearing.proud_mm + self.base_stack.allowance_mm

    def shoulder_pivot_z_mm(self, arm: Optional[ArmGeometry] = None) -> float:
        """
        Height of the shoulder pivot above the desk surface, in mm.

        This is ``ArmGeometry.base_height_mm`` by definition; the method exists
        so the CAD can assert that its own stack actually lands there rather
        than assuming it.
        """
        arm = DEFAULT_ARM if arm is None else arm
        return float(arm.base_height_mm)

    def pedestal_height_mm(self, arm: Optional[ArmGeometry] = None) -> float:
        """
        Height of the base pedestal alone, in mm.

        Derived as ``base_height_mm`` less everything stacked above the
        pedestal's top face -- the bearing's proud height, the yaw turntable
        and the shoulder bracket -- so the pedestal plus that stack lands the
        shoulder pivot at exactly ``base_height_mm``.

        The bearing's proud height was folded in during Session D.1d. Leaving
        it out had put the pivot 0.5 mm high: the turntable rides the bearing's
        inner race, which stands above the turret's top face rather than flush
        with it.

        Raises
        ------
        ValueError
            If the stack consumes the entire base height, which would call for
            a pedestal of zero or negative height.
        """
        arm = DEFAULT_ARM if arm is None else arm
        height = arm.base_height_mm - self.above_pedestal_allowance_mm
        if height <= 0.0:
            raise ValueError(
                f"Base stack ({self.above_pedestal_allowance_mm:.1f} mm) "
                f"meets or exceeds base_height_mm ({arm.base_height_mm:.1f} mm); "
                "the pedestal would have non-positive height. Either shorten "
                "the turntable/bracket budget or raise base_height_mm."
            )
        return float(height)

    def bill_of_materials(self) -> Tuple[str, ...]:
        """Off-the-shelf parts to buy, one human-readable line each."""
        clamp = self.desk_clamp
        horn, idler = self.servo_horn, self.shoulder_idler_bearing
        return (
            f"1 x {self.base_yaw_servo.name} servo (base yaw)",
            f"1 x {self.shoulder_pitch_servo.name} servo (shoulder pitch)",
            f"1 x {self.elbow_pitch_servo.name} servo (elbow pitch)",
            f"2 x {horn.name} ({horn.disc_diameter_mm:.0f} mm disc, "
            f"{horn.bolt_count} x M{horn.bolt_nominal_diameter_mm:.0f} on a "
            f"{horn.bolt_circle_mm:.0f} mm circle) - yaw turntable and "
            f"shoulder",
            f"1 x {self.thrust_bearing.name} bearing "
            f"({self.thrust_bearing.bore_diameter_mm:.0f} x "
            f"{self.thrust_bearing.outer_diameter_mm:.0f} x "
            f"{self.thrust_bearing.width_mm:.0f} mm) - yaw thrust",
            f"1 x {idler.name} bearing "
            f"({idler.bore_diameter_mm:.0f} x "
            f"{idler.outer_diameter_mm:.0f} x "
            f"{idler.width_mm:.0f} mm) - shoulder yoke idler",
            f"1 x {clamp.bolt_thread} x {clamp.bolt_length_mm:.0f} mm hex-head "
            f"machine screw (DIN 933)",
            f"1 x {clamp.bolt_thread} hex nut (DIN 934), "
            f"{clamp.nut_across_flats_mm:.0f} mm across flats",
            f"1 x rubber anti-slip sheet, {clamp.pad_thickness_mm:.1f} mm thick "
            f"(cut 2 pads, glued into the jaw recesses)",
            "12 x M3 screws (servo retention, 4 per servo)",
            f"{2 * horn.bolt_count} x "
            f"M{horn.bolt_nominal_diameter_mm:.0f} socket-head screws "
            f"(turntable and upper arm onto their horns)",
            f"{self.yaw_turntable.bracket_bolt_count} x "
            f"M{self.yaw_turntable.bracket_bolt_nominal_diameter_mm:.0f} "
            f"screws (shoulder bracket onto the turntable)",
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
            f"    bearing proud      : {self.thrust_bearing.proud_mm:.1f} mm\n"
            f"    turntable plate    : "
            f"{self.base_stack.turntable_plate_thickness_mm:.1f} mm\n"
            f"    shoulder bracket   : "
            f"{self.base_stack.shoulder_bracket_rise_mm:.1f} mm\n"
            f"    -> pedestal        : {self.pedestal_height_mm(arm):.1f} mm\n"
            f"\n"
            f"  Yaw turntable        : {self.yaw_turntable.diameter_mm:.1f} mm "
            f"dia x {self.yaw_turntable.thickness_mm:.1f} mm, bracket bolts on "
            f"a {self.yaw_turntable.bracket_bolt_pattern_mm[0]:.0f} x "
            f"{self.yaw_turntable.bracket_bolt_pattern_mm[1]:.0f} mm rectangle\n"
            f"  Shoulder bracket     : "
            f"{self.shoulder_bracket.base_plate_x_mm:.0f} x "
            f"{self.shoulder_bracket.base_plate_y_mm:.0f} mm base, pitch axis "
            f"{self.shoulder_bracket.bracket_height_mm:.1f} mm up, shaft "
            f"{self.shoulder_bracket.shaft_direction}\n"
            f"  Upper arm (L1)       : {self.upper_arm_link.length_mm:.0f} mm, "
            f"{self.upper_arm_link.cross_section_width_mm:.0f} x "
            f"{self.upper_arm_link.cross_section_height_mm:.0f} mm section, "
            f"{self.upper_arm_link.wall_thickness_mm:.0f} mm walls\n"
            f"    bending stress     : "
            f"{self.upper_arm_link.bending_stress_mpa(arm.shoulder_moment_nm()):.2f} "
            f"MPa at {arm.shoulder_moment_nm():.2f} N.m "
            f"(allowable {PETG_ALLOWABLE_STRESS_MPA:.0f})\n"
            f"    beam mass          : "
            f"{self.upper_arm_link.estimated_mass_g():.0f} g of PETG\n"
            f"  Servo horn           : {self.servo_horn.name}, "
            f"{self.servo_horn.disc_diameter_mm:.1f} mm disc x "
            f"{self.servo_horn.total_height_mm:.1f} mm tall, indexes every "
            f"{self.servo_horn.index_resolution_deg:.1f} deg\n"
            f"\n"
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
            f"\n"
            f"  !! {self.servo_horn.unverified_report()}\n"
        )


# Default hardware singleton, paired with DEFAULT_ARM.
DEFAULT_HARDWARE = HardwareSpec()


if __name__ == "__main__":
    print(DEFAULT_ARM.coverage_report())
    print(DEFAULT_HARDWARE.hardware_report())
