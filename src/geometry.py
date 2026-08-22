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
class HardwareSpec:
    """
    Every off-the-shelf component the CAD needs, in one place.

    Held separately from :class:`ArmGeometry` because these are bill-of-
    materials facts rather than kinematic parameters, but kept in this module
    so downstream code still has a single import for "the robot's numbers".
    """

    base_yaw_servo: ServoSpec = field(default_factory=ServoSpec)
    thrust_bearing: BearingSpec = field(default_factory=BearingSpec)
    mounting_fastener: FastenerSpec = field(default_factory=FastenerSpec)
    base_stack: BaseStack = field(default_factory=BaseStack)

    #: Bolt-circle diameter for bolting the pedestal flange to the desk.
    mount_bolt_circle_diameter_mm: float = 60.0

    #: Number of desk-mounting bolts, evenly spaced on the bolt circle.
    mount_bolt_count: int = 4

    #: Per-surface gap added to printed pockets so parts actually fit. FDM
    #: prints slightly undersize on internal features; 0.25 mm per wall is a
    #: conservative starting point for PETG on a well-tuned machine.
    print_clearance_mm: float = 0.25

    #: Minimum printed wall thickness. Four 0.4 mm perimeters plus margin.
    min_wall_thickness_mm: float = 4.0

    def __post_init__(self) -> None:
        if self.mount_bolt_count < 3:
            raise ValueError(
                "HardwareSpec: at least 3 mounting bolts are needed to "
                f"constrain the flange, got {self.mount_bolt_count}."
            )
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

    def hardware_report(self, arm: Optional[ArmGeometry] = None) -> str:
        """Human-readable component summary, including verification warnings."""
        arm = DEFAULT_ARM if arm is None else arm
        servo, bearing, fastener = (
            self.base_yaw_servo, self.thrust_bearing, self.mounting_fastener
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
            f"  Mounting fastener    : {fastener.name} x "
            f"{self.mount_bolt_count} on a "
            f"{self.mount_bolt_circle_diameter_mm:.0f} mm bolt circle\n"
            f"  Base height budget   : {arm.base_height_mm:.1f} mm total\n"
            f"    turntable plate    : "
            f"{self.base_stack.turntable_plate_thickness_mm:.1f} mm\n"
            f"    shoulder bracket   : "
            f"{self.base_stack.shoulder_bracket_rise_mm:.1f} mm\n"
            f"    -> pedestal        : {self.pedestal_height_mm(arm):.1f} mm\n"
            f"  Print clearance      : {self.print_clearance_mm:.2f} mm/surface\n"
            f"  Min wall thickness   : {self.min_wall_thickness_mm:.1f} mm\n"
            f"\n"
            f"  !! {servo.unverified_report()}\n"
        )


# Default hardware singleton, paired with DEFAULT_ARM.
DEFAULT_HARDWARE = HardwareSpec()


if __name__ == "__main__":
    print(DEFAULT_ARM.coverage_report())
    print(DEFAULT_HARDWARE.hardware_report())
