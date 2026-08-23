"""
Base pedestal -- a monolithic U-clamp that wraps the desk edge and carries the
base yaw joint.

Run directly to write ``cad/output/base_pedestal.stl``::

    python3 -m cad.base_pedestal
    python3 -m cad.base_pedestal --output /tmp/pedestal.stl
    python3 -m cad.base_pedestal --report      # dimensions only, no export

Shape
-----
One solid body in the shape of a C, seen from the side, with a servo turret
rising from the top arm::

                    +----------+  z = +70   bearing seat in the top face
                    |  servo   |
                    |  turret  |
        +-----------+----------+---------+  z = +15   top arm top
        |///|  pads      cavity opening  |  z =   0   desk surface
        |///+------------------+---------+
        |///|                            |
        |sp |        throat (desk)        |
        |ine|                            |
        |///+------------------+---------+  z = -45   bottom arm top
        |///|      nut pocket  | O bolt  |
        +---+------------------+---------+  z = -60   bottom arm underside
         x=-45   x=-30                x=+30
                  ^ desk edge

Local frame: the origin sits **on the yaw axis at the desk's top surface**, so
``z = 0`` is the desk plane and ``+X`` points inward over the desk. The desk
edge is therefore at ``x = -servo_shaft_offset_from_edge_mm``. This choice is
what lets ``ArmGeometry.base_height_mm`` (desk surface to shoulder pivot) and
the clamp geometry share one datum without a conversion step.

Why a turret rather than a thick arm
------------------------------------
The DS3218 is 40.5 mm tall, and with print clearance, the shaft boss, a
ceiling and the bearing seat it needs 55.5 mm of housing. A 15 mm top arm
cannot contain it. Thickening the arm to 55 mm would make the part a slab
rather than a C-profile and triple its material, so the arm stays thin and the
servo lives in a turret above it, reaching the 70 mm pedestal height exactly.

Why the screw points up
-----------------------
The nut is captive in a pocket in the bottom arm's **underside**; the screw
threads up through it, and its tip carries a printed pressure foot that bears
on the desk's underside. The knob hangs below on the screw's head. A screw
entering from the top of the bottom arm would press against nothing and could
not clamp at all.

Why gussets rather than fillets
-------------------------------
The two inner corners of the U carry the whole load path and want stress
relief, but a swept fillet is the most fragile operation to re-run when an
upstream dimension moves -- and every dimension here derives from
``src/geometry.py`` precisely so that it *can* move. Triangular gussets give
most of the benefit and always regenerate.

Sizing philosophy
-----------------
Nothing below is a magic number. The clamp's width is derived from what the
servo cavity and bearing seat need; the screw is placed as far outboard as the
pressure foot allows, because that simultaneously shortens the bottom arm's
cantilever and lengthens the lever it resists tipping with.
:meth:`PedestalParameters.validate` re-checks every clearance and refuses to
build a part that would print with a wall thinner than specified, a pocket
breaking into another, or a screw too short to reach the desk.

Units are millimetres throughout, matching ``src.geometry`` and STL.

.. warning::
   Several ``ServoSpec`` fields this part depends on -- notably the mounting
   flange span, ear position and shaft offset -- are UNVERIFIED placeholders.
   See ``ServoSpec.UNVERIFIED_FIELDS``. The generated STL is geometrically
   valid, but measure a real DS3218 before printing for final assembly.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from build123d import Align, Box, Cylinder, Part, Pos, Rot, export_stl

from cad._design import DesignRuleError, DesignStatus
from cad._primitives import hex_prism, right_triangle_prism
from src.geometry import (
    DEFAULT_ARM,
    DEFAULT_HARDWARE,
    ArmGeometry,
    HardwareSpec,
)

__all__ = [
    "DesignStatus",
    "PedestalDesignError",
    "PedestalParameters",
    "build_pedestal",
    "export_pedestal",
    "DEFAULT_STL_PATH",
]

#: Where ``python3 -m cad.base_pedestal`` writes by default.
DEFAULT_STL_PATH = Path(__file__).resolve().parent / "output" / "base_pedestal.stl"

#: (x_min, x_max, y_min, y_max) of one anti-slip pad recess, in mm.
PadRect = Tuple[float, float, float, float]


class PedestalDesignError(DesignRuleError):
    """
    Raised when a parameter set would produce an unbuildable pedestal.

    Carries a :class:`~cad._design.DesignStatus` naming the violated
    constraint, so a caller sweeping parameters can tell "servo too big" from
    "clamp screw too short".
    """


@dataclass(frozen=True)
class PedestalParameters:
    """
    Fully resolved U-clamp dimensions, in millimetres.

    Build these with :meth:`from_geometry` rather than by hand; the direct
    constructor exists so tests can inject deliberately-broken values and
    confirm :meth:`validate` catches them.
    """

    # ---- U-profile envelope -----------------------------------------------
    clamp_width_mm: float
    desk_seat_x_mm: float
    spine_inner_x_mm: float
    spine_outer_x_mm: float
    top_arm_inner_x_mm: float
    top_arm_thickness_mm: float
    bottom_arm_inner_x_mm: float
    bottom_arm_thickness_mm: float
    throat_opening_mm: float
    gusset_size_mm: float

    # ---- Servo turret -----------------------------------------------------
    turret_top_z_mm: float
    turret_x_min_mm: float
    turret_x_max_mm: float
    turret_y_min_mm: float
    turret_y_max_mm: float

    # ---- Servo cavity (long axis across the clamp, along Y) ---------------
    cavity_x_span_mm: float
    cavity_body_span_y_mm: float
    cavity_ear_span_y_mm: float
    cavity_offset_y_mm: float
    cavity_top_z_mm: float
    ear_shelf_z_mm: float

    # ---- Shaft bore and bearing seat --------------------------------------
    shaft_bore_diameter_mm: float
    bearing_seat_diameter_mm: float
    bearing_seat_depth_mm: float
    bearing_proud_mm: float

    # ---- Anti-slip pads on the top arm's underside ------------------------
    pad_recess_depth_mm: float
    pad_edge_margin_mm: float

    # ---- Clamping screw ---------------------------------------------------
    bolt_axis_x_mm: float
    bolt_hole_diameter_mm: float
    nut_pocket_across_flats_mm: float
    nut_pocket_depth_mm: float
    pressure_foot_diameter_mm: float

    # ---- Servo retention --------------------------------------------------
    servo_screw_hole_diameter_mm: float
    servo_screw_hole_depth_mm: float
    servo_screw_spacing_y_mm: float
    servo_screw_spacing_x_mm: float

    # ---- Cable exit -------------------------------------------------------
    cable_slot_width_mm: float
    cable_slot_height_mm: float

    # ---- Print constraints carried through for validation -----------------
    min_wall_thickness_mm: float

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_geometry(
        cls,
        arm: Optional[ArmGeometry] = None,
        hardware: Optional[HardwareSpec] = None,
        *,
        ear_top_offset_from_body_top_mm: float = 10.0,
        cable_slot_width_mm: float = 8.0,
        cable_slot_height_mm: float = 5.0,
        servo_screw_hole_depth_mm: float = 6.0,
    ) -> "PedestalParameters":
        """
        Derive every dimension from the geometry and hardware singletons.

        Parameters
        ----------
        arm, hardware:
            Sources of truth. Default to ``DEFAULT_ARM`` / ``DEFAULT_HARDWARE``;
            injectable so tests can vary the design without touching globals.
        ear_top_offset_from_body_top_mm:
            Distance from the top of the servo body down to the top of its
            mounting ears. UNVERIFIED placeholder -- see the module warning.
        cable_slot_width_mm, cable_slot_height_mm:
            Radial slot letting the servo lead out through the turret's
            inboard wall.
        servo_screw_hole_depth_mm:
            Depth of the blind M3 pilot holes in the retention shelf.

        Raises
        ------
        PedestalDesignError
            If a supplied parameter is out of range, or if the resulting
            design violates a clearance (see :meth:`validate`).
        """
        arm = DEFAULT_ARM if arm is None else arm
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        servo = hardware.base_yaw_servo
        bearing = hardware.thrust_bearing
        clamp = hardware.desk_clamp
        clearance = hardware.print_clearance_mm
        wall = hardware.min_wall_thickness_mm

        for name, value in (
            ("ear_top_offset_from_body_top_mm", ear_top_offset_from_body_top_mm),
            ("cable_slot_width_mm", cable_slot_width_mm),
            ("cable_slot_height_mm", cable_slot_height_mm),
            ("servo_screw_hole_depth_mm", servo_screw_hole_depth_mm),
        ):
            if value <= 0.0:
                raise PedestalDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )
        # ---- Vertical layout, resolved from the turret's top face down ----
        try:
            turret_top_z = hardware.pedestal_height_mm(arm)
        except ValueError as exc:
            raise PedestalDesignError(DesignStatus.NEGATIVE_HEIGHT, str(exc)) from exc

        # The bearing's proud height is already subtracted from the pedestal
        # budget by HardwareSpec.pedestal_height_mm, so the seat depth follows
        # from the same single source rather than a local constant.
        seat_depth = bearing.seat_depth_mm
        seat_bottom_z = turret_top_z - seat_depth
        # The servo's output boss must reach up to the underside of the
        # bearing, so the body top sits one boss-height below the seat floor.
        cavity_top_z = seat_bottom_z - servo.shaft_boss_height_mm
        ear_shelf_z = cavity_top_z - ear_top_offset_from_body_top_mm

        # ---- Cavity cross-section -----------------------------------------
        # The servo's long axis runs ACROSS the clamp (along Y). Along X it
        # would need 54.5 mm inside a 60 mm arm and would break out past the
        # spine, since the body sits off-centre from the shaft.
        cavity_x_span = servo.body_width_mm + 2.0 * clearance
        cavity_body_span_y = servo.body_length_mm + 2.0 * clearance
        cavity_ear_span_y = servo.flange_span_mm + 2.0 * clearance
        cavity_offset_y = -servo.body_offset_from_shaft_axis_mm

        # ---- U-profile envelope -------------------------------------------
        # The gusset at the top inner corner hangs into the throat, so the
        # desk's edge comes to rest against it rather than against the spine.
        # That gusset line -- not the spine face -- is therefore the real desk
        # seating plane, and the one the shaft offset is measured from. Placing
        # the spine a gusset further out keeps the shaft exactly
        # servo_shaft_offset_from_edge_mm from where the desk actually stops.
        desk_seat_x = -clamp.servo_shaft_offset_from_edge_mm
        spine_inner_x = desk_seat_x - clamp.gusset_size_mm
        spine_outer_x = spine_inner_x - clamp.spine_thickness_mm
        top_arm_inner_x = desk_seat_x + clamp.top_arm_depth_mm
        bottom_arm_inner_x = desk_seat_x + clamp.bottom_arm_depth_mm

        # ---- Turret: whatever encloses the cavity AND the bearing seat ----
        seat_diameter = bearing.seat_diameter_mm
        turret_x_max = max(cavity_x_span / 2.0, seat_diameter / 2.0) + wall
        turret_x_min = -turret_x_max
        turret_y_min = min(
            cavity_offset_y - cavity_ear_span_y / 2.0, -seat_diameter / 2.0
        ) - wall
        turret_y_max = max(
            cavity_offset_y + cavity_ear_span_y / 2.0, seat_diameter / 2.0
        ) + wall

        # Symmetric about the yaw axis, wide enough for the turret.
        clamp_width = 2.0 * max(abs(turret_y_min), abs(turret_y_max))

        # ---- Clamping screw ------------------------------------------------
        # As far outboard as the pressure foot allows: that shortens the bottom
        # arm's cantilever AND lengthens the lever resisting tipping.
        bolt_axis_x = (
            desk_seat_x
            + clamp.pad_edge_margin_mm
            + clamp.pressure_foot_diameter_mm / 2.0
        )

        params = cls(
            clamp_width_mm=clamp_width,
            desk_seat_x_mm=desk_seat_x,
            spine_inner_x_mm=spine_inner_x,
            spine_outer_x_mm=spine_outer_x,
            top_arm_inner_x_mm=top_arm_inner_x,
            top_arm_thickness_mm=clamp.top_arm_thickness_mm,
            bottom_arm_inner_x_mm=bottom_arm_inner_x,
            bottom_arm_thickness_mm=clamp.bottom_arm_thickness_mm,
            throat_opening_mm=clamp.throat_max_opening_mm,
            gusset_size_mm=clamp.gusset_size_mm,
            turret_top_z_mm=turret_top_z,
            turret_x_min_mm=turret_x_min,
            turret_x_max_mm=turret_x_max,
            turret_y_min_mm=turret_y_min,
            turret_y_max_mm=turret_y_max,
            cavity_x_span_mm=cavity_x_span,
            cavity_body_span_y_mm=cavity_body_span_y,
            cavity_ear_span_y_mm=cavity_ear_span_y,
            cavity_offset_y_mm=cavity_offset_y,
            cavity_top_z_mm=cavity_top_z,
            ear_shelf_z_mm=ear_shelf_z,
            shaft_bore_diameter_mm=servo.shaft_boss_diameter_mm + 2.0 * clearance,
            bearing_seat_diameter_mm=seat_diameter,
            bearing_seat_depth_mm=seat_depth,
            bearing_proud_mm=bearing.proud_mm,
            pad_recess_depth_mm=clamp.pad_recess_depth_mm,
            pad_edge_margin_mm=clamp.pad_edge_margin_mm,
            bolt_axis_x_mm=bolt_axis_x,
            bolt_hole_diameter_mm=clamp.bolt_clearance_hole_diameter_mm,
            nut_pocket_across_flats_mm=clamp.nut_across_flats_mm + 2.0 * clearance,
            nut_pocket_depth_mm=clamp.nut_pocket_depth_mm,
            pressure_foot_diameter_mm=clamp.pressure_foot_diameter_mm,
            servo_screw_hole_diameter_mm=servo.flange_hole_diameter_mm,
            servo_screw_hole_depth_mm=servo_screw_hole_depth_mm,
            servo_screw_spacing_y_mm=servo.flange_hole_spacing_long_mm,
            servo_screw_spacing_x_mm=servo.flange_hole_spacing_short_mm,
            cable_slot_width_mm=cable_slot_width_mm,
            cable_slot_height_mm=cable_slot_height_mm,
            min_wall_thickness_mm=wall,
        )
        params.validate()

        # The screw spans a stack this part only half determines, so the check
        # lives here where both halves are known.
        if clamp.bolt_length_mm < clamp.required_bolt_length_mm:
            raise PedestalDesignError(
                DesignStatus.FASTENER_TOO_SHORT,
                f"{clamp.bolt_thread} x {clamp.bolt_length_mm:.1f} mm cannot "
                f"span the clamp stack, which needs "
                f"{clamp.required_bolt_length_mm:.1f} mm to reach the thinnest "
                f"supported desk ({clamp.min_desk_thickness_mm:.0f} mm). Fit a "
                "longer screw or narrow the throat.",
            )
        return params

    # =====================================================================
    # Derived accessors
    # =====================================================================

    @property
    def throat_top_z_mm(self) -> float:
        """The desk's top surface: the underside of the top arm."""
        return 0.0

    @property
    def throat_bottom_z_mm(self) -> float:
        """Top face of the bottom arm."""
        return -self.throat_opening_mm

    @property
    def bottom_arm_bottom_z_mm(self) -> float:
        """Underside of the bottom arm -- the lowest point of the part."""
        return self.throat_bottom_z_mm - self.bottom_arm_thickness_mm

    @property
    def overall_height_mm(self) -> float:
        """Full printed Z extent, turret top to bottom arm underside."""
        return self.turret_top_z_mm - self.bottom_arm_bottom_z_mm

    @property
    def servo_shaft_output_z_mm(self) -> float:
        """
        Height of the servo's output shaft crown above the desk surface, in mm.

        This is where the yaw drive actually emerges: the top of the shaft
        boss, level with the bearing seat's floor. Exposed as a property so
        the assembly preview and the tests read the same number the solid was
        cut from, and it cannot drift from the geometry.
        """
        return self.turret_top_z_mm - self.bearing_seat_depth_mm

    @property
    def bearing_top_z_mm(self) -> float:
        """
        Height of the thrust bearing's upper face above the desk, in mm.

        The bearing stands proud of the turret, so this -- not
        :attr:`turret_top_z_mm` -- is the surface the yaw turntable rests on,
        and the datum the rest of the base stack is measured from.
        """
        return self.turret_top_z_mm + self.bearing_proud_mm

    @property
    def top_arm_depth_mm(self) -> float:
        """Reach inward from the desk's seating plane, not from the spine."""
        return self.top_arm_inner_x_mm - self.desk_seat_x_mm

    @property
    def bottom_arm_overhang_mm(self) -> float:
        """
        Cantilever length of the bottom arm, spine to screw.

        The screw's reaction bends the bottom arm about the spine, so this is
        the lever that sets its bending stress.
        """
        return self.bolt_axis_x_mm - self.spine_inner_x_mm

    @property
    def tipping_lever_arm_mm(self) -> float:
        """
        Lever between the tipping pivot and the clamp screw.

        The arm reaches inward, so a tipping moment lifts the spine side. The
        assembly would rotate about the top arm's inboard edge -- its last line
        of contact with the desk -- and the screw's preload resists at
        :attr:`bolt_axis_x_mm`.
        """
        return self.top_arm_inner_x_mm - self.bolt_axis_x_mm

    @property
    def pad_recesses(self) -> Tuple[PadRect, ...]:
        """
        The anti-slip pad recesses in the top arm's underside.

        Two strips flanking the servo cavity's opening, which lands in the
        middle of that face. Placed clear of the top gusset, which hangs into
        the throat near the spine.
        """
        margin = self.pad_edge_margin_mm
        cavity_half_x = self.cavity_x_span_mm / 2.0
        y_min = -self.clamp_width_mm / 2.0 + margin
        y_max = self.clamp_width_mm / 2.0 - margin
        return (
            (
                self.desk_seat_x_mm + margin,
                -cavity_half_x - margin,
                y_min,
                y_max,
            ),
            (
                cavity_half_x + margin,
                self.top_arm_inner_x_mm - margin,
                y_min,
                y_max,
            ),
        )

    @property
    def pad_area_mm2(self) -> float:
        """Total anti-slip contact area, in square millimetres."""
        return sum(
            (x_max - x_min) * (y_max - y_min)
            for x_min, x_max, y_min, y_max in self.pad_recesses
        )

    @property
    def nut_pocket_across_corners_mm(self) -> float:
        """Hexagon across-corners for the captive nut's pocket."""
        return self.nut_pocket_across_flats_mm * 2.0 / math.sqrt(3.0)

    @property
    def servo_screw_positions(self) -> Tuple[Tuple[float, float], ...]:
        """(x, y) centres of the four M3 servo-retention pilot holes."""
        return tuple(
            (
                sx * self.servo_screw_spacing_x_mm / 2.0,
                self.cavity_offset_y_mm + sy * self.servo_screw_spacing_y_mm / 2.0,
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )

    # =====================================================================
    # Validation
    # =====================================================================

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unprintable part.

        Returns :attr:`DesignStatus.OK` on success. This is a design rule
        check, not a modelling detail: it is what makes the module safe to
        drive from swept parameters, because a violating parameter set fails
        loudly here instead of silently producing a part with a 0.2 mm wall.

        Raises
        ------
        PedestalDesignError
            Naming the first violated constraint.
        """
        # ---- U-profile sanity ---------------------------------------------
        if self.turret_top_z_mm <= self.top_arm_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.NEGATIVE_HEIGHT,
                f"The turret must rise above the top arm: turret top "
                f"{self.turret_top_z_mm:.2f} mm vs arm thickness "
                f"{self.top_arm_thickness_mm:.2f} mm.",
            )
        if self.desk_seat_x_mm >= self.top_arm_inner_x_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"The top arm has no depth: desk seats at "
                f"{self.desk_seat_x_mm:.2f} mm, inner end at "
                f"{self.top_arm_inner_x_mm:.2f} mm.",
            )
        if not self.spine_outer_x_mm < self.spine_inner_x_mm <= self.desk_seat_x_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"Expected spine outer ({self.spine_outer_x_mm:.2f}) < spine "
                f"inner ({self.spine_inner_x_mm:.2f}) <= desk seat "
                f"({self.desk_seat_x_mm:.2f}); the gusset must reach the desk "
                "seating plane exactly.",
            )
        if self.pad_recess_depth_mm >= self.top_arm_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"pad_recess_depth_mm ({self.pad_recess_depth_mm}) must be "
                f"less than the top arm thickness "
                f"({self.top_arm_thickness_mm}), or the recess cuts through.",
            )
        if self.gusset_size_mm >= self.throat_opening_mm / 2.0:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Gussets of {self.gusset_size_mm:.2f} mm would consume the "
                f"{self.throat_opening_mm:.2f} mm throat from both sides.",
            )

        # ---- Vertical layout ordering, top arm up through the turret ------
        seat_bottom_z = self.turret_top_z_mm - self.bearing_seat_depth_mm
        if not (
            self.top_arm_thickness_mm
            < self.ear_shelf_z_mm
            < self.cavity_top_z_mm
            < seat_bottom_z
        ):
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                "Vertical layout is out of order: expected top arm "
                f"({self.top_arm_thickness_mm:.2f}) < ear shelf "
                f"({self.ear_shelf_z_mm:.2f}) < cavity top "
                f"({self.cavity_top_z_mm:.2f}) < bearing seat floor "
                f"({seat_bottom_z:.2f}). The servo does not fit in the "
                "available turret height.",
            )
        ceiling = seat_bottom_z - self.cavity_top_z_mm
        if ceiling < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {ceiling:.2f} mm of material between the servo cavity "
                f"ceiling and the bearing seat floor; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )

        # ---- Turret must enclose the cavity and the seat -------------------
        cavity_half_x = self.cavity_x_span_mm / 2.0
        if self.turret_x_max_mm - cavity_half_x < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {self.turret_x_max_mm - cavity_half_x:.2f} mm of turret "
                f"wall beside the servo cavity; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )
        seat_radius = self.bearing_seat_diameter_mm / 2.0
        if self.turret_x_max_mm - seat_radius < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {self.turret_x_max_mm - seat_radius:.2f} mm between the "
                f"bearing seat and the turret's side; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )
        cavity_y_min = self.cavity_offset_y_mm - self.cavity_ear_span_y_mm / 2.0
        cavity_y_max = self.cavity_offset_y_mm + self.cavity_ear_span_y_mm / 2.0
        if (cavity_y_min - self.turret_y_min_mm) < self.min_wall_thickness_mm or (
            self.turret_y_max_mm - cavity_y_max
        ) < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                "The servo cavity reaches within "
                f"{self.min_wall_thickness_mm:.2f} mm of the turret's ends.",
            )
        if self.shaft_bore_diameter_mm >= self.bearing_seat_diameter_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Shaft bore ({self.shaft_bore_diameter_mm:.2f} mm) is not "
                f"smaller than the bearing seat "
                f"({self.bearing_seat_diameter_mm:.2f} mm), so the seat would "
                "have no floor for the outer race to sit on.",
            )

        # ---- Turret must sit on the top arm, not overhang it --------------
        if (
            self.turret_x_min_mm < self.spine_inner_x_mm
            or self.turret_x_max_mm > self.top_arm_inner_x_mm
        ):
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The turret spans x = {self.turret_x_min_mm:.2f} .. "
                f"{self.turret_x_max_mm:.2f} mm but the top arm only covers "
                f"{self.spine_inner_x_mm:.2f} .. {self.top_arm_inner_x_mm:.2f} mm, "
                "so it would hang unsupported over the desk edge.",
            )
        half_width = self.clamp_width_mm / 2.0
        if self.turret_y_min_mm < -half_width or self.turret_y_max_mm > half_width:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The turret is wider than the clamp "
                f"({self.clamp_width_mm:.2f} mm).",
            )

        # ---- Anti-slip pads ------------------------------------------------
        for index, (x_min, x_max, y_min, y_max) in enumerate(self.pad_recesses):
            if x_max - x_min <= 0.0 or y_max - y_min <= 0.0:
                raise PedestalDesignError(
                    DesignStatus.FEATURE_COLLISION,
                    f"Pad recess {index} has no area: the servo cavity opening "
                    "and the gusset leave no room on the top arm's underside.",
                )

        # ---- Clamping screw and captive nut --------------------------------
        if self.bottom_arm_overhang_mm <= 0.0:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                "The clamp screw lies outboard of the desk edge; its pressure "
                "foot would press on air rather than the desk underside.",
            )
        foot_outer_x = (
            self.bolt_axis_x_mm - self.pressure_foot_diameter_mm / 2.0
        )
        if foot_outer_x < self.desk_seat_x_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The pressure foot reaches x = {foot_outer_x:.2f} mm, outboard "
                f"of where the desk seats ({self.desk_seat_x_mm:.2f} mm), so it "
                "would press on the bottom gusset rather than the desk.",
            )
        if self.bolt_axis_x_mm + self.pressure_foot_diameter_mm / 2.0 > (
            self.bottom_arm_inner_x_mm
        ):
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                "The pressure foot overhangs the bottom arm's inboard end.",
            )
        nut_floor = self.bottom_arm_thickness_mm - self.nut_pocket_depth_mm
        if nut_floor < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {nut_floor:.2f} mm of floor above the nut pocket; "
                f"{self.min_wall_thickness_mm:.2f} mm required. This floor "
                "carries the whole clamp load.",
            )
        if self.bolt_hole_diameter_mm >= self.nut_pocket_across_flats_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Bolt hole ({self.bolt_hole_diameter_mm:.2f} mm) is not "
                f"smaller than the nut pocket across flats "
                f"({self.nut_pocket_across_flats_mm:.2f} mm), so the nut would "
                "have no shoulder to bear against.",
            )
        nut_half = self.nut_pocket_across_corners_mm / 2.0
        if half_width - nut_half < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                "The nut pocket reaches within "
                f"{self.min_wall_thickness_mm:.2f} mm of the clamp's side.",
            )
        if (self.bolt_axis_x_mm - nut_half) - self.spine_inner_x_mm < (
            self.min_wall_thickness_mm
        ):
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                "The nut pocket reaches within "
                f"{self.min_wall_thickness_mm:.2f} mm of the spine.",
            )

        # ---- Servo retention screws must land in shelf material -----------
        shelf_step = (
            self.cavity_ear_span_y_mm - self.cavity_body_span_y_mm
        ) / 2.0
        if shelf_step <= 0.0:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Ear slot ({self.cavity_ear_span_y_mm:.2f} mm) is no longer "
                f"than the body pocket ({self.cavity_body_span_y_mm:.2f} mm), "
                "so there is no shelf for the servo's ears to rest on.",
            )
        shelf_height = self.cavity_top_z_mm - self.ear_shelf_z_mm
        if self.servo_screw_hole_depth_mm >= shelf_height:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"M3 pilot holes are {self.servo_screw_hole_depth_mm:.2f} mm "
                f"deep but the shelf is only {shelf_height:.2f} mm tall; they "
                "would break through into the shaft bore region.",
            )
        half_body_y = self.cavity_body_span_y_mm / 2.0
        half_ear_y = self.cavity_ear_span_y_mm / 2.0
        for index, (_, screw_y) in enumerate(self.servo_screw_positions):
            offset = abs(screw_y - self.cavity_offset_y_mm)
            if not half_body_y < offset < half_ear_y:
                raise PedestalDesignError(
                    DesignStatus.FEATURE_COLLISION,
                    f"Servo screw {index} at y = {screw_y:.2f} mm does not land "
                    "in the retention shelf.",
                )

        # ---- Cable slot ----------------------------------------------------
        if self.cable_slot_height_mm >= self.ear_shelf_z_mm - self.top_arm_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Cable slot ({self.cable_slot_height_mm:.2f} mm tall) does not "
                f"fit between the top arm ({self.top_arm_thickness_mm:.2f} mm) "
                f"and the ear shelf ({self.ear_shelf_z_mm:.2f} mm).",
            )
        if self.cable_slot_width_mm >= self.cavity_ear_span_y_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"Cable slot width ({self.cable_slot_width_mm:.2f} mm) must be "
                f"less than the cavity ({self.cavity_ear_span_y_mm:.2f} mm).",
            )

        return DesignStatus.OK

    # =====================================================================
    # Reporting
    # =====================================================================

    def report(self, hardware: Optional[HardwareSpec] = None) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        clamp = hardware.desk_clamp
        seat_bottom_z = self.turret_top_z_mm - self.bearing_seat_depth_mm
        max_torque = clamp.max_tightening_torque_nm(
            self.clamp_width_mm, self.bottom_arm_overhang_mm
        )
        hand_torque = clamp.hand_torque_limit_nm()
        tipping = DEFAULT_ARM.tipping_moment_nm()
        needed_preload = tipping / (self.tipping_lever_arm_mm / 1000.0)
        return (
            f"Base pedestal (monolithic U-clamp)\n"
            f"----------------------------------\n"
            f"  Overall envelope       : "
            f"{self.top_arm_inner_x_mm - self.spine_outer_x_mm:.2f} (X) x "
            f"{self.clamp_width_mm:.2f} (Y) x {self.overall_height_mm:.2f} (Z) mm\n"
            f"  Desk seats at x        : {self.desk_seat_x_mm:.2f} mm "
            f"(yaw axis at x = 0, against the gusset)\n"
            f"  Top arm                : x {self.spine_inner_x_mm:.1f}.."
            f"{self.top_arm_inner_x_mm:.1f}, z 0..{self.top_arm_thickness_mm:.1f}\n"
            f"  Spine                  : x {self.spine_outer_x_mm:.1f}.."
            f"{self.spine_inner_x_mm:.1f}, z {self.bottom_arm_bottom_z_mm:.1f}.."
            f"{self.top_arm_thickness_mm:.1f}\n"
            f"  Bottom arm             : x {self.spine_inner_x_mm:.1f}.."
            f"{self.bottom_arm_inner_x_mm:.1f}, z "
            f"{self.bottom_arm_bottom_z_mm:.1f}..{self.throat_bottom_z_mm:.1f}\n"
            f"  Throat                 : {self.throat_opening_mm:.1f} mm for "
            f"{clamp.min_desk_thickness_mm:.0f}-"
            f"{clamp.max_desk_thickness_mm:.0f} mm desks\n"
            f"  Gussets                : {self.gusset_size_mm:.1f} mm at both "
            f"inner corners\n"
            f"\n"
            f"  Servo turret           : x {self.turret_x_min_mm:.2f}.."
            f"{self.turret_x_max_mm:.2f}, y {self.turret_y_min_mm:.2f}.."
            f"{self.turret_y_max_mm:.2f}, z {self.top_arm_thickness_mm:.1f}.."
            f"{self.turret_top_z_mm:.1f}\n"
            f"  Servo cavity           : {self.cavity_x_span_mm:.2f} (X) x "
            f"{self.cavity_body_span_y_mm:.2f} (Y) body, "
            f"{self.cavity_ear_span_y_mm:.2f} (Y) ears\n"
            f"  Cavity Y offset        : {self.cavity_offset_y_mm:.2f} mm "
            f"(puts the output shaft on the yaw axis)\n"
            f"  Ear shelf at z         : {self.ear_shelf_z_mm:.2f} mm\n"
            f"  Servo shaft output     : z = {self.servo_shaft_output_z_mm:.2f} mm\n"
            f"  Bearing seat           : {self.bearing_seat_diameter_mm:.2f} mm "
            f"dia x {self.bearing_seat_depth_mm:.2f} mm deep (floor at z = "
            f"{seat_bottom_z:.2f})\n"
            f"  Bearing top (turntable): z = {self.bearing_top_z_mm:.2f} mm\n"
            f"  Shoulder pivot         : z = {DEFAULT_ARM.base_height_mm:.2f} mm "
            f"({hardware.above_pedestal_allowance_mm:.1f} mm of stack above the "
            f"turret)\n"
            f"\n"
            f"  Anti-slip pads         : 2 strips, "
            + ", ".join(
                f"{x_max - x_min:.2f} x {y_max - y_min:.2f}"
                for x_min, x_max, y_min, y_max in self.pad_recesses
            )
            + f" mm ({self.pad_area_mm2:.0f} mm2 total)\n"
            f"  Clamp screw            : {self.bolt_hole_diameter_mm:.2f} mm dia "
            f"at x = {self.bolt_axis_x_mm:.2f} mm\n"
            f"  Nut pocket             : {self.nut_pocket_across_flats_mm:.2f} mm "
            f"across flats x {self.nut_pocket_depth_mm:.2f} mm deep (underside)\n"
            f"\n"
            f"  Clamp mechanics\n"
            f"    bottom arm overhang  : {self.bottom_arm_overhang_mm:.2f} mm\n"
            f"    tipping lever arm    : {self.tipping_lever_arm_mm:.2f} mm\n"
            f"    arm tipping moment   : {tipping:.2f} N.m (worst case)\n"
            f"    preload needed       : {needed_preload:.0f} N "
            f"({clamp.preload_to_torque_nm(needed_preload):.2f} N.m at the knob)\n"
            f"    hand torque available: {hand_torque:.2f} N.m on a "
            f"{clamp.knob_diameter_mm:.0f} mm knob -> "
            f"{clamp.bolt_preload_n(hand_torque):.0f} N\n"
            f"    grip margin          : "
            f"{clamp.bolt_preload_n(hand_torque) / needed_preload:.2f}x\n"
            f"    bottom arm limit     : {max_torque:.2f} N.m "
            f"({'hand cannot reach it' if max_torque > hand_torque else 'REACHABLE BY HAND'})\n"
        )


def build_pedestal(params: Optional[PedestalParameters] = None) -> Part:
    """
    Construct the U-clamp solid.

    Built in build123d's algebra mode: union the three limbs of the U plus the
    turret and gussets, then subtract each internal feature. Every operation is
    positioned from ``params``, so the model has no literals of its own.

    Parameters
    ----------
    params:
        Resolved dimensions. Defaults to :meth:`PedestalParameters.from_geometry`.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the yaw axis at the desk's top surface.

    Raises
    ------
    PedestalDesignError
        If ``params`` fails :meth:`PedestalParameters.validate`.
    """
    params = PedestalParameters.from_geometry() if params is None else params
    params.validate()

    width = params.clamp_width_mm
    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    def slab(x_min, x_max, z_min, z_max, y_span=None):
        """A box spanning the given X and Z range, centred on Y."""
        y_span = width if y_span is None else y_span
        return Pos((x_min + x_max) / 2.0, 0, z_min) * Box(
            x_max - x_min, y_span, z_max - z_min, align=bottom
        )

    # ---- The three limbs of the U, as one solid ---------------------------
    part = slab(
        params.spine_inner_x_mm, params.top_arm_inner_x_mm,
        0.0, params.top_arm_thickness_mm,
    )
    part += slab(
        params.spine_outer_x_mm, params.spine_inner_x_mm,
        params.bottom_arm_bottom_z_mm, params.top_arm_thickness_mm,
    )
    part += slab(
        params.spine_inner_x_mm, params.bottom_arm_inner_x_mm,
        params.bottom_arm_bottom_z_mm, params.throat_bottom_z_mm,
    )

    # ---- Servo turret rising from the top arm -----------------------------
    part += Pos(
        (params.turret_x_min_mm + params.turret_x_max_mm) / 2.0,
        (params.turret_y_min_mm + params.turret_y_max_mm) / 2.0,
        params.top_arm_thickness_mm,
    ) * Box(
        params.turret_x_max_mm - params.turret_x_min_mm,
        params.turret_y_max_mm - params.turret_y_min_mm,
        params.turret_top_z_mm - params.top_arm_thickness_mm,
        align=bottom,
    )

    # ---- Gussets at the two inner corners of the U ------------------------
    # Upper: hangs down from the top arm's underside beside the spine.
    part += (
        Pos(params.spine_inner_x_mm, 0, 0)
        * Rot(180, 0, 0)
        * right_triangle_prism(params.gusset_size_mm, params.gusset_size_mm, width)
    )
    # Lower: rises from the bottom arm's top face beside the spine.
    part += Pos(
        params.spine_inner_x_mm, 0, params.throat_bottom_z_mm
    ) * right_triangle_prism(
        params.gusset_size_mm, params.gusset_size_mm, width
    )

    # ---- Servo cavity, lower section: wide enough to pass the ears --------
    # Open at the top arm's underside so the servo is inserted from below and
    # pushed up until its ears meet the shelf where this section ends.
    part -= Pos(0, params.cavity_offset_y_mm, 0) * Box(
        params.cavity_x_span_mm,
        params.cavity_ear_span_y_mm,
        params.ear_shelf_z_mm,
        align=bottom,
    )

    # ---- Servo cavity, upper section: body only. The step between the two
    #      spans IS the retention shelf. ---------------------------------
    part -= Pos(
        0, params.cavity_offset_y_mm, params.ear_shelf_z_mm
    ) * Box(
        params.cavity_x_span_mm,
        params.cavity_body_span_y_mm,
        params.cavity_top_z_mm - params.ear_shelf_z_mm,
        align=bottom,
    )

    # ---- Output shaft bore through the ceiling, on the yaw axis -----------
    part -= Pos(0, 0, params.cavity_top_z_mm) * Cylinder(
        radius=params.shaft_bore_diameter_mm / 2.0,
        height=params.turret_top_z_mm - params.cavity_top_z_mm,
        align=bottom,
    )

    # ---- 608ZZ press-fit seat in the turret's top face --------------------
    part -= Pos(
        0, 0, params.turret_top_z_mm - params.bearing_seat_depth_mm
    ) * Cylinder(
        radius=params.bearing_seat_diameter_mm / 2.0,
        height=params.bearing_seat_depth_mm,
        align=bottom,
    )

    # ---- Blind M3 pilot holes in the shelf, for screws driven up through
    #      the servo's mounting ears. ------------------------------------
    for screw_x, screw_y in params.servo_screw_positions:
        part -= Pos(screw_x, screw_y, params.ear_shelf_z_mm) * Cylinder(
            radius=params.servo_screw_hole_diameter_mm / 2.0,
            height=params.servo_screw_hole_depth_mm,
            align=bottom,
        )

    # ---- Anti-slip pad recesses in the top arm's underside ----------------
    for x_min, x_max, y_min, y_max in params.pad_recesses:
        part -= Pos(
            (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0.0
        ) * Box(
            x_max - x_min,
            y_max - y_min,
            params.pad_recess_depth_mm,
            align=bottom,
        )

    # ---- Clamp screw through-hole and captive nut pocket ------------------
    part -= Pos(
        params.bolt_axis_x_mm, 0, params.bottom_arm_bottom_z_mm
    ) * Cylinder(
        radius=params.bolt_hole_diameter_mm / 2.0,
        height=params.bottom_arm_thickness_mm,
        align=bottom,
    )
    part -= Pos(
        params.bolt_axis_x_mm, 0, params.bottom_arm_bottom_z_mm
    ) * hex_prism(params.nut_pocket_across_flats_mm, params.nut_pocket_depth_mm)

    # ---- Cable slot out through the turret's inboard wall -----------------
    slot_z = params.top_arm_thickness_mm + (
        params.ear_shelf_z_mm
        - params.top_arm_thickness_mm
        - params.cable_slot_height_mm
    ) / 2.0
    part -= Pos(params.turret_x_max_mm / 2.0, 0, slot_z) * Box(
        params.turret_x_max_mm,
        params.cable_slot_width_mm,
        params.cable_slot_height_mm,
        align=bottom,
    )

    return part


def export_pedestal(
    output_path: Optional[Path] = None,
    params: Optional[PedestalParameters] = None,
) -> Path:
    """
    Build the pedestal and write it to an STL, creating parent directories.

    Raises
    ------
    PedestalDesignError
        If the parameters are unbuildable.
    RuntimeError
        If build123d reports the export failed.
    """
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_pedestal(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the base pedestal STL from src/geometry.py."
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_STL_PATH,
        help=f"STL destination (default: {DEFAULT_STL_PATH})",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = PedestalParameters.from_geometry()
    except PedestalDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    print(
        "  !! "
        + DEFAULT_HARDWARE.base_yaw_servo.unverified_report().replace("\n", "\n  ")
    )
    print()

    if args.report:
        return 0

    written = export_pedestal(args.output, params)
    print(f"Wrote {written}  ({written.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
