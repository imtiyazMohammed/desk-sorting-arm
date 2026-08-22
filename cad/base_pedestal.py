"""
Base pedestal -- clamps to the desk edge and carries the base yaw joint.

Run directly to write ``cad/output/base_pedestal.stl``::

    python3 -m cad.base_pedestal
    python3 -m cad.base_pedestal --output /tmp/pedestal.stl
    python3 -m cad.base_pedestal --report      # dimensions only, no export

What the part does
------------------
The pedestal proper, bottom to top along +Z:

1. **Servo cavity** -- a stepped rectangular pocket, open at the bottom. The
   servo is inserted from below and pushed up until its mounting ears meet
   the internal shelf where the pocket narrows from the ear span to the body
   width. Four M3 pilot holes in that shelf take screws driven upward through
   the ears. The cavity is offset laterally by
   ``ServoSpec.body_offset_from_shaft_axis_mm`` so the servo's *output shaft*,
   not its body centre, lands on the yaw axis.
2. **Shaft bore** -- clears the servo's output boss through the ceiling above
   the cavity.
3. **Bearing seat** -- a press-fit pocket in the top face for a 608ZZ. The
   seat is cut ``bearing_proud_mm`` shallower than the bearing is wide, so the
   bearing stands slightly proud and the yaw turntable above rides on the
   bearing's inner race alone rather than scrubbing on the printed top face.
   This is what takes axial load off the servo's output shaft.

Plus the **upper jaw** of the desk clamp (Session D.1b), a plate extending in
+X from the pedestal at desk level:

4. **Pad recess** on the jaw's underside, for a glued-in anti-slip rubber pad.
5. **M8 through-hole** near the outboard end, for the clamping screw. The
   desk edge is meant to fall between the pad's outer edge and that hole --
   see :attr:`PedestalParameters.desk_edge_window_mm`.

A radial cable slot on the -X side (opposite the clamp) lets the servo lead
out of the otherwise-enclosed cavity.

Why a clamp instead of bolts
----------------------------
The Session D.1 design bolted a flange to the desk through four M4 holes.
That required drilling the desk, which the user does not want, and fixed the
arm in one place. A desk-edge clamp needs no holes and can be repositioned or
removed. The pedestal internals -- cavity, shelf, bore, bearing seat -- are
unchanged; only the mounting changed.

Sizing philosophy
-----------------
Nothing below is a magic number. The body radius is *derived*: it is whatever
is needed to keep ``min_wall_thickness_mm`` of material outside the furthest
internal feature. The jaw's reach is derived from where the servo cavity ends,
the pad size, and the clamp screw's clearance. Feed a different servo into
``src/geometry.py`` and the pedestal resizes itself.
:meth:`PedestalParameters.validate` then re-checks every clearance and refuses
to build a part that would print with a wall thinner than specified, a hole
breaking into a pocket, or a clamp screw too short to reach its nut.

Units are millimetres throughout, matching ``src.geometry`` and STL's
convention.

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

from build123d import Align, Box, Cylinder, Part, Pos, export_stl

from cad._design import DesignRuleError, DesignStatus

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
    Fully resolved pedestal dimensions, in millimetres.

    Build these with :meth:`from_geometry` rather than by hand; the direct
    constructor exists so tests can inject deliberately-broken values and
    confirm :meth:`validate` catches them.
    """

    # ---- Overall envelope -------------------------------------------------
    total_height_mm: float
    body_radius_mm: float

    # ---- Servo cavity -----------------------------------------------------
    cavity_body_length_mm: float
    cavity_ear_length_mm: float
    cavity_width_mm: float
    cavity_offset_x_mm: float
    cavity_top_z_mm: float
    ear_shelf_z_mm: float

    # ---- Shaft bore and bearing seat --------------------------------------
    shaft_bore_diameter_mm: float
    bearing_seat_diameter_mm: float
    bearing_seat_depth_mm: float

    # ---- Desk clamp: upper jaw --------------------------------------------
    jaw_thickness_mm: float
    jaw_width_mm: float
    jaw_reach_mm: float
    pad_inner_x_mm: float
    pad_length_mm: float
    pad_width_mm: float
    pad_recess_depth_mm: float
    bolt_axis_x_mm: float
    bolt_hole_diameter_mm: float
    desk_edge_window_mm: float
    knob_diameter_mm: float

    # ---- Servo retention --------------------------------------------------
    servo_screw_hole_diameter_mm: float
    servo_screw_hole_depth_mm: float
    servo_screw_spacing_x_mm: float
    servo_screw_spacing_y_mm: float

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
        bearing_proud_mm: float = 0.5,
        ear_top_offset_from_body_top_mm: float = 10.0,
        desk_edge_window_mm: float = 10.0,
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
        bearing_proud_mm:
            How far the bearing stands above the top face. Must be positive,
            or the turntable rubs the printed face instead of turning on the
            bearing's inner race.
        ear_top_offset_from_body_top_mm:
            Distance from the top of the servo body down to the top of its
            mounting ears. UNVERIFIED placeholder -- see the module warning.
        desk_edge_window_mm:
            Gap between the pad's outer edge and the clamp screw's hole. The
            desk edge must fall inside this band, so it is the positioning
            tolerance the user gets when siting the arm. Wider is friendlier
            to use but lengthens the jaw's overhang, which is what limits how
            hard the knob may be tightened.
        cable_slot_width_mm, cable_slot_height_mm:
            Radial slot letting the servo lead out of the cavity, on -X.
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
            ("bearing_proud_mm", bearing_proud_mm),
            ("ear_top_offset_from_body_top_mm", ear_top_offset_from_body_top_mm),
            ("desk_edge_window_mm", desk_edge_window_mm),
            ("cable_slot_width_mm", cable_slot_width_mm),
            ("cable_slot_height_mm", cable_slot_height_mm),
            ("servo_screw_hole_depth_mm", servo_screw_hole_depth_mm),
        ):
            if value <= 0.0:
                raise PedestalDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )

        if bearing_proud_mm >= bearing.width_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"bearing_proud_mm ({bearing_proud_mm}) must be less than the "
                f"bearing width ({bearing.width_mm}); otherwise there is no "
                "seat left to press the outer race into.",
            )

        # ---- Vertical layout, resolved from the top face downward ---------
        try:
            total_height = hardware.pedestal_height_mm(arm)
        except ValueError as exc:
            raise PedestalDesignError(DesignStatus.NEGATIVE_HEIGHT, str(exc)) from exc

        seat_depth = bearing.width_mm - bearing_proud_mm
        seat_bottom_z = total_height - seat_depth

        # The servo's output boss must reach up to the underside of the
        # bearing, so the body top sits one boss-height below the seat floor.
        cavity_top_z = seat_bottom_z - servo.shaft_boss_height_mm
        ear_shelf_z = cavity_top_z - ear_top_offset_from_body_top_mm

        # ---- Cavity cross-sections ----------------------------------------
        cavity_body_length = servo.body_length_mm + 2.0 * clearance
        cavity_ear_length = servo.flange_span_mm + 2.0 * clearance
        cavity_width = servo.body_width_mm + 2.0 * clearance
        # Negated: body_offset_from_shaft_axis_mm is measured from the shaft
        # toward the body's far end, so the body sits on the opposite side of
        # the axis for the shaft to be centred.
        cavity_offset_x = -servo.body_offset_from_shaft_axis_mm

        # ---- Radial sizing: derive, do not assume -------------------------
        # Furthest internal feature from the Z axis, plus one wall.
        ear_corner_radius = math.hypot(
            abs(cavity_offset_x) + cavity_ear_length / 2.0, cavity_width / 2.0
        )
        body_radius = ear_corner_radius + wall

        # ---- Clamp jaw, laid out along +X ---------------------------------
        # The pad starts one wall outboard of where the servo cavity ends, so
        # the recess never breaks into the open pocket beneath the servo.
        cavity_outer_x = cavity_offset_x + cavity_ear_length / 2.0
        pad_length, pad_width = clamp.upper_jaw_contact_mm
        pad_inner_x = cavity_outer_x + wall
        pad_outer_x = pad_inner_x + pad_length

        bolt_hole_diameter = clamp.bolt_clearance_hole_diameter_mm
        bolt_axis_x = pad_outer_x + desk_edge_window_mm + bolt_hole_diameter / 2.0
        jaw_reach = bolt_axis_x + bolt_hole_diameter / 2.0 + wall
        # Wide enough for both the pad and the knob that sits above the bolt.
        jaw_width = max(pad_width, clamp.knob_diameter_mm)

        params = cls(
            total_height_mm=total_height,
            body_radius_mm=body_radius,
            cavity_body_length_mm=cavity_body_length,
            cavity_ear_length_mm=cavity_ear_length,
            cavity_width_mm=cavity_width,
            cavity_offset_x_mm=cavity_offset_x,
            cavity_top_z_mm=cavity_top_z,
            ear_shelf_z_mm=ear_shelf_z,
            shaft_bore_diameter_mm=servo.shaft_boss_diameter_mm + 2.0 * clearance,
            bearing_seat_diameter_mm=bearing.seat_diameter_mm,
            bearing_seat_depth_mm=seat_depth,
            jaw_thickness_mm=clamp.upper_jaw_total_thickness_mm,
            jaw_width_mm=jaw_width,
            jaw_reach_mm=jaw_reach,
            pad_inner_x_mm=pad_inner_x,
            pad_length_mm=pad_length,
            pad_width_mm=pad_width,
            pad_recess_depth_mm=clamp.pad_recess_depth_mm,
            bolt_axis_x_mm=bolt_axis_x,
            bolt_hole_diameter_mm=bolt_hole_diameter,
            desk_edge_window_mm=desk_edge_window_mm,
            knob_diameter_mm=clamp.knob_diameter_mm,
            servo_screw_hole_diameter_mm=servo.flange_hole_diameter_mm,
            servo_screw_hole_depth_mm=servo_screw_hole_depth_mm,
            servo_screw_spacing_x_mm=servo.flange_hole_spacing_long_mm,
            servo_screw_spacing_y_mm=servo.flange_hole_spacing_short_mm,
            cable_slot_width_mm=cable_slot_width_mm,
            cable_slot_height_mm=cable_slot_height_mm,
            min_wall_thickness_mm=wall,
        )
        params.validate()

        # The clamp screw has to span a stack this part only half determines,
        # so the check lives here where both halves are known.
        if clamp.bolt_length_mm < clamp.required_bolt_length_mm:
            raise PedestalDesignError(
                DesignStatus.FASTENER_TOO_SHORT,
                f"{clamp.bolt_thread} x {clamp.bolt_length_mm:.1f} mm cannot "
                f"span the clamp stack, which needs "
                f"{clamp.required_bolt_length_mm:.1f} mm at maximum throat "
                f"({clamp.throat_max_opening_mm:.1f} mm). Fit a longer screw "
                "or reduce throat_max_opening_mm.",
            )
        return params

    # =====================================================================
    # Derived accessors
    # =====================================================================

    @property
    def cavity_bottom_z_mm(self) -> float:
        """The cavity is open to the underside, so it starts at z = 0."""
        return 0.0

    @property
    def pad_outer_x_mm(self) -> float:
        """Outboard edge of the anti-slip pad recess."""
        return self.pad_inner_x_mm + self.pad_length_mm

    @property
    def jaw_overhang_mm(self) -> float:
        """
        Worst-case unsupported jaw length, from the desk edge to the bolt.

        The desk may sit anywhere in the positioning window, so the longest
        overhang -- and therefore the highest bending stress -- happens when
        the edge is at the pad's outer edge. This is what
        ``DeskClampSpec.max_tightening_torque_nm`` is evaluated against.
        """
        return self.bolt_axis_x_mm - self.pad_outer_x_mm

    @property
    def tipping_lever_arm_mm(self) -> float:
        """
        Lever between the tipping pivot and the clamp screw.

        The assembly would rotate about the inboard edge of the pad, the last
        line of contact with the desk, so that is the pivot.
        """
        return self.bolt_axis_x_mm - self.pad_inner_x_mm

    @property
    def servo_screw_positions(self) -> Tuple[Tuple[float, float], ...]:
        """(x, y) centres of the four M3 servo-retention pilot holes."""
        return tuple(
            (
                self.cavity_offset_x_mm + sx * self.servo_screw_spacing_x_mm / 2.0,
                sy * self.servo_screw_spacing_y_mm / 2.0,
            )
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )

    def _rect_distance_to_point(
        self, half_length: float, half_width: float, px: float, py: float
    ) -> float:
        """
        Distance from ``(px, py)`` to the nearest point of the cavity rectangle.

        Zero when the point lies inside.
        """
        dx = max(abs(px - self.cavity_offset_x_mm) - half_length, 0.0)
        dy = max(abs(py) - half_width, 0.0)
        return math.hypot(dx, dy)

    # =====================================================================
    # Validation
    # =====================================================================

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unprintable part.

        Returns :attr:`DesignStatus.OK` on success. This is a design rule
        check, not a modelling detail: it is what makes the module safe to
        drive from swept parameters, because a violating parameter set fails
        loudly here instead of silently producing a part with a 0.2 mm wall
        or a bolt hole opening into the servo pocket.

        Raises
        ------
        PedestalDesignError
            Naming the first violated constraint.
        """
        if self.total_height_mm <= 0.0:
            raise PedestalDesignError(
                DesignStatus.NEGATIVE_HEIGHT,
                f"total_height_mm must be positive, got {self.total_height_mm}.",
            )
        if self.jaw_thickness_mm >= self.total_height_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"jaw_thickness_mm ({self.jaw_thickness_mm}) must be less "
                f"than total_height_mm ({self.total_height_mm}).",
            )
        if self.pad_recess_depth_mm >= self.jaw_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"pad_recess_depth_mm ({self.pad_recess_depth_mm}) must be "
                f"less than the jaw thickness ({self.jaw_thickness_mm}), or "
                "the recess cuts straight through the jaw.",
            )

        # ---- Vertical ordering: jaw < ear shelf < cavity top < seat -------
        seat_bottom_z = self.total_height_mm - self.bearing_seat_depth_mm
        if not (
            self.jaw_thickness_mm
            < self.ear_shelf_z_mm
            < self.cavity_top_z_mm
            < seat_bottom_z
        ):
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                "Vertical layout is out of order: expected jaw top "
                f"({self.jaw_thickness_mm:.2f}) < ear shelf "
                f"({self.ear_shelf_z_mm:.2f}) < cavity top "
                f"({self.cavity_top_z_mm:.2f}) < bearing seat floor "
                f"({seat_bottom_z:.2f}). The servo does not fit in the "
                "available height.",
            )

        # ---- Ceiling between the cavity and the bearing seat --------------
        ceiling = seat_bottom_z - self.cavity_top_z_mm
        if ceiling < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {ceiling:.2f} mm of material between the servo cavity "
                f"ceiling and the bearing seat floor; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )

        # ---- Radial wall around the widest internal pocket ----------------
        ear_corner_radius = math.hypot(
            abs(self.cavity_offset_x_mm) + self.cavity_ear_length_mm / 2.0,
            self.cavity_width_mm / 2.0,
        )
        radial_wall = self.body_radius_mm - ear_corner_radius
        if radial_wall < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {radial_wall:.2f} mm of wall between the ear slot corner "
                f"(r = {ear_corner_radius:.2f} mm) and the body outer surface "
                f"(r = {self.body_radius_mm:.2f} mm); "
                f"{self.min_wall_thickness_mm:.2f} mm required. The servo is "
                "too large for this body diameter.",
            )

        # ---- Bearing seat must not undercut the body wall -----------------
        seat_wall = self.body_radius_mm - self.bearing_seat_diameter_mm / 2.0
        if seat_wall < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {seat_wall:.2f} mm between the bearing seat and the body "
                f"outer surface; {self.min_wall_thickness_mm:.2f} mm required.",
            )
        if self.shaft_bore_diameter_mm >= self.bearing_seat_diameter_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Shaft bore ({self.shaft_bore_diameter_mm:.2f} mm) is not "
                f"smaller than the bearing seat "
                f"({self.bearing_seat_diameter_mm:.2f} mm), so the seat would "
                "have no floor for the outer race to sit on.",
            )

        # ---- Clamp jaw ----------------------------------------------------
        cavity_outer_x = self.cavity_offset_x_mm + self.cavity_ear_length_mm / 2.0
        pad_gap = self.pad_inner_x_mm - cavity_outer_x
        if pad_gap < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Pad recess starts {pad_gap:.2f} mm outboard of the servo "
                f"cavity, which ends at x = {cavity_outer_x:.2f} mm; "
                f"{self.min_wall_thickness_mm:.2f} mm required or the recess "
                "breaks into the open pocket under the servo.",
            )
        if self.desk_edge_window_mm <= 0.0:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                "The clamp screw hole overlaps the pad recess; there is no "
                "band left for the desk edge to sit in.",
            )
        jaw_tip_wall = (
            self.jaw_reach_mm
            - self.bolt_axis_x_mm
            - self.bolt_hole_diameter_mm / 2.0
        )
        if jaw_tip_wall < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {jaw_tip_wall:.2f} mm of jaw beyond the clamp screw "
                f"hole; {self.min_wall_thickness_mm:.2f} mm required.",
            )
        jaw_side_wall = (
            self.jaw_width_mm / 2.0 - self.bolt_hole_diameter_mm / 2.0
        )
        if jaw_side_wall < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {jaw_side_wall:.2f} mm between the clamp screw hole and "
                f"the jaw's side; {self.min_wall_thickness_mm:.2f} mm required.",
            )
        if self.pad_width_mm > self.jaw_width_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"Pad width ({self.pad_width_mm:.2f} mm) exceeds the jaw width "
                f"({self.jaw_width_mm:.2f} mm).",
            )
        knob_inner_x = self.bolt_axis_x_mm - self.knob_diameter_mm / 2.0
        if knob_inner_x < self.body_radius_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The knob (dia {self.knob_diameter_mm:.1f} mm) would reach "
                f"x = {knob_inner_x:.2f} mm and foul the pedestal body "
                f"(r = {self.body_radius_mm:.2f} mm). Lengthen the jaw or fit "
                "a smaller knob.",
            )

        # ---- Servo retention screws must land in shelf material -----------
        shelf_step = (self.cavity_ear_length_mm - self.cavity_body_length_mm) / 2.0
        if shelf_step <= 0.0:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Ear slot ({self.cavity_ear_length_mm:.2f} mm) is no longer "
                f"than the body pocket ({self.cavity_body_length_mm:.2f} mm), "
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
        for index, (sx, sy) in enumerate(self.servo_screw_positions):
            screw_outer = math.hypot(sx, sy) + self.servo_screw_hole_diameter_mm / 2.0
            if self.body_radius_mm - screw_outer < self.min_wall_thickness_mm:
                raise PedestalDesignError(
                    DesignStatus.WALL_TOO_THIN,
                    f"Servo screw {index} at ({sx:.2f}, {sy:.2f}) reaches "
                    f"r = {screw_outer:.2f} mm, leaving "
                    f"{self.body_radius_mm - screw_outer:.2f} mm of wall "
                    f"against a body radius of {self.body_radius_mm:.2f} mm.",
                )

        # ---- Cable slot ---------------------------------------------------
        if self.cable_slot_height_mm >= self.ear_shelf_z_mm - self.jaw_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Cable slot ({self.cable_slot_height_mm:.2f} mm tall) does not "
                f"fit between the jaw top ({self.jaw_thickness_mm:.2f} mm) "
                f"and the ear shelf ({self.ear_shelf_z_mm:.2f} mm).",
            )
        if self.cable_slot_width_mm >= self.cavity_width_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"Cable slot width ({self.cable_slot_width_mm:.2f} mm) must be "
                f"less than the cavity width ({self.cavity_width_mm:.2f} mm).",
            )

        return DesignStatus.OK

    # =====================================================================
    # Reporting
    # =====================================================================

    def report(self, hardware: Optional[HardwareSpec] = None) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        clamp = hardware.desk_clamp
        seat_bottom_z = self.total_height_mm - self.bearing_seat_depth_mm
        ear_corner_radius = math.hypot(
            abs(self.cavity_offset_x_mm) + self.cavity_ear_length_mm / 2.0,
            self.cavity_width_mm / 2.0,
        )
        max_torque = clamp.max_tightening_torque_nm(self.jaw_overhang_mm)
        return (
            f"Base pedestal parameters\n"
            f"------------------------\n"
            f"  Total height           : {self.total_height_mm:.2f} mm\n"
            f"  Body outer diameter    : {2 * self.body_radius_mm:.2f} mm\n"
            f"\n"
            f"  Servo cavity (body)    : {self.cavity_body_length_mm:.2f} x "
            f"{self.cavity_width_mm:.2f} mm\n"
            f"  Servo cavity (ears)    : {self.cavity_ear_length_mm:.2f} x "
            f"{self.cavity_width_mm:.2f} mm\n"
            f"  Cavity X offset        : {self.cavity_offset_x_mm:.2f} mm "
            f"(puts the output shaft on the yaw axis)\n"
            f"  Ear shelf at z         : {self.ear_shelf_z_mm:.2f} mm\n"
            f"  Cavity ceiling at z    : {self.cavity_top_z_mm:.2f} mm\n"
            f"\n"
            f"  Shaft bore             : {self.shaft_bore_diameter_mm:.2f} mm dia\n"
            f"  Bearing seat           : {self.bearing_seat_diameter_mm:.2f} mm dia "
            f"x {self.bearing_seat_depth_mm:.2f} mm deep (floor at z = "
            f"{seat_bottom_z:.2f})\n"
            f"\n"
            f"  Clamp upper jaw        : {self.jaw_reach_mm:.2f} mm reach x "
            f"{self.jaw_width_mm:.2f} mm wide x {self.jaw_thickness_mm:.2f} mm thick\n"
            f"  Pad recess             : {self.pad_length_mm:.1f} x "
            f"{self.pad_width_mm:.1f} x {self.pad_recess_depth_mm:.1f} mm deep, "
            f"x = {self.pad_inner_x_mm:.2f}..{self.pad_outer_x_mm:.2f}\n"
            f"  Clamp screw hole       : {self.bolt_hole_diameter_mm:.2f} mm dia "
            f"at x = {self.bolt_axis_x_mm:.2f} mm\n"
            f"  Desk edge window       : x = {self.pad_outer_x_mm:.2f}.."
            f"{self.bolt_axis_x_mm - self.bolt_hole_diameter_mm / 2.0:.2f} mm "
            f"({self.desk_edge_window_mm:.1f} mm wide)\n"
            f"\n"
            f"  Servo screws           : 4 x "
            f"{self.servo_screw_hole_diameter_mm:.2f} mm, "
            f"{self.servo_screw_hole_depth_mm:.2f} mm deep\n"
            f"  Thinnest radial wall   : "
            f"{self.body_radius_mm - ear_corner_radius:.2f} mm "
            f"(minimum {self.min_wall_thickness_mm:.2f} mm)\n"
            f"  Ceiling thickness      : "
            f"{seat_bottom_z - self.cavity_top_z_mm:.2f} mm\n"
            f"\n"
            f"  Clamp mechanics\n"
            f"    jaw overhang (worst) : {self.jaw_overhang_mm:.2f} mm\n"
            f"    tipping lever arm    : {self.tipping_lever_arm_mm:.2f} mm\n"
            f"    arm tipping moment   : {DEFAULT_ARM.tipping_moment_nm():.2f} N.m\n"
            f"    preload needed       : "
            f"{DEFAULT_ARM.tipping_moment_nm() / (self.tipping_lever_arm_mm / 1000.0):.0f} N "
            f"({clamp.preload_to_torque_nm(DEFAULT_ARM.tipping_moment_nm() / (self.tipping_lever_arm_mm / 1000.0)):.2f} N.m at the knob)\n"
            f"    MAX safe knob torque : {max_torque:.2f} N.m  "
            f"<-- hand-tight only, do not use a wrench\n"
        )


def build_pedestal(params: Optional[PedestalParameters] = None) -> Part:
    """
    Construct the pedestal solid, including the clamp's upper jaw.

    Built in build123d's algebra mode: start with the body and jaw as a union
    of solid stock, then subtract each internal feature. Every subtraction is
    positioned from ``params``, so the model has no literals of its own.

    Parameters
    ----------
    params:
        Resolved dimensions. Defaults to :meth:`PedestalParameters.from_geometry`.

    Returns
    -------
    build123d.Part
        A single solid, origin on the yaw axis at desk level, +Z up.

    Raises
    ------
    PedestalDesignError
        If ``params`` fails :meth:`PedestalParameters.validate`.
    """
    params = PedestalParameters.from_geometry() if params is None else params
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Solid stock: pedestal column plus the clamp's upper jaw ---------
    part = Cylinder(
        radius=params.body_radius_mm,
        height=params.total_height_mm,
        align=bottom,
    )
    part += Pos(params.jaw_reach_mm / 2.0, 0, 0) * Box(
        params.jaw_reach_mm,
        params.jaw_width_mm,
        params.jaw_thickness_mm,
        align=bottom,
    )

    # ---- Servo cavity, lower section: wide enough to pass the ears -------
    # Open at the bottom so the servo is inserted from underneath and pushed
    # up until its ears meet the shelf where this section ends. Subtracted
    # after the jaw union so the jaw does not seal the insertion path.
    part -= Pos(params.cavity_offset_x_mm, 0, 0) * Box(
        params.cavity_ear_length_mm,
        params.cavity_width_mm,
        params.ear_shelf_z_mm,
        align=bottom,
    )

    # ---- Servo cavity, upper section: body only. The step between the two
    #      widths IS the retention shelf. -------------------------------
    part -= Pos(params.cavity_offset_x_mm, 0, params.ear_shelf_z_mm) * Box(
        params.cavity_body_length_mm,
        params.cavity_width_mm,
        params.cavity_top_z_mm - params.ear_shelf_z_mm,
        align=bottom,
    )

    # ---- Output shaft bore through the ceiling, on the yaw axis ----------
    part -= Pos(0, 0, params.cavity_top_z_mm) * Cylinder(
        radius=params.shaft_bore_diameter_mm / 2.0,
        height=params.total_height_mm - params.cavity_top_z_mm,
        align=bottom,
    )

    # ---- 608ZZ press-fit seat in the top face ----------------------------
    part -= Pos(
        0, 0, params.total_height_mm - params.bearing_seat_depth_mm
    ) * Cylinder(
        radius=params.bearing_seat_diameter_mm / 2.0,
        height=params.bearing_seat_depth_mm,
        align=bottom,
    )

    # ---- Anti-slip pad recess in the jaw's underside ---------------------
    part -= Pos(
        params.pad_inner_x_mm + params.pad_length_mm / 2.0, 0, 0
    ) * Box(
        params.pad_length_mm,
        params.pad_width_mm,
        params.pad_recess_depth_mm,
        align=bottom,
    )

    # ---- Clamp screw through-hole ----------------------------------------
    part -= Pos(params.bolt_axis_x_mm, 0, 0) * Cylinder(
        radius=params.bolt_hole_diameter_mm / 2.0,
        height=params.jaw_thickness_mm,
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

    # ---- Radial cable slot, on -X so the lead exits away from the clamp --
    slot_z = params.jaw_thickness_mm + (
        params.ear_shelf_z_mm - params.jaw_thickness_mm - params.cable_slot_height_mm
    ) / 2.0
    part -= Pos(-params.body_radius_mm / 2.0, 0, slot_z) * Box(
        params.body_radius_mm,
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

    Returns
    -------
    Path
        The path actually written.

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
    size_kb = written.stat().st_size / 1024.0
    print(f"Wrote {written}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
