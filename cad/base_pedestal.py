"""
Base pedestal -- the part that bolts to the desk and carries the base yaw joint.

Run directly to write ``cad/output/base_pedestal.stl``::

    python3 -m cad.base_pedestal
    python3 -m cad.base_pedestal --output /tmp/pedestal.stl
    python3 -m cad.base_pedestal --report      # dimensions only, no export

What the part does
------------------
Bottom to top, along +Z:

1. **Bottom flange** -- a disc that spreads load onto the desk and carries
   ``mount_bolt_count`` M4 clearance holes on the
   ``mount_bolt_circle_diameter_mm`` bolt circle.
2. **Bolt access channels** -- the bolt circle (60 mm) is smaller than the
   body the servo cavity forces (~85 mm OD), so those holes sit *underneath*
   the body wall where no driver can reach them. Each bolt therefore gets a
   vertical counterbore running from the top face down to the flange, sized
   for a long hex key. Bolts are placed at 45-degree azimuths, which
   :meth:`PedestalParameters.validate` proves clear of the servo cavity.
   Assembly order is: bolt the pedestal down first, then fit the servo.
3. **Servo cavity** -- a stepped rectangular pocket, open at the bottom. The
   servo is inserted from below and pushed up until its mounting ears meet
   the internal shelf where the pocket narrows from the ear span to the body
   width. Four M3 pilot holes in that shelf take screws driven upward through
   the ears. The cavity is offset laterally by
   ``ServoSpec.body_offset_from_shaft_axis_mm`` so the servo's *output shaft*,
   not its body centre, lands on the yaw axis.
4. **Shaft bore** -- clears the servo's output boss through the ceiling above
   the cavity.
5. **Bearing seat** -- a press-fit pocket in the top face for a 608ZZ. The
   seat is cut ``bearing_proud_mm`` shallower than the bearing is wide, so the
   bearing stands slightly proud and the yaw turntable above rides on the
   bearing's inner race alone rather than scrubbing on the printed top face.
   This is what takes axial load off the servo's output shaft.

A radial cable slot lets the servo lead out of the otherwise-enclosed cavity.

Sizing philosophy
-----------------
Nothing below is a magic number. The body radius is *derived*: it is whatever
is needed to keep ``min_wall_thickness_mm`` of material outside the furthest
internal feature. Feed a different servo into ``src/geometry.py`` and the
pedestal resizes itself. :meth:`PedestalParameters.validate` then re-checks
every clearance and refuses to build a part that would print with a wall
thinner than specified or a hole breaking into a pocket.

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
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

from build123d import Align, Box, Cylinder, Part, Pos, export_stl

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


class DesignStatus(Enum):
    """
    Structured reasons a parameter set cannot be built.

    Mirrors the ``IKStatus`` / ``SourceStatus`` pattern used elsewhere in the
    project: a failed design reports *which* constraint it violated, so a
    caller sweeping parameters can tell "servo too big" from "bolt circle
    collides with the cavity".
    """

    OK = auto()
    """All clearances satisfied."""

    NEGATIVE_HEIGHT = auto()
    """The base-height budget leaves no room for a pedestal."""

    WALL_TOO_THIN = auto()
    """A feature sits closer to the outer surface than min_wall_thickness_mm."""

    FEATURE_COLLISION = auto()
    """Two internal features intersect that must not."""

    INVALID_PARAMETER = auto()
    """A directly-supplied parameter is out of range."""


class PedestalDesignError(ValueError):
    """
    Raised when a parameter set would produce an unbuildable or unprintable part.

    Attributes
    ----------
    status:
        The :class:`DesignStatus` naming the violated constraint.
    """

    def __init__(self, status: DesignStatus, message: str) -> None:
        super().__init__(f"[{status.name}] {message}")
        self.status = status


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
    flange_radius_mm: float
    flange_thickness_mm: float

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

    # ---- Desk mounting ----------------------------------------------------
    bolt_circle_radius_mm: float
    bolt_count: int
    bolt_hole_diameter_mm: float
    bolt_channel_diameter_mm: float
    bolt_azimuth_offset_deg: float

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
        flange_thickness_mm: float = 5.0,
        flange_lip_mm: float = 4.0,
        bearing_proud_mm: float = 0.5,
        ear_top_offset_from_body_top_mm: float = 10.0,
        bolt_channel_diameter_mm: float = 8.0,
        bolt_azimuth_offset_deg: float = 45.0,
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
        flange_thickness_mm:
            Height of the bottom mounting disc.
        flange_lip_mm:
            How far the flange overhangs the body radially.
        bearing_proud_mm:
            How far the bearing stands above the top face. Must be positive,
            or the turntable rubs the printed face instead of turning on the
            bearing's inner race.
        ear_top_offset_from_body_top_mm:
            Distance from the top of the servo body down to the top of its
            mounting ears. UNVERIFIED placeholder -- see the module warning.
        bolt_channel_diameter_mm:
            Diameter of the vertical hex-key access bore over each M4 hole.
        bolt_azimuth_offset_deg:
            Rotation of the bolt pattern. The default 45 degrees is what keeps
            the channels clear of the offset servo cavity.
        cable_slot_width_mm, cable_slot_height_mm:
            Radial slot letting the servo lead out of the cavity.
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
        fastener = hardware.mounting_fastener
        clearance = hardware.print_clearance_mm
        wall = hardware.min_wall_thickness_mm

        for name, value in (
            ("flange_thickness_mm", flange_thickness_mm),
            ("flange_lip_mm", flange_lip_mm),
            ("bearing_proud_mm", bearing_proud_mm),
            ("ear_top_offset_from_body_top_mm", ear_top_offset_from_body_top_mm),
            ("bolt_channel_diameter_mm", bolt_channel_diameter_mm),
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
        bolt_circle_radius = hardware.mount_bolt_circle_diameter_mm / 2.0
        channel_outer_radius = bolt_circle_radius + bolt_channel_diameter_mm / 2.0
        body_radius = max(ear_corner_radius, channel_outer_radius) + wall
        flange_radius = body_radius + flange_lip_mm

        # ---- Servo retention screws ---------------------------------------
        # Ear holes are on the servo body's centreline, so their X positions
        # are measured from the (offset) cavity centre, not from the Z axis.
        params = cls(
            total_height_mm=total_height,
            body_radius_mm=body_radius,
            flange_radius_mm=flange_radius,
            flange_thickness_mm=flange_thickness_mm,
            cavity_body_length_mm=cavity_body_length,
            cavity_ear_length_mm=cavity_ear_length,
            cavity_width_mm=cavity_width,
            cavity_offset_x_mm=cavity_offset_x,
            cavity_top_z_mm=cavity_top_z,
            ear_shelf_z_mm=ear_shelf_z,
            shaft_bore_diameter_mm=servo.shaft_boss_diameter_mm + 2.0 * clearance,
            bearing_seat_diameter_mm=bearing.seat_diameter_mm,
            bearing_seat_depth_mm=seat_depth,
            bolt_circle_radius_mm=bolt_circle_radius,
            bolt_count=hardware.mount_bolt_count,
            bolt_hole_diameter_mm=fastener.clearance_hole_diameter_mm,
            bolt_channel_diameter_mm=bolt_channel_diameter_mm,
            bolt_azimuth_offset_deg=bolt_azimuth_offset_deg,
            servo_screw_hole_diameter_mm=servo.flange_hole_diameter_mm,
            servo_screw_hole_depth_mm=servo_screw_hole_depth_mm,
            servo_screw_spacing_x_mm=servo.flange_hole_spacing_long_mm,
            servo_screw_spacing_y_mm=servo.flange_hole_spacing_short_mm,
            cable_slot_width_mm=cable_slot_width_mm,
            cable_slot_height_mm=cable_slot_height_mm,
            min_wall_thickness_mm=wall,
        )
        params.validate()
        return params

    # =====================================================================
    # Derived accessors
    # =====================================================================

    @property
    def cavity_bottom_z_mm(self) -> float:
        """The cavity is open to the underside, so it starts at z = 0."""
        return 0.0

    @property
    def bolt_positions(self) -> Tuple[Tuple[float, float], ...]:
        """(x, y) centres of the desk-mounting bolts, evenly spaced."""
        step = 360.0 / self.bolt_count
        return tuple(
            (
                self.bolt_circle_radius_mm
                * math.cos(math.radians(self.bolt_azimuth_offset_deg + i * step)),
                self.bolt_circle_radius_mm
                * math.sin(math.radians(self.bolt_azimuth_offset_deg + i * step)),
            )
            for i in range(self.bolt_count)
        )

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

        Zero when the point lies inside. Used by :meth:`validate` to prove the
        bolt channels do not break into the servo pocket.
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
        if self.flange_thickness_mm >= self.total_height_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"flange_thickness_mm ({self.flange_thickness_mm}) must be less "
                f"than total_height_mm ({self.total_height_mm}).",
            )
        if self.flange_radius_mm < self.body_radius_mm:
            raise PedestalDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"flange_radius_mm ({self.flange_radius_mm:.2f}) cannot be less "
                f"than body_radius_mm ({self.body_radius_mm:.2f}).",
            )

        # ---- Vertical ordering: bottom < ear shelf < cavity top < seat ----
        seat_bottom_z = self.total_height_mm - self.bearing_seat_depth_mm
        if not (
            self.flange_thickness_mm
            < self.ear_shelf_z_mm
            < self.cavity_top_z_mm
            < seat_bottom_z
        ):
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                "Vertical layout is out of order: expected flange top "
                f"({self.flange_thickness_mm:.2f}) < ear shelf "
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

        # ---- Bolt channels: inside the body, clear of the cavity ----------
        channel_radius = self.bolt_channel_diameter_mm / 2.0
        channel_outer = self.bolt_circle_radius_mm + channel_radius
        if self.body_radius_mm - channel_outer < self.min_wall_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Bolt access channels reach r = {channel_outer:.2f} mm, "
                f"leaving {self.body_radius_mm - channel_outer:.2f} mm of wall "
                f"against a body radius of {self.body_radius_mm:.2f} mm; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )

        half_ear = self.cavity_ear_length_mm / 2.0
        half_width = self.cavity_width_mm / 2.0
        for index, (bx, by) in enumerate(self.bolt_positions):
            gap = self._rect_distance_to_point(half_ear, half_width, bx, by)
            if gap < channel_radius + self.min_wall_thickness_mm:
                raise PedestalDesignError(
                    DesignStatus.FEATURE_COLLISION,
                    f"Bolt {index} at ({bx:.2f}, {by:.2f}) is {gap:.2f} mm from "
                    f"the servo ear slot, but its access channel needs "
                    f"{channel_radius:.2f} mm plus "
                    f"{self.min_wall_thickness_mm:.2f} mm of wall. Rotate the "
                    "bolt pattern with bolt_azimuth_offset_deg, or widen the "
                    "bolt circle.",
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
        if self.cable_slot_height_mm >= self.ear_shelf_z_mm - self.flange_thickness_mm:
            raise PedestalDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Cable slot ({self.cable_slot_height_mm:.2f} mm tall) does not "
                f"fit between the flange top ({self.flange_thickness_mm:.2f} mm) "
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

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        seat_bottom_z = self.total_height_mm - self.bearing_seat_depth_mm
        ear_corner_radius = math.hypot(
            abs(self.cavity_offset_x_mm) + self.cavity_ear_length_mm / 2.0,
            self.cavity_width_mm / 2.0,
        )
        return (
            f"Base pedestal parameters\n"
            f"------------------------\n"
            f"  Total height           : {self.total_height_mm:.2f} mm\n"
            f"  Body outer diameter    : {2 * self.body_radius_mm:.2f} mm\n"
            f"  Flange outer diameter  : {2 * self.flange_radius_mm:.2f} mm\n"
            f"  Flange thickness       : {self.flange_thickness_mm:.2f} mm\n"
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
            f"  Desk bolts             : {self.bolt_count} x "
            f"{self.bolt_hole_diameter_mm:.2f} mm on a "
            f"{2 * self.bolt_circle_radius_mm:.1f} mm bolt circle\n"
            f"  Bolt access channels   : {self.bolt_channel_diameter_mm:.2f} mm dia, "
            f"offset {self.bolt_azimuth_offset_deg:.0f} deg\n"
            f"  Servo screws           : 4 x "
            f"{self.servo_screw_hole_diameter_mm:.2f} mm, "
            f"{self.servo_screw_hole_depth_mm:.2f} mm deep\n"
            f"\n"
            f"  Thinnest radial wall   : "
            f"{self.body_radius_mm - ear_corner_radius:.2f} mm "
            f"(minimum {self.min_wall_thickness_mm:.2f} mm)\n"
            f"  Ceiling thickness      : "
            f"{seat_bottom_z - self.cavity_top_z_mm:.2f} mm\n"
        )


def build_pedestal(params: Optional[PedestalParameters] = None) -> Part:
    """
    Construct the pedestal solid.

    Built in build123d's algebra mode: start with the flange and body as a
    union of two cylinders, then subtract each internal feature. Every
    subtraction is positioned from ``params``, so the model has no literals of
    its own.

    Parameters
    ----------
    params:
        Resolved dimensions. Defaults to :meth:`PedestalParameters.from_geometry`.

    Returns
    -------
    build123d.Part
        A single solid, origin at the centre of the flange's underside, +Z up.

    Raises
    ------
    PedestalDesignError
        If ``params`` fails :meth:`PedestalParameters.validate`.
    """
    params = PedestalParameters.from_geometry() if params is None else params
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Solid stock: flange disc plus body column -----------------------
    part = Cylinder(
        radius=params.flange_radius_mm,
        height=params.flange_thickness_mm,
        align=bottom,
    )
    part += Pos(0, 0, params.flange_thickness_mm) * Cylinder(
        radius=params.body_radius_mm,
        height=params.total_height_mm - params.flange_thickness_mm,
        align=bottom,
    )

    # ---- Servo cavity, lower section: wide enough to pass the ears -------
    # Open at the bottom so the servo is inserted from underneath and pushed
    # up until its ears meet the shelf where this section ends.
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

    # ---- Desk mounting: clearance hole through the flange, plus a hex-key
    #      access channel above it. --------------------------------------
    for bolt_x, bolt_y in params.bolt_positions:
        part -= Pos(bolt_x, bolt_y, 0) * Cylinder(
            radius=params.bolt_hole_diameter_mm / 2.0,
            height=params.flange_thickness_mm,
            align=bottom,
        )
        part -= Pos(bolt_x, bolt_y, params.flange_thickness_mm) * Cylinder(
            radius=params.bolt_channel_diameter_mm / 2.0,
            height=params.total_height_mm - params.flange_thickness_mm,
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

    # ---- Radial cable slot, on the +X side away from the offset body -----
    slot_z = params.flange_thickness_mm + (
        params.ear_shelf_z_mm - params.flange_thickness_mm - params.cable_slot_height_mm
    ) / 2.0
    part -= Pos(params.body_radius_mm / 2.0, 0, slot_z) * Box(
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
