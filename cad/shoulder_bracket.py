"""
Shoulder bracket -- the clevis that carries the shoulder pitch servo.

Run directly to write both STLs::

    python3 -m cad.shoulder_bracket
    python3 -m cad.shoulder_bracket --report      # dimensions only

Outputs ``cad/output/shoulder_bracket.stl`` and
``cad/output/shoulder_idler_plug.stl``.

What the part does
------------------
It bolts to the yaw turntable and holds the second DS3218 between two walls.
That servo's output shaft *is* the shoulder pitch axis, and the bracket's whole
purpose is to put that axis exactly ``ArmGeometry.base_height_mm`` above the
desk. The chain that gets it there -- pedestal, bearing, turntable, bracket --
is checked end to end by ``tests/test_cad.py``.

Shape, looking along the pitch axis::

        z = 34.5   +------+                +------+   wall tops, slot open
                   |      |   servo slot   |      |
        z = 24.0   |   O  |. . . . . . . . |  O   |   pitch axis
                   |      |                |      |
        z =  6.0   +------+----------------+------+   base plate top
                   |                              |
        z =  0.0   +------------------------------+   on the turntable
                 x = -40.25                    x = +31.75

Why the servo lies on its side
------------------------------
Because standing it up does not fit. The base plate underside to the pitch axis
is 24 mm; the servo's mounting ears are 49.5 mm apart along its length, so
standing the body on end puts the lower ear at 24 - 24.75 = -0.75 mm, i.e.
below the turntable the bracket is bolted to. Laid down, the body's 20 mm width
straddles the axis from 14 to 34 mm and both ears are reachable. Laying it down
also keeps every part of the bracket within about 24 mm of the pitch axis, which
is what lets the upper arm's yoke swing through the joint's full travel.

Why the two walls are identical
-------------------------------
docs/PROOF_OF_CONCEPT.md section 2.2 records that this joint wants roughly
34.5 kg.cm and the DS3218 supplies 20. Both walls therefore carry the same
servo slot and the same four mounting holes; today the undriven one is filled
by the idler plug built here, which carries the axle for the upper arm's
pivot bearing.

.. note::
   **A second DS3218 in parallel does not fit at this wall spacing**, and it is
   worth being plain about that. The walls sit 31 mm apart because that is what
   one servo needs: its ears bear on the driven wall and its body reaches
   30.5 mm back to the other. Two servos back to back need 61 mm between the
   ear planes, which would widen the yoke from 87 to about 108 mm. The wall
   spacing is a parameter, not a shape, so choosing the twin-servo route in
   Phase C is a re-run rather than a redesign -- but it is not a drop-in swap
   for the plug, and the option in section 2.2 should say so.

Coordinate frame
----------------
Origin on the yaw axis at the base plate's underside, which is the turntable's
top face. ``+X`` is the arm-forward direction at yaw zero, ``+Z`` is up, and
the servo's output shaft points along ``+Y`` -- the same axis
``forward_kinematics.py`` rotates shoulder pitch about, so the servo's positive
rotation is positive theta_2 with no sign flip.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from build123d import Align, Box, Cylinder, Part, Pos, Rot, export_stl

from cad._design import DesignRuleError, DesignStatus
from cad._primitives import right_triangle_prism
from src.geometry import DEFAULT_ARM, DEFAULT_HARDWARE, ArmGeometry, HardwareSpec

__all__ = [
    "ShoulderBracketDesignError",
    "ShoulderBracketParameters",
    "build_shoulder_bracket",
    "build_idler_plug",
    "export_shoulder_bracket",
    "export_idler_plug",
    "DEFAULT_STL_PATH",
    "DEFAULT_PLUG_STL_PATH",
]

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_STL_PATH = _OUTPUT_DIR / "shoulder_bracket.stl"
DEFAULT_PLUG_STL_PATH = _OUTPUT_DIR / "shoulder_idler_plug.stl"


class ShoulderBracketDesignError(DesignRuleError):
    """Raised when a parameter set would produce an unbuildable bracket."""


@dataclass(frozen=True)
class ShoulderBracketParameters:
    """Fully resolved shoulder-bracket dimensions, in millimetres."""

    # ---- Base plate ---------------------------------------------------------
    plate_x_min_mm: float
    plate_x_max_mm: float
    plate_half_y_mm: float
    plate_thickness_mm: float

    # ---- Walls --------------------------------------------------------------
    wall_x_min_mm: float
    wall_x_max_mm: float
    wall_top_z_mm: float
    wall_inner_y_mm: float
    wall_thickness_mm: float
    gusset_size_mm: float

    # ---- Servo --------------------------------------------------------------
    pitch_axis_z_mm: float
    cavity_x_min_mm: float
    cavity_x_max_mm: float
    cavity_z_min_mm: float
    cavity_z_max_mm: float
    servo_screw_positions: Tuple[Tuple[float, float], ...]
    servo_screw_diameter_mm: float
    servo_screw_depth_mm: float
    shaft_bore_diameter_mm: float
    shaft_face_y_mm: float
    horn_face_y_mm: float
    shaft_direction: str

    # ---- Turntable interface ------------------------------------------------
    turntable_bolt_positions: Tuple[Tuple[float, float], ...]
    turntable_bolt_clearance_mm: float
    turntable_radius_mm: float

    # ---- Cable exit ---------------------------------------------------------
    cable_slot_x_mm: float
    cable_slot_y_mm: float

    # ---- Idler plug ---------------------------------------------------------
    plug_thickness_mm: float
    plug_boss_diameter_mm: float
    plug_boss_length_mm: float
    idler_axle_diameter_mm: float
    idler_axle_length_mm: float
    idler_bearing_seat_diameter_mm: float
    idler_bearing_width_mm: float

    # ---- Carried through for validation -------------------------------------
    min_wall_thickness_mm: float
    shoulder_pivot_z_mm: float
    stack_below_mm: float

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_geometry(
        cls,
        arm: Optional[ArmGeometry] = None,
        hardware: Optional[HardwareSpec] = None,
    ) -> "ShoulderBracketParameters":
        """
        Derive every dimension from the geometry and hardware singletons.

        Raises
        ------
        ShoulderBracketDesignError
            If the resulting design violates a clearance.
        """
        arm = DEFAULT_ARM if arm is None else arm
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        spec = hardware.shoulder_bracket
        servo = hardware.shoulder_pitch_servo
        horn = hardware.servo_horn
        idler = hardware.shoulder_idler_bearing
        turntable = hardware.yaw_turntable
        clearance = hardware.print_clearance_mm
        wall = hardware.min_wall_thickness_mm

        pitch_axis_z = spec.bracket_height_mm

        # ---- The servo, lying on its side --------------------------------
        # Its length runs along X with the shaft 10 mm from one end, so the
        # body sits behind the axis; its width straddles the axis vertically.
        body_x_centre = -servo.body_offset_from_shaft_axis_mm
        cavity_x_half = (servo.body_length_mm + 2.0 * clearance) / 2.0
        cavity_z_half = (servo.body_width_mm + 2.0 * clearance) / 2.0

        # The ears sit ear_offset_from_shaft_face_mm inside the shaft-end face,
        # so what has to fit between the walls is the body BEHIND the ears --
        # shared symmetrically, which is what makes the two walls identical.
        wall_inner_y = servo.body_depth_behind_ears_mm / 2.0 + clearance
        wall_outer_y = wall_inner_y + spec.wall_thickness_mm
        shaft_face_y = wall_inner_y + servo.ear_offset_from_shaft_face_mm
        horn_face_y = (
            shaft_face_y + servo.shaft_boss_height_mm + horn.total_height_mm
        )

        screw_positions = tuple(
            (
                body_x_centre + sx * servo.flange_hole_spacing_long_mm / 2.0,
                pitch_axis_z + sz * servo.flange_hole_spacing_short_mm / 2.0,
            )
            for sx in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        )
        screw_x = [x for x, _ in screw_positions]
        screw_z = [z for _, z in screw_positions]
        hole_radius = servo.flange_hole_diameter_mm / 2.0

        wall_x_min = min(screw_x) - hole_radius - wall
        wall_x_max = max(screw_x) + hole_radius + wall
        wall_top_z = max(
            max(screw_z) + hole_radius + wall,
            pitch_axis_z + cavity_z_half,
        )

        # ---- Base plate ---------------------------------------------------
        # Anchored on the wall's rear edge and grown forward to whatever the
        # turntable's bolt pattern needs.
        plate_x_min = wall_x_min
        plate_x_max = plate_x_min + spec.base_plate_x_mm

        params = cls(
            plate_x_min_mm=plate_x_min,
            plate_x_max_mm=plate_x_max,
            plate_half_y_mm=spec.base_plate_y_mm / 2.0,
            plate_thickness_mm=spec.base_plate_thickness_mm,
            wall_x_min_mm=wall_x_min,
            wall_x_max_mm=wall_x_max,
            wall_top_z_mm=wall_top_z,
            wall_inner_y_mm=wall_inner_y,
            wall_thickness_mm=spec.wall_thickness_mm,
            gusset_size_mm=spec.gusset_size_mm,
            pitch_axis_z_mm=pitch_axis_z,
            cavity_x_min_mm=body_x_centre - cavity_x_half,
            cavity_x_max_mm=body_x_centre + cavity_x_half,
            cavity_z_min_mm=pitch_axis_z - cavity_z_half,
            cavity_z_max_mm=pitch_axis_z + cavity_z_half,
            servo_screw_positions=screw_positions,
            servo_screw_diameter_mm=servo.flange_hole_diameter_mm,
            servo_screw_depth_mm=spec.servo_screw_depth_mm,
            shaft_bore_diameter_mm=servo.shaft_boss_diameter_mm + 2.0 * clearance,
            shaft_face_y_mm=shaft_face_y,
            horn_face_y_mm=horn_face_y,
            shaft_direction=spec.shaft_direction,
            turntable_bolt_positions=turntable.bracket_bolt_positions(),
            turntable_bolt_clearance_mm=(
                turntable.bracket_bolt_nominal_diameter_mm + 0.4
            ),
            turntable_radius_mm=turntable.diameter_mm / 2.0,
            cable_slot_x_mm=spec.cable_slot_mm[1],
            cable_slot_y_mm=spec.cable_slot_mm[0],
            plug_thickness_mm=spec.idler_plug_thickness_mm,
            plug_boss_diameter_mm=spec.idler_boss_diameter_mm,
            plug_boss_length_mm=wall,
            idler_axle_diameter_mm=(
                idler.bore_diameter_mm - 2.0 * clearance
            ),
            idler_axle_length_mm=(
                horn_face_y
                + idler.width_mm
                - (wall_outer_y + spec.idler_plug_thickness_mm + wall)
            ),
            idler_bearing_seat_diameter_mm=idler.seat_diameter_mm,
            idler_bearing_width_mm=idler.width_mm,
            min_wall_thickness_mm=wall,
            shoulder_pivot_z_mm=hardware.shoulder_pivot_z_mm(arm),
            stack_below_mm=(
                hardware.pedestal_height_mm(arm)
                + hardware.thrust_bearing.proud_mm
                + hardware.yaw_turntable.thickness_mm
            ),
        )
        params.validate()
        return params

    # =====================================================================
    # Derived accessors
    # =====================================================================

    @property
    def wall_outer_y_mm(self) -> float:
        return self.wall_inner_y_mm + self.wall_thickness_mm

    @property
    def plug_outer_y_mm(self) -> float:
        """Outer face of the idler plug, in mm from the mid-plane."""
        return self.wall_outer_y_mm + self.plug_thickness_mm

    @property
    def servo_cavity_span_mm(self) -> Tuple[float, float, float]:
        """(X, Y, Z) of the space the servo body occupies, in mm."""
        return (
            self.cavity_x_max_mm - self.cavity_x_min_mm,
            2.0 * self.wall_inner_y_mm,
            self.cavity_z_max_mm - self.cavity_z_min_mm,
        )

    @property
    def shaft_axis_z_in_desk_frame_mm(self) -> float:
        """
        Where this bracket puts the shoulder pitch axis above the desk, in mm.

        The whole point of the part: pedestal + bearing proud + turntable +
        this rise. It must equal ``ArmGeometry.base_height_mm``, which
        :meth:`validate` insists on.
        """
        return self.stack_below_mm + self.pitch_axis_z_mm

    @property
    def max_radius_from_pitch_axis_mm(self) -> float:
        """
        Furthest any bracket material reaches from the pitch axis, in mm.

        The upper arm's yoke has to start outside this or it cannot swing.
        Taken over the wall's four corners, which bound the whole part above
        the base plate.
        """
        return max(
            ((x**2 + (z - self.pitch_axis_z_mm) ** 2) ** 0.5)
            for x in (self.wall_x_min_mm, self.wall_x_max_mm)
            for z in (0.0, self.wall_top_z_mm)
        )

    @property
    def turntable_overhang_mm(self) -> float:
        """
        How far the base plate's furthest corner reaches past the turntable.

        Positive by design: the plate is sized by the servo's 54.5 mm flange,
        which is wider than the disc the bearing needs. The load path is the
        bolt pattern, well inside the turntable, so the overhang is unloaded --
        it is only worth knowing so nobody reads it as a modelling error.
        """
        corner = (
            max(abs(self.plate_x_min_mm), abs(self.plate_x_max_mm)) ** 2
            + self.plate_half_y_mm**2
        ) ** 0.5
        return corner - self.turntable_radius_mm

    # =====================================================================
    # Validation
    # =====================================================================

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unbuildable bracket.

        Raises
        ------
        ShoulderBracketDesignError
            Naming the first violated constraint.
        """
        for name, value in (
            ("plate_thickness_mm", self.plate_thickness_mm),
            ("wall_thickness_mm", self.wall_thickness_mm),
            ("pitch_axis_z_mm", self.pitch_axis_z_mm),
            ("gusset_size_mm", self.gusset_size_mm),
            ("plug_thickness_mm", self.plug_thickness_mm),
        ):
            if value <= 0.0:
                raise ShoulderBracketDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )

        # ---- The reason this part exists ----------------------------------
        if abs(self.shaft_axis_z_in_desk_frame_mm - self.shoulder_pivot_z_mm) > 1e-6:
            raise ShoulderBracketDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"The stack puts the pitch axis at "
                f"{self.shaft_axis_z_in_desk_frame_mm:.3f} mm above the desk "
                f"but ArmGeometry.base_height_mm is "
                f"{self.shoulder_pivot_z_mm:.3f} mm. Everything below "
                "(pedestal, bearing, turntable) plus this bracket's rise has "
                "to add up to the kinematic base height exactly.",
            )

        # ---- The servo has to fit, and clear the plate ---------------------
        if self.cavity_z_min_mm <= self.plate_thickness_mm:
            raise ShoulderBracketDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The servo's underside reaches z = "
                f"{self.cavity_z_min_mm:.2f} mm but the base plate's top face "
                f"is at {self.plate_thickness_mm:.2f} mm. Raise "
                "bracket_height_mm or thin the plate.",
            )
        lowest_screw_z = min(z for _, z in self.servo_screw_positions)
        screw_edge = lowest_screw_z - self.servo_screw_diameter_mm / 2.0
        if screw_edge <= self.plate_thickness_mm:
            raise ShoulderBracketDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The lowest servo mounting hole reaches down to z = "
                f"{screw_edge:.2f} mm, into the base plate at "
                f"{self.plate_thickness_mm:.2f} mm. This is what rules out "
                "standing the servo on end: its ears are 49.5 mm apart and the "
                "bracket only rises 24 mm.",
            )
        if self.servo_screw_depth_mm >= self.wall_thickness_mm:
            raise ShoulderBracketDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"A {self.servo_screw_depth_mm:.2f} mm blind hole in a "
                f"{self.wall_thickness_mm:.2f} mm wall leaves no floor.",
            )
        for screw_x, screw_z in self.servo_screw_positions:
            if self.cavity_x_min_mm <= screw_x <= self.cavity_x_max_mm:
                raise ShoulderBracketDesignError(
                    DesignStatus.FEATURE_COLLISION,
                    f"A servo mounting hole at x = {screw_x:.2f} mm falls "
                    f"inside the servo slot "
                    f"({self.cavity_x_min_mm:.2f} .. "
                    f"{self.cavity_x_max_mm:.2f}), where there is no wall.",
                )

        # ---- Walls have to stand on the plate -----------------------------
        if (
            self.wall_x_min_mm < self.plate_x_min_mm - 1e-9
            or self.wall_x_max_mm > self.plate_x_max_mm + 1e-9
        ):
            raise ShoulderBracketDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The walls span x {self.wall_x_min_mm:.2f} .. "
                f"{self.wall_x_max_mm:.2f} mm but the base plate only reaches "
                f"{self.plate_x_min_mm:.2f} .. {self.plate_x_max_mm:.2f}.",
            )
        if self.wall_outer_y_mm > self.plate_half_y_mm + 1e-9:
            raise ShoulderBracketDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The walls reach y = {self.wall_outer_y_mm:.2f} mm but the "
                f"base plate only reaches {self.plate_half_y_mm:.2f}.",
            )

        # ---- Turntable screws must miss the walls and stay on the plate ----
        radius = self.turntable_bolt_clearance_mm / 2.0
        for bolt_x, bolt_y in self.turntable_bolt_positions:
            if abs(bolt_y) + radius > self.wall_inner_y_mm:
                raise ShoulderBracketDesignError(
                    DesignStatus.FEATURE_COLLISION,
                    f"A turntable screw at y = {bolt_y:.2f} mm lands under a "
                    f"wall, whose inner face is at "
                    f"{self.wall_inner_y_mm:.2f} mm. There would be no way to "
                    "get a driver to it. Bring the pattern inboard along Y or "
                    "outboard past the walls.",
                )
            if (
                bolt_x - radius < self.plate_x_min_mm + self.min_wall_thickness_mm
                or bolt_x + radius > self.plate_x_max_mm - self.min_wall_thickness_mm
            ):
                raise ShoulderBracketDesignError(
                    DesignStatus.WALL_TOO_THIN,
                    f"A turntable screw at x = {bolt_x:.2f} mm leaves less "
                    f"than {self.min_wall_thickness_mm:.2f} mm of plate edge.",
                )

        # ---- The cable slot must not undercut a wall or a screw -----------
        slot_x_max = self.plate_x_min_mm + self.cable_slot_x_mm
        for bolt_x, bolt_y in self.turntable_bolt_positions:
            if bolt_x - radius < slot_x_max and abs(bolt_y) < self.cable_slot_y_mm / 2.0:
                raise ShoulderBracketDesignError(
                    DesignStatus.FEATURE_COLLISION,
                    f"The cable slot reaches x = {slot_x_max:.2f} mm and would "
                    f"break into the turntable screw at "
                    f"({bolt_x:.2f}, {bolt_y:.2f}).",
                )
        if self.cable_slot_y_mm / 2.0 >= self.wall_inner_y_mm:
            raise ShoulderBracketDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The cable slot is {self.cable_slot_y_mm:.2f} mm wide and "
                f"would cut into the walls at "
                f"{self.wall_inner_y_mm:.2f} mm.",
            )

        # ---- The idler axle has to reach its bearing ----------------------
        if self.idler_axle_length_mm <= self.idler_bearing_width_mm:
            raise ShoulderBracketDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"The idler axle is {self.idler_axle_length_mm:.2f} mm long "
                f"but has to pass through a "
                f"{self.idler_bearing_width_mm:.2f} mm bearing and span the "
                "gap to it.",
            )
        if self.plug_boss_diameter_mm <= self.idler_axle_diameter_mm:
            raise ShoulderBracketDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"The idler boss ({self.plug_boss_diameter_mm:.2f} mm) must be "
                f"wider than the axle it carries "
                f"({self.idler_axle_diameter_mm:.2f} mm).",
            )
        return DesignStatus.OK

    # =====================================================================
    # Reporting
    # =====================================================================

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        cav_x, cav_y, cav_z = self.servo_cavity_span_mm
        return (
            f"Shoulder bracket parameters\n"
            f"---------------------------\n"
            f"  Base plate             : x {self.plate_x_min_mm:.2f} .. "
            f"{self.plate_x_max_mm:.2f}, y +/-{self.plate_half_y_mm:.2f}, "
            f"{self.plate_thickness_mm:.2f} mm thick\n"
            f"  Walls                  : x {self.wall_x_min_mm:.2f} .. "
            f"{self.wall_x_max_mm:.2f}, z {self.plate_thickness_mm:.2f} .. "
            f"{self.wall_top_z_mm:.2f}, inner faces at y "
            f"+/-{self.wall_inner_y_mm:.2f}, {self.wall_thickness_mm:.2f} mm "
            f"thick\n"
            f"  Servo slot             : {cav_x:.2f} (X) x {cav_y:.2f} (Y) x "
            f"{cav_z:.2f} (Z) mm, open at the top\n"
            f"  Servo screws           : 4 x "
            f"{self.servo_screw_diameter_mm:.2f} mm blind, "
            f"{self.servo_screw_depth_mm:.2f} mm deep, at x "
            f"{sorted({x for x, _ in self.servo_screw_positions})} "
            f"z {sorted({z for _, z in self.servo_screw_positions})}\n"
            f"\n"
            f"  Pitch axis             : z = {self.pitch_axis_z_mm:.2f} mm "
            f"local, {self.shaft_axis_z_in_desk_frame_mm:.3f} mm above the "
            f"desk (target {self.shoulder_pivot_z_mm:.3f})\n"
            f"  Shaft direction        : {self.shaft_direction} "
            f"(matches the FK rotation axis for shoulder pitch)\n"
            f"  Shaft face at y        : {self.shaft_face_y_mm:.2f} mm\n"
            f"  Horn face at y         : {self.horn_face_y_mm:.2f} mm "
            f"(the upper arm's driven flange bears here)\n"
            f"  Bracket max radius     : "
            f"{self.max_radius_from_pitch_axis_mm:.2f} mm from the pitch axis "
            f"(the yoke must start outside this)\n"
            f"\n"
            f"  Turntable screws       : 4 x "
            f"{self.turntable_bolt_clearance_mm:.2f} mm clearance at "
            f"{sorted(set(self.turntable_bolt_positions))}\n"
            f"  Plate overhangs disc   : "
            f"{self.turntable_overhang_mm:.2f} mm at the rear corners\n"
            f"  Cable slot             : {self.cable_slot_x_mm:.2f} (X) x "
            f"{self.cable_slot_y_mm:.2f} (Y) mm notch in the rear edge\n"
            f"\n"
            f"  Idler plug             : {self.plug_thickness_mm:.2f} mm plate "
            f"on the far wall, boss "
            f"{self.plug_boss_diameter_mm:.2f} x "
            f"{self.plug_boss_length_mm:.2f} mm\n"
            f"  Idler axle             : "
            f"{self.idler_axle_diameter_mm:.2f} mm dia x "
            f"{self.idler_axle_length_mm:.2f} mm, running in a "
            f"{self.idler_bearing_seat_diameter_mm:.2f} mm seat in the yoke\n"
        )


def _wall(params: ShoulderBracketParameters, sign: float) -> Part:
    """One wall, with its servo slot and blind mounting holes, at +/-Y."""
    bottom = (Align.CENTER, Align.CENTER, Align.MIN)
    inner = sign * params.wall_inner_y_mm
    centre_y = inner + sign * params.wall_thickness_mm / 2.0

    wall = Pos(
        (params.wall_x_min_mm + params.wall_x_max_mm) / 2.0,
        centre_y,
        params.plate_thickness_mm,
    ) * Box(
        params.wall_x_max_mm - params.wall_x_min_mm,
        params.wall_thickness_mm,
        params.wall_top_z_mm - params.plate_thickness_mm,
        align=bottom,
    )

    # Servo slot: open at the top so the servo drops in past its 54.5 mm ears,
    # which could never pass through a closed window.
    slot_height = params.wall_top_z_mm - params.cavity_z_min_mm
    wall -= Pos(
        (params.cavity_x_min_mm + params.cavity_x_max_mm) / 2.0,
        centre_y,
        params.cavity_z_min_mm,
    ) * Box(
        params.cavity_x_max_mm - params.cavity_x_min_mm,
        params.wall_thickness_mm * 2.0,
        slot_height,
        align=bottom,
    )

    # Blind pilot holes, drilled from the inner face outward.
    for screw_x, screw_z in params.servo_screw_positions:
        wall -= Pos(screw_x, inner, screw_z) * Rot(90, 0, 0) * Cylinder(
            radius=params.servo_screw_diameter_mm / 2.0,
            height=params.servo_screw_depth_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN if sign < 0 else Align.MAX),
        )
    return wall


def build_shoulder_bracket(
    params: Optional[ShoulderBracketParameters] = None,
) -> Part:
    """
    Construct the shoulder bracket solid.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the yaw axis at the base plate's underside.

    Raises
    ------
    ShoulderBracketDesignError
        If ``params`` fails :meth:`ShoulderBracketParameters.validate`.
    """
    params = (
        ShoulderBracketParameters.from_geometry() if params is None else params
    )
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Base plate --------------------------------------------------------
    part = Pos(
        (params.plate_x_min_mm + params.plate_x_max_mm) / 2.0, 0, 0
    ) * Box(
        params.plate_x_max_mm - params.plate_x_min_mm,
        2.0 * params.plate_half_y_mm,
        params.plate_thickness_mm,
        align=bottom,
    )

    # ---- Two identical walls ----------------------------------------------
    for sign in (1.0, -1.0):
        part += _wall(params, sign)

        # Gusset along the inner foot of each wall. A triangular prism rather
        # than a fillet, for the reason cad/README.md gives.
        part += (
            Pos(
                (params.wall_x_min_mm + params.wall_x_max_mm) / 2.0,
                sign * params.wall_inner_y_mm,
                params.plate_thickness_mm,
            )
            * Rot(0, 0, -90.0 * sign)
            * right_triangle_prism(
                params.gusset_size_mm,
                params.gusset_size_mm,
                params.wall_x_max_mm - params.wall_x_min_mm,
            )
        )

    # No separate shaft bore: the servo slot already spans the pitch axis, so
    # the case's protruding top section and its output shaft pass through the
    # same opening the body is inserted into.

    # ---- Turntable mounting holes through the base plate ------------------
    for bolt_x, bolt_y in params.turntable_bolt_positions:
        part -= Pos(bolt_x, bolt_y, 0) * Cylinder(
            radius=params.turntable_bolt_clearance_mm / 2.0,
            height=params.plate_thickness_mm,
            align=bottom,
        )

    # ---- Cable notch in the rear edge -------------------------------------
    # Not a hole on the yaw axis: the yaw servo's shaft and its horn fill that
    # axis solid, so the shoulder servo's lead cannot pass down it. It leaves
    # at the rear, outside the turntable's rim, and needs a service loop for
    # the +/-135 degrees of yaw travel.
    part -= Pos(
        params.plate_x_min_mm + params.cable_slot_x_mm / 2.0, 0, 0
    ) * Box(
        params.cable_slot_x_mm,
        params.cable_slot_y_mm,
        params.plate_thickness_mm,
        align=bottom,
    )
    return part


def build_idler_plug(
    params: Optional[ShoulderBracketParameters] = None,
) -> Part:
    """
    Construct the idler plug: the plate that fills the undriven wall.

    It bolts to the far wall's outer face with the same four screws a servo
    would use, and carries the axle the upper arm's pivot bearing runs on.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the pitch axis at the plug's mounting face,
        extruded along +Y; the caller mirrors it onto whichever wall is
        undriven.
    """
    params = (
        ShoulderBracketParameters.from_geometry() if params is None else params
    )
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    plate_z_min = params.cavity_z_min_mm - params.min_wall_thickness_mm
    part = Pos(
        (params.wall_x_min_mm + params.wall_x_max_mm) / 2.0,
        0,
        plate_z_min,
    ) * Rot(-90, 0, 0) * Box(
        params.wall_x_max_mm - params.wall_x_min_mm,
        params.wall_top_z_mm - plate_z_min,
        params.plug_thickness_mm,
        # Align.MAX on the box's Y so that, once Rot(-90) turns +Y into -Z,
        # the plate grows upward from plate_z_min rather than downward.
        align=(Align.CENTER, Align.MAX, Align.MIN),
    )

    # Boss and axle, both on the pitch axis, growing outward.
    part += Pos(0, params.plug_thickness_mm, params.pitch_axis_z_mm) * Rot(
        -90, 0, 0
    ) * Cylinder(
        radius=params.plug_boss_diameter_mm / 2.0,
        height=params.plug_boss_length_mm,
        align=bottom,
    )
    part += Pos(
        0, params.plug_thickness_mm + params.plug_boss_length_mm,
        params.pitch_axis_z_mm,
    ) * Rot(-90, 0, 0) * Cylinder(
        radius=params.idler_axle_diameter_mm / 2.0,
        height=params.idler_axle_length_mm,
        align=bottom,
    )

    # Clearance holes on the servo's own pattern, so the plug and a servo are
    # interchangeable in this wall.
    for screw_x, screw_z in params.servo_screw_positions:
        part -= Pos(screw_x, 0, screw_z) * Rot(-90, 0, 0) * Cylinder(
            radius=(params.servo_screw_diameter_mm + 0.4) / 2.0,
            height=params.plug_thickness_mm,
            align=bottom,
        )
    return part


def export_shoulder_bracket(
    output_path: Optional[Path] = None,
    params: Optional[ShoulderBracketParameters] = None,
) -> Path:
    """Build the bracket and write it to an STL."""
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_shoulder_bracket(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def export_idler_plug(
    output_path: Optional[Path] = None,
    params: Optional[ShoulderBracketParameters] = None,
) -> Path:
    """Build the idler plug and write it to an STL."""
    output_path = (
        DEFAULT_PLUG_STL_PATH if output_path is None else Path(output_path)
    )
    part = build_idler_plug(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the shoulder bracket STLs from src/geometry.py."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument("--plug-output", type=Path, default=DEFAULT_PLUG_STL_PATH)
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = ShoulderBracketParameters.from_geometry()
    except ShoulderBracketDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    if args.report:
        return 0

    for label, exporter, destination in (
        ("bracket", export_shoulder_bracket, args.output),
        ("idler plug", export_idler_plug, args.plug_output),
    ):
        written = exporter(destination, params)
        print(
            f"Wrote {label:<10} {written}  "
            f"({written.stat().st_size / 1024.0:.1f} KB)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
