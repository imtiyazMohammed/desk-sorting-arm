"""
Desk clamp pressure foot -- the puck on the clamp screw's tip.

Run directly to write ``cad/output/desk_clamp_pressure_foot.stl``::

    python3 -m cad.desk_clamp_pressure_foot
    python3 -m cad.desk_clamp_pressure_foot --report

What the part does
------------------
A small disc that threads onto the M8 clamp screw's tip and bears on the
desk's underside. Bottom to top along +Z:

1. **Conical seat** the screw's tip pivots in. The foot rests on the tip
   rather than threading onto it, so it stays still while the screw turns.
2. **Web**, the material carrying the load from the seat's apex into the
   pad. Loaded in pure compression, so it does not need a structural
   thickness.
3. **Pad recess** in the top face, for a glued-in anti-slip rubber disc that
   contacts the desk.

Why it swivels rather than threads on (Session D.1d)
----------------------------------------------------
The first version threaded onto the screw's tip. That fails in two ways at
once: the foot turns with the screw, dragging its rubber pad across the desk
as you tighten, and the friction that resists tightening then acts at the
pad's radius instead of the screw's -- roughly halving the preload a given
hand torque produces. Seated on a cone, the foot is free to stay put while the
screw turns inside it, and the friction lever shrinks to the tip's own radius.

The foot is not captive: it rests on the tip and will drop off if the assembly
is inverted. Fit it as the clamp goes on; once the pad touches the desk it
stays put on its own.

Why it exists
-------------
An M8 tip is 8 mm across and the clamp puts a few hundred newtons through it.
Bearing that directly on a desk's underside would dent or bite into most
materials, and a point contact is also the least stable way to hold the
assembly. The foot spreads the load over roughly nine times the area and
carries the second anti-slip pad.

Its height is not free: at the thickest supported desk the throat leaves only
``throat_max_opening_mm - max_desk_thickness_mm`` of room between the bottom
arm and the desk, and the foot has to fit inside that.
:meth:`PressureFootParameters.validate` enforces it.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from build123d import Align, Cone, Cylinder, Part, Pos, export_stl

from cad._design import DesignRuleError, DesignStatus
from src.geometry import DEFAULT_HARDWARE, HardwareSpec

__all__ = [
    "PressureFootDesignError",
    "PressureFootParameters",
    "build_pressure_foot",
    "export_pressure_foot",
    "DEFAULT_STL_PATH",
]

DEFAULT_STL_PATH = (
    Path(__file__).resolve().parent / "output" / "desk_clamp_pressure_foot.stl"
)


class PressureFootDesignError(DesignRuleError):
    """Raised when a parameter set would produce an unbuildable pressure foot."""


@dataclass(frozen=True)
class PressureFootParameters:
    """Fully resolved pressure-foot dimensions, in millimetres."""

    diameter_mm: float
    height_mm: float
    seat_diameter_mm: float
    seat_apex_diameter_mm: float
    seat_depth_mm: float
    seat_angle_deg: float
    web_thickness_mm: float
    pad_diameter_mm: float
    pad_recess_depth_mm: float
    tip_contact_radius_mm: float
    tip_seat_height_mm: float
    throat_clearance_mm: float
    min_wall_thickness_mm: float

    @classmethod
    def from_geometry(
        cls, hardware: Optional[HardwareSpec] = None
    ) -> "PressureFootParameters":
        """
        Derive every dimension from the hardware singleton.

        Raises
        ------
        PressureFootDesignError
            If the resulting design violates a clearance.
        """
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        clamp = hardware.desk_clamp

        params = cls(
            diameter_mm=clamp.pressure_foot_diameter_mm,
            height_mm=clamp.pressure_foot_height_mm,
            seat_diameter_mm=clamp.pressure_foot_seat_diameter_mm,
            seat_apex_diameter_mm=clamp.pressure_foot_seat_apex_diameter_mm,
            seat_depth_mm=clamp.pressure_foot_seat_depth_mm,
            seat_angle_deg=clamp.pressure_foot_seat_angle_deg,
            web_thickness_mm=clamp.pressure_foot_web_mm,
            pad_diameter_mm=clamp.pressure_foot_pad_diameter_mm,
            pad_recess_depth_mm=clamp.pad_recess_depth_mm,
            tip_contact_radius_mm=clamp.screw_tip_contact_radius_mm,
            tip_seat_height_mm=clamp.screw_tip_seat_height_mm,
            throat_clearance_mm=clamp.desk_removal_clearance_mm,
            min_wall_thickness_mm=hardware.min_wall_thickness_mm,
        )
        params.validate()
        return params

    # ---- Derived accessors ------------------------------------------------

    @property
    def rise_above_tip_mm(self) -> float:
        """How far the contact face stands above the seated screw tip."""
        return self.height_mm - self.tip_seat_height_mm

    @property
    def pad_area_mm2(self) -> float:
        """Contact area of the glued-in rubber disc, in square millimetres."""
        return 3.141592653589793 * (self.pad_diameter_mm / 2.0) ** 2

    @property
    def seat_wall_thickness_mm(self) -> float:
        """Material between the seat's mouth and the foot's outer surface."""
        return (self.diameter_mm - self.seat_diameter_mm) / 2.0

    @property
    def bearing_area_ratio(self) -> float:
        """How much the foot spreads the load, relative to a bare M8 tip."""
        return (self.pad_diameter_mm / 8.0) ** 2

    # ---- Validation -------------------------------------------------------

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unprintable part.

        Raises
        ------
        PressureFootDesignError
            Naming the first violated constraint.
        """
        for name, value in (
            ("diameter_mm", self.diameter_mm),
            ("height_mm", self.height_mm),
            ("seat_diameter_mm", self.seat_diameter_mm),
            ("seat_depth_mm", self.seat_depth_mm),
            ("web_thickness_mm", self.web_thickness_mm),
            ("pad_diameter_mm", self.pad_diameter_mm),
        ):
            if value <= 0.0:
                raise PressureFootDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )
        stack = self.seat_depth_mm + self.web_thickness_mm + self.pad_recess_depth_mm
        if abs(stack - self.height_mm) > 1e-9:
            raise PressureFootDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"Height {self.height_mm:.2f} mm does not match its parts "
                f"(seat {self.seat_depth_mm:.2f} + web {self.web_thickness_mm} "
                f"+ pad {self.pad_recess_depth_mm} = {stack:.2f} mm).",
            )
        if self.pad_diameter_mm >= self.diameter_mm:
            raise PressureFootDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"The pad recess ({self.pad_diameter_mm:.2f} mm) must be "
                f"smaller than the foot ({self.diameter_mm:.2f} mm).",
            )
        if self.seat_diameter_mm >= self.pad_diameter_mm:
            raise PressureFootDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The seat ({self.seat_diameter_mm:.2f} mm) is not smaller "
                f"than the pad recess ({self.pad_diameter_mm:.2f} mm), so the "
                "web would have no material.",
            )
        if self.tip_contact_radius_mm >= self.seat_diameter_mm / 2.0:
            raise PressureFootDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The screw tip (radius {self.tip_contact_radius_mm:.2f} mm) is "
                f"as wide as the seat's mouth "
                f"({self.seat_diameter_mm / 2.0:.2f} mm radius); it would rest "
                "on the foot's face rather than seating in the cone.",
            )
        rim = (self.diameter_mm - self.pad_diameter_mm) / 2.0
        if rim < self.min_wall_thickness_mm / 2.0:
            raise PressureFootDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {rim:.2f} mm of rim outside the pad recess.",
            )
        if self.height_mm > self.throat_clearance_mm:
            raise PressureFootDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The foot is {self.height_mm:.2f} mm tall but at the thickest "
                f"supported desk the throat leaves only "
                f"{self.throat_clearance_mm:.2f} mm between the bottom arm and "
                "the desk. Shorten the foot or widen the throat.",
            )
        return DesignStatus.OK

    # ---- Reporting --------------------------------------------------------

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        return (
            f"Desk clamp pressure foot parameters\n"
            f"-----------------------------------\n"
            f"  Foot                   : {self.diameter_mm:.2f} mm dia x "
            f"{self.height_mm:.2f} mm\n"
            f"  Swivel seat (underside): {self.seat_diameter_mm:.2f} mm mouth "
            f"-> {self.seat_apex_diameter_mm:.2f} mm flat, "
            f"{self.seat_depth_mm:.2f} mm deep, "
            f"{self.seat_angle_deg:.0f} deg included\n"
            f"  Screw tip seats at     : {self.tip_seat_height_mm:.2f} mm "
            f"(contact radius {self.tip_contact_radius_mm:.2f} mm)\n"
            f"  Web                    : {self.web_thickness_mm:.2f} mm "
            f"(compression only)\n"
            f"  Pad recess (top)       : {self.pad_diameter_mm:.2f} mm dia x "
            f"{self.pad_recess_depth_mm:.2f} mm deep\n"
            f"  Rise above screw tip   : {self.rise_above_tip_mm:.2f} mm\n"
            f"\n"
            f"  Contact area           : {self.pad_area_mm2:.0f} mm2, "
            f"{self.bearing_area_ratio:.1f}x a bare M8 tip\n"
            f"  Throat clearance       : {self.throat_clearance_mm:.2f} mm "
            f"available at the thickest desk\n"
        )


def build_pressure_foot(
    params: Optional[PressureFootParameters] = None,
) -> Part:
    """
    Construct the pressure foot solid.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the screw axis at the foot's underside.

    Raises
    ------
    PressureFootDesignError
        If ``params`` fails :meth:`PressureFootParameters.validate`.
    """
    params = (
        PressureFootParameters.from_geometry() if params is None else params
    )
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    part = Cylinder(
        radius=params.diameter_mm / 2.0, height=params.height_mm, align=bottom
    )
    # Conical swivel seat, opening downward: mouth at the underside, apex up.
    # Truncated: a true apex is unprintable and tessellates to degenerate
    # triangles. The tip contacts the wall well above it, so the flat is free.
    part -= Cone(
        bottom_radius=params.seat_diameter_mm / 2.0,
        top_radius=params.seat_apex_diameter_mm / 2.0,
        height=params.seat_depth_mm,
        align=bottom,
    )
    # Anti-slip pad recess in the top face.
    part -= Pos(0, 0, params.height_mm - params.pad_recess_depth_mm) * Cylinder(
        radius=params.pad_diameter_mm / 2.0,
        height=params.pad_recess_depth_mm,
        align=bottom,
    )
    return part


def export_pressure_foot(
    output_path: Optional[Path] = None,
    params: Optional[PressureFootParameters] = None,
) -> Path:
    """Build the pressure foot and write it to an STL."""
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_pressure_foot(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the pressure foot STL from src/geometry.py."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = PressureFootParameters.from_geometry()
    except PressureFootDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    if args.report:
        return 0

    written = export_pressure_foot(args.output, params)
    print(f"Wrote {written}  ({written.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
