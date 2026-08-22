"""
Desk clamp lower jaw -- the part that goes under the desk and holds the nut.

Run directly to write ``cad/output/desk_clamp_lower_jaw.stl``::

    python3 -m cad.desk_clamp_lower_jaw
    python3 -m cad.desk_clamp_lower_jaw --report

What the part does
------------------
A flat plate that sits under the desk, opposite the pedestal's upper jaw.
Bottom to top along +Z:

1. **Hex nut pocket** in the underside, holding an M8 nut captive so the
   assembly can be tightened one-handed from the knob above. The pocket is
   cut from ``DeskClampSpec.nut_thickness_max_mm`` rather than the nominal
   figure -- see that class's note on DIN 934 versus DIN EN ISO 4032.
2. **Structural web**, ``jaw_thickness_mm`` of solid material carrying the
   clamp load between the nut's bearing face and the pad.
3. **Pad recess** in the top surface, for a glued-in anti-slip rubber pad
   that bears on the desk's underside.

A single M8 clearance hole runs the full height on the bolt axis.

Coordinate frame
----------------
The origin is on the **clamp screw axis** at the plate's underside, with the
plate extending in **-X** (inboard, under the desk) to mirror the pedestal's
upper jaw, which extends in +X from the yaw axis. In the assembled clamp the
two bolt axes are collinear.

Why the pad sits where it does
------------------------------
The desk edge may fall anywhere in the pedestal's positioning window, so the
lower pad starts at the *widest* of those offsets from the bolt. That
guarantees the pad is under solid desk no matter where in the window the user
sites the arm, rather than only when they place it perfectly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from build123d import Align, Box, Cylinder, Part, Pos, export_stl

from cad._design import DesignRuleError, DesignStatus
from cad._primitives import hex_prism
from src.geometry import DEFAULT_HARDWARE, HardwareSpec

__all__ = [
    "LowerJawDesignError",
    "LowerJawParameters",
    "build_lower_jaw",
    "export_lower_jaw",
    "DEFAULT_STL_PATH",
]

DEFAULT_STL_PATH = (
    Path(__file__).resolve().parent / "output" / "desk_clamp_lower_jaw.stl"
)


class LowerJawDesignError(DesignRuleError):
    """Raised when a parameter set would produce an unbuildable lower jaw."""


@dataclass(frozen=True)
class LowerJawParameters:
    """Fully resolved lower-jaw dimensions, in millimetres."""

    total_thickness_mm: float
    plate_width_mm: float
    inboard_length_mm: float
    outboard_length_mm: float

    pad_gap_from_bolt_mm: float
    pad_length_mm: float
    pad_width_mm: float
    pad_recess_depth_mm: float

    bolt_hole_diameter_mm: float
    nut_pocket_across_flats_mm: float
    nut_pocket_depth_mm: float

    min_wall_thickness_mm: float
    structural_thickness_mm: float

    @classmethod
    def from_geometry(
        cls,
        hardware: Optional[HardwareSpec] = None,
        *,
        desk_edge_window_mm: float = 10.0,
    ) -> "LowerJawParameters":
        """
        Derive every dimension from the hardware singleton.

        Parameters
        ----------
        hardware:
            Source of truth. Defaults to ``DEFAULT_HARDWARE``.
        desk_edge_window_mm:
            Must match the value used for the pedestal's upper jaw; it sets
            how far the pad has to stand back from the bolt to stay under the
            desk at every legal edge position.

        Raises
        ------
        LowerJawDesignError
            If the resulting design violates a clearance.
        """
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        clamp = hardware.desk_clamp
        clearance = hardware.print_clearance_mm
        wall = hardware.min_wall_thickness_mm

        if desk_edge_window_mm <= 0.0:
            raise LowerJawDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"desk_edge_window_mm must be positive, got {desk_edge_window_mm}.",
            )

        bolt_hole = clamp.bolt_clearance_hole_diameter_mm
        pad_length, pad_width = clamp.lower_jaw_contact_mm

        # Widest offset the desk edge can take from the bolt axis; standing the
        # pad back this far keeps it under solid desk at every legal position.
        pad_gap = desk_edge_window_mm + bolt_hole / 2.0

        nut_pocket_af = clamp.nut_across_flats_mm + 2.0 * clearance
        nut_pocket_across_corners = nut_pocket_af * 2.0 / 3.0**0.5

        # The outboard tail must wrap the hex pocket's corners, not just the
        # round bolt hole, so it is sized from across-corners.
        outboard_length = nut_pocket_across_corners / 2.0 + wall
        inboard_length = pad_gap + pad_length + wall
        plate_width = max(pad_width, nut_pocket_across_corners + 2.0 * wall)

        params = cls(
            total_thickness_mm=clamp.lower_jaw_total_thickness_mm,
            plate_width_mm=plate_width,
            inboard_length_mm=inboard_length,
            outboard_length_mm=outboard_length,
            pad_gap_from_bolt_mm=pad_gap,
            pad_length_mm=pad_length,
            pad_width_mm=pad_width,
            pad_recess_depth_mm=clamp.pad_recess_depth_mm,
            bolt_hole_diameter_mm=bolt_hole,
            nut_pocket_across_flats_mm=nut_pocket_af,
            nut_pocket_depth_mm=clamp.nut_pocket_depth_mm,
            min_wall_thickness_mm=wall,
            structural_thickness_mm=clamp.jaw_thickness_mm,
        )
        params.validate()
        return params

    # ---- Derived accessors ------------------------------------------------

    @property
    def plate_length_mm(self) -> float:
        """Total length, outboard tail plus inboard reach."""
        return self.inboard_length_mm + self.outboard_length_mm

    @property
    def nut_pocket_across_corners_mm(self) -> float:
        """Hexagon across-corners for the nut pocket."""
        return self.nut_pocket_across_flats_mm * 2.0 / 3.0**0.5

    @property
    def nut_bearing_thickness_mm(self) -> float:
        """
        Material between the nut pocket's ceiling and the pad recess floor.

        This is the web that actually carries the clamp load into the plate,
        so it must not fall below the structural thickness.
        """
        return (
            self.total_thickness_mm
            - self.nut_pocket_depth_mm
            - self.pad_recess_depth_mm
        )

    # ---- Validation -------------------------------------------------------

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unprintable part.

        Raises
        ------
        LowerJawDesignError
            Naming the first violated constraint.
        """
        if self.total_thickness_mm <= 0.0:
            raise LowerJawDesignError(
                DesignStatus.NEGATIVE_HEIGHT,
                f"total_thickness_mm must be positive, got "
                f"{self.total_thickness_mm}.",
            )
        if self.nut_pocket_depth_mm + self.pad_recess_depth_mm >= self.total_thickness_mm:
            raise LowerJawDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Nut pocket ({self.nut_pocket_depth_mm:.2f} mm) and pad recess "
                f"({self.pad_recess_depth_mm:.2f} mm) together meet or exceed "
                f"the plate thickness ({self.total_thickness_mm:.2f} mm); they "
                "would break into each other.",
            )
        if self.nut_bearing_thickness_mm < self.structural_thickness_mm:
            raise LowerJawDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {self.nut_bearing_thickness_mm:.2f} mm of web between "
                f"the nut pocket and the pad recess; the structural thickness "
                f"is {self.structural_thickness_mm:.2f} mm. This web carries "
                "the whole clamp load.",
            )
        if self.bolt_hole_diameter_mm >= self.nut_pocket_across_flats_mm:
            raise LowerJawDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Bolt hole ({self.bolt_hole_diameter_mm:.2f} mm) is not "
                f"smaller than the nut pocket across flats "
                f"({self.nut_pocket_across_flats_mm:.2f} mm), so the nut would "
                "have no shoulder to bear against.",
            )
        pocket_side_wall = (
            self.plate_width_mm / 2.0 - self.nut_pocket_across_corners_mm / 2.0
        )
        if pocket_side_wall < self.min_wall_thickness_mm:
            raise LowerJawDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {pocket_side_wall:.2f} mm between the nut pocket's "
                f"corners and the plate edge; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )
        pocket_tail_wall = (
            self.outboard_length_mm - self.nut_pocket_across_corners_mm / 2.0
        )
        if pocket_tail_wall < self.min_wall_thickness_mm:
            raise LowerJawDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {pocket_tail_wall:.2f} mm of plate beyond the nut "
                f"pocket's outboard corner; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )
        if self.pad_gap_from_bolt_mm <= self.nut_pocket_across_corners_mm / 2.0:
            raise LowerJawDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Pad recess starts {self.pad_gap_from_bolt_mm:.2f} mm from the "
                f"bolt axis, inside the nut pocket's "
                f"{self.nut_pocket_across_corners_mm / 2.0:.2f} mm corner radius.",
            )
        if self.pad_width_mm > self.plate_width_mm:
            raise LowerJawDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"Pad width ({self.pad_width_mm:.2f} mm) exceeds the plate "
                f"width ({self.plate_width_mm:.2f} mm).",
            )
        pad_end_wall = self.inboard_length_mm - (
            self.pad_gap_from_bolt_mm + self.pad_length_mm
        )
        if pad_end_wall < self.min_wall_thickness_mm:
            raise LowerJawDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {pad_end_wall:.2f} mm of plate beyond the pad recess's "
                f"inboard edge; {self.min_wall_thickness_mm:.2f} mm required.",
            )
        return DesignStatus.OK

    # ---- Reporting --------------------------------------------------------

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        return (
            f"Desk clamp lower jaw parameters\n"
            f"-------------------------------\n"
            f"  Plate                  : {self.plate_length_mm:.2f} x "
            f"{self.plate_width_mm:.2f} x {self.total_thickness_mm:.2f} mm\n"
            f"    inboard of bolt      : {self.inboard_length_mm:.2f} mm\n"
            f"    outboard of bolt     : {self.outboard_length_mm:.2f} mm\n"
            f"\n"
            f"  Pad recess (top)       : {self.pad_length_mm:.1f} x "
            f"{self.pad_width_mm:.1f} x {self.pad_recess_depth_mm:.1f} mm deep, "
            f"starting {self.pad_gap_from_bolt_mm:.2f} mm from the bolt\n"
            f"  Bolt hole              : {self.bolt_hole_diameter_mm:.2f} mm dia, "
            f"full height\n"
            f"  Nut pocket (underside) : {self.nut_pocket_across_flats_mm:.2f} mm "
            f"across flats "
            f"({self.nut_pocket_across_corners_mm:.2f} across corners) x "
            f"{self.nut_pocket_depth_mm:.2f} mm deep\n"
            f"\n"
            f"  Load-bearing web       : {self.nut_bearing_thickness_mm:.2f} mm "
            f"(structural minimum {self.structural_thickness_mm:.2f} mm)\n"
        )


def build_lower_jaw(params: Optional[LowerJawParameters] = None) -> Part:
    """
    Construct the lower jaw solid.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the clamp screw axis at the plate's
        underside; the plate extends in -X.

    Raises
    ------
    LowerJawDesignError
        If ``params`` fails :meth:`LowerJawParameters.validate`.
    """
    params = LowerJawParameters.from_geometry() if params is None else params
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Plate stock, centred so the bolt axis lands at x = 0 ------------
    plate_centre_x = (params.outboard_length_mm - params.inboard_length_mm) / 2.0
    part = Pos(plate_centre_x, 0, 0) * Box(
        params.plate_length_mm,
        params.plate_width_mm,
        params.total_thickness_mm,
        align=bottom,
    )

    # ---- Captive hex nut pocket, opening downward ------------------------
    part -= hex_prism(
        params.nut_pocket_across_flats_mm, params.nut_pocket_depth_mm
    )

    # ---- Clamp screw clearance hole, full height -------------------------
    part -= Cylinder(
        radius=params.bolt_hole_diameter_mm / 2.0,
        height=params.total_thickness_mm,
        align=bottom,
    )

    # ---- Anti-slip pad recess in the top surface -------------------------
    pad_centre_x = -(params.pad_gap_from_bolt_mm + params.pad_length_mm / 2.0)
    part -= Pos(
        pad_centre_x,
        0,
        params.total_thickness_mm - params.pad_recess_depth_mm,
    ) * Box(
        params.pad_length_mm,
        params.pad_width_mm,
        params.pad_recess_depth_mm,
        align=bottom,
    )

    return part


def export_lower_jaw(
    output_path: Optional[Path] = None,
    params: Optional[LowerJawParameters] = None,
) -> Path:
    """Build the lower jaw and write it to an STL, creating parent directories."""
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_lower_jaw(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the desk clamp lower jaw STL from src/geometry.py."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = LowerJawParameters.from_geometry()
    except LowerJawDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    if args.report:
        return 0

    written = export_lower_jaw(args.output, params)
    print(f"Wrote {written}  ({written.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
