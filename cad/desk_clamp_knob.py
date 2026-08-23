"""
Desk clamp knob -- the hand grip that drives the M8 clamping screw.

Run directly to write ``cad/output/desk_clamp_knob.stl``::

    python3 -m cad.desk_clamp_knob
    python3 -m cad.desk_clamp_knob --report

What the part does
------------------
A fluted disc pressed onto the head of the clamp screw so the whole assembly
can be tightened by hand, with no tool. Bottom to top along +Z:

1. **Knob body** -- the disc the hand grips, with ``knob_flute_count``
   scallops cut around its perimeter.
2. **Hex socket** in the top face, taking the screw's hex head so the knob and
   screw turn together. It is sunk ``knob_head_recess_mm`` below the top face,
   which both protects the head and shortens the screw length the clamp stack
   needs.

A clearance hole runs from the socket floor through to the underside for the
screw's shank.

The knob touches nothing but the screw
--------------------------------------
Session D.1b gave this part a small bearing boss, because there the knob bore
against the clamp's upper jaw and confining that rubbing contact to a small
radius kept hand torque useful. The D.1c U-clamp has no such contact: the
knob hangs free below the bottom arm -- the load path runs screw, captive nut,
nut pocket -- and a knob that reached the arm would jam against it instead of
loading the desk. The boss was doing nothing, and Session D.1d removed it.

Print and fit notes
-------------------
Print **socket-face down**: the hex socket then needs no support, and its
first layers -- the surfaces that actually grip the head -- come out crisp.

The socket is modelled at the head's across-flats plus
``press_fit_clearance_mm``. FDM parts print internal features slightly
undersize, so this typically ends up a snug press fit in practice. If a test
print comes out loose, a drop of epoxy in the socket is sufficient: the knob
never carries more than a few newton-metres.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from build123d import Align, Cylinder, Part, Pos, export_stl

from cad._design import DesignRuleError, DesignStatus
from cad._primitives import hex_prism
from src.geometry import DEFAULT_HARDWARE, HardwareSpec

__all__ = [
    "KnobDesignError",
    "KnobParameters",
    "build_knob",
    "export_knob",
    "DEFAULT_STL_PATH",
]

DEFAULT_STL_PATH = Path(__file__).resolve().parent / "output" / "desk_clamp_knob.stl"


class KnobDesignError(DesignRuleError):
    """Raised when a parameter set would produce an unbuildable knob."""


@dataclass(frozen=True)
class KnobParameters:
    """Fully resolved knob dimensions, in millimetres."""

    body_diameter_mm: float
    body_thickness_mm: float

    socket_across_flats_mm: float
    socket_depth_mm: float
    bolt_hole_diameter_mm: float

    flute_count: int
    flute_radius_mm: float

    min_wall_thickness_mm: float

    @classmethod
    def from_geometry(
        cls,
        hardware: Optional[HardwareSpec] = None,
        *,
        press_fit_clearance_mm: float = 0.10,
        flute_depth_fraction: float = 0.16,
    ) -> "KnobParameters":
        """
        Derive every dimension from the hardware singleton.

        Parameters
        ----------
        hardware:
            Source of truth. Defaults to ``DEFAULT_HARDWARE``.
        press_fit_clearance_mm:
            Added to the head's across-flats for the socket. Deliberately much
            tighter than ``print_clearance_mm``: this joint must not slip.
        flute_depth_fraction:
            Scallop depth as a fraction of the knob radius. Sets how much bite
            the grip has.

        Raises
        ------
        KnobDesignError
            If the resulting design violates a clearance.
        """
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        clamp = hardware.desk_clamp

        if press_fit_clearance_mm < 0.0:
            raise KnobDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"press_fit_clearance_mm must be non-negative, got "
                f"{press_fit_clearance_mm}.",
            )
        if not 0.0 < flute_depth_fraction < 0.5:
            raise KnobDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"flute_depth_fraction must be in (0, 0.5), got "
                f"{flute_depth_fraction}.",
            )

        params = cls(
            body_diameter_mm=clamp.knob_diameter_mm,
            body_thickness_mm=clamp.knob_thickness_mm,
            socket_across_flats_mm=(
                clamp.bolt_head_across_flats_mm + press_fit_clearance_mm
            ),
            socket_depth_mm=clamp.knob_socket_depth_mm,
            bolt_hole_diameter_mm=clamp.bolt_clearance_hole_diameter_mm,
            flute_count=clamp.knob_flute_count,
            flute_radius_mm=(
                clamp.knob_diameter_mm / 2.0 * flute_depth_fraction
            ),
            min_wall_thickness_mm=hardware.min_wall_thickness_mm,
        )
        params.validate()
        return params

    # ---- Derived accessors ------------------------------------------------

    @property
    def total_height_mm(self) -> float:
        """
        The knob's full printed height.

        Equal to the body thickness since Session D.1d removed the bearing
        boss. Kept as a distinct property because the clamp stack and the
        assembly preview both measure against it.
        """
        return self.body_thickness_mm

    @property
    def socket_across_corners_mm(self) -> float:
        """Hexagon across-corners for the head socket."""
        return self.socket_across_flats_mm * 2.0 / math.sqrt(3.0)

    @property
    def shank_bore_length_mm(self) -> float:
        """Length of the plain bore between the socket floor and the boss face."""
        return self.total_height_mm - self.socket_depth_mm

    @property
    def grip_min_radius_mm(self) -> float:
        """Radius at the deepest point of a flute."""
        return self.body_diameter_mm / 2.0 - self.flute_radius_mm

    @property
    def flute_positions(self) -> Tuple[Tuple[float, float], ...]:
        """(x, y) centres of the perimeter scallops."""
        radius = self.body_diameter_mm / 2.0
        step = 360.0 / self.flute_count
        return tuple(
            (
                radius * math.cos(math.radians(i * step)),
                radius * math.sin(math.radians(i * step)),
            )
            for i in range(self.flute_count)
        )

    # ---- Validation -------------------------------------------------------

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unprintable part.

        Raises
        ------
        KnobDesignError
            Naming the first violated constraint.
        """
        for name, value in (
            ("body_diameter_mm", self.body_diameter_mm),
            ("body_thickness_mm", self.body_thickness_mm),
            ("socket_depth_mm", self.socket_depth_mm),
            ("flute_radius_mm", self.flute_radius_mm),
        ):
            if value <= 0.0:
                raise KnobDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )
        if self.flute_count < 3:
            raise KnobDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"flute_count must be at least 3, got {self.flute_count}.",
            )
        if self.socket_depth_mm >= self.total_height_mm:
            raise KnobDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Hex socket ({self.socket_depth_mm:.2f} mm deep) reaches "
                f"through the knob's full height "
                f"({self.total_height_mm:.2f} mm); there would be no floor for "
                "the head to bear on.",
            )
        if self.bolt_hole_diameter_mm >= self.socket_across_flats_mm:
            raise KnobDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Bolt bore ({self.bolt_hole_diameter_mm:.2f} mm) is not "
                f"smaller than the hex socket across flats "
                f"({self.socket_across_flats_mm:.2f} mm), so the head would "
                "have no shoulder to bear on.",
            )
        socket_wall = (
            self.grip_min_radius_mm - self.socket_across_corners_mm / 2.0
        )
        if socket_wall < self.min_wall_thickness_mm:
            raise KnobDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {socket_wall:.2f} mm between the hex socket's corners "
                f"and the deepest flute; "
                f"{self.min_wall_thickness_mm:.2f} mm required.",
            )
        if self.grip_min_radius_mm <= self.bolt_hole_diameter_mm / 2.0:
            raise KnobDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"Flutes cut to r = {self.grip_min_radius_mm:.2f} mm, inside "
                f"the shank bore ({self.bolt_hole_diameter_mm / 2.0:.2f} mm).",
            )
        return DesignStatus.OK

    # ---- Reporting --------------------------------------------------------

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        return (
            f"Desk clamp knob parameters\n"
            f"--------------------------\n"
            f"  Body                   : {self.body_diameter_mm:.2f} mm dia x "
            f"{self.body_thickness_mm:.2f} mm\n"
            f"  Total printed height   : {self.total_height_mm:.2f} mm\n"
            f"\n"
            f"  Hex socket (top)       : {self.socket_across_flats_mm:.2f} mm "
            f"across flats "
            f"({self.socket_across_corners_mm:.2f} across corners) x "
            f"{self.socket_depth_mm:.2f} mm deep\n"
            f"  Shank bore             : {self.bolt_hole_diameter_mm:.2f} mm dia "
            f"x {self.shank_bore_length_mm:.2f} mm\n"
            f"\n"
            f"  Grip flutes            : {self.flute_count} x "
            f"{self.flute_radius_mm:.2f} mm radius, cutting to r = "
            f"{self.grip_min_radius_mm:.2f} mm\n"
        )


def build_knob(params: Optional[KnobParameters] = None) -> Part:
    """
    Construct the knob solid.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the clamp screw axis at the knob's
        underside, +Z toward its top.

    Raises
    ------
    KnobDesignError
        If ``params`` fails :meth:`KnobParameters.validate`.
    """
    params = KnobParameters.from_geometry() if params is None else params
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Knob body -------------------------------------------------------
    part = Cylinder(
        radius=params.body_diameter_mm / 2.0,
        height=params.body_thickness_mm,
        align=bottom,
    )

    # ---- Grip flutes around the perimeter --------------------------------
    # Centred on the rim so each cylinder bites a half-round scallop out of it.
    for flute_x, flute_y in params.flute_positions:
        part -= Pos(flute_x, flute_y, 0) * Cylinder(
            radius=params.flute_radius_mm,
            height=params.body_thickness_mm,
            align=bottom,
        )

    # ---- Hex socket sunk into the top face -------------------------------
    part -= Pos(
        0, 0, params.total_height_mm - params.socket_depth_mm
    ) * hex_prism(params.socket_across_flats_mm, params.socket_depth_mm)

    # ---- Shank bore from the socket floor to the underside ---------------
    part -= Cylinder(
        radius=params.bolt_hole_diameter_mm / 2.0,
        height=params.shank_bore_length_mm,
        align=bottom,
    )

    return part


def export_knob(
    output_path: Optional[Path] = None,
    params: Optional[KnobParameters] = None,
) -> Path:
    """Build the knob and write it to an STL, creating parent directories."""
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_knob(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the desk clamp knob STL from src/geometry.py."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = KnobParameters.from_geometry()
    except KnobDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    if args.report:
        return 0

    written = export_knob(args.output, params)
    print(f"Wrote {written}  ({written.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
