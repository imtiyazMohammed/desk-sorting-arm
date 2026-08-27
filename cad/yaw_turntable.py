"""
Yaw turntable -- the plate that turns on the yaw bearing and carries the arm.

Run directly to write ``cad/output/yaw_turntable.stl``::

    python3 -m cad.yaw_turntable
    python3 -m cad.yaw_turntable --report      # dimensions only, no export

What the part does
------------------
It is the first moving part of the arm. The base yaw servo, buried in the
pedestal's turret, turns it through a bought aluminium horn; it rides the
6806ZZ thrust bearing pressed into the turret's top face; and it presents a
bolt circle that the shoulder bracket stands on. Everything above the desk
clamp rotates with this plate.

Section, from the bottom up::

        z = +6   +---------------------------------+   bracket bolt circle
                 |                                 |
        z =  0   +--+   +-----------------+   +----+   land on the inner ring
                    |   |    relief       |   |        (bearing top face)
        z = -1      | +-+-----------------+-+ |
                    | |   horn pocket ceiling | |
                    | |                       | |
        z = -7      +-+-----------------------+-+      spigot in the bore
                          ^ horn sits in here

Why the underside is stepped rather than flat
---------------------------------------------
The bearing stands ``BearingSpec.proud_mm`` -- half a millimetre -- above the
turret. A plate with a flat underside would land on the printed turret face
instead of on the bearing, and a plain recess deeper than 0.5 mm would do the
same thing while also clearing the inner ring. So the underside is relieved
over the **outer** ring only, and lands on the inner ring inside that relief.
That is the difference between a bearing and a spacer.

Why there is a spigot at all
----------------------------
Two jobs. It fills the bearing's bore so the plate is located radially by a
steel ring rather than by four M3 screws, and it houses the servo horn: the
horn sits inside it, bearing on its cap, with the screws pulling up into the
plate. The spigot is the reason the yaw bearing had to grow from a 608 to a
6806 in Session D.2a -- an 8 mm bore has room for neither.

Coordinate frame
----------------
Origin on the yaw axis at the **land plane**: the face that touches the
bearing's inner ring. In the pedestal's frame that is
``PedestalParameters.bearing_top_z_mm``, so the plate occupies desk-frame
z = 70.0 to 76.0 and the spigot hangs into the bearing below it. ``+X`` is the
arm-forward direction at yaw zero.
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
from src.geometry import DEFAULT_HARDWARE, HardwareSpec

__all__ = [
    "TurntableDesignError",
    "TurntableParameters",
    "build_turntable",
    "export_turntable",
    "DEFAULT_STL_PATH",
]

DEFAULT_STL_PATH = Path(__file__).resolve().parent / "output" / "yaw_turntable.stl"


class TurntableDesignError(DesignRuleError):
    """Raised when a parameter set would produce an unbuildable turntable."""


@dataclass(frozen=True)
class TurntableParameters:
    """
    Fully resolved turntable dimensions, in millimetres.

    Build these with :meth:`from_geometry`; the direct constructor exists so
    tests can inject broken values and confirm :meth:`validate` catches them.
    """

    # ---- Plate --------------------------------------------------------------
    diameter_mm: float
    thickness_mm: float

    # ---- Bearing interface (below the land plane, z < 0) --------------------
    spigot_diameter_mm: float
    spigot_depth_mm: float
    race_relief_inner_diameter_mm: float
    race_relief_outer_diameter_mm: float
    race_relief_depth_mm: float
    bearing_proud_mm: float
    inner_race_outer_diameter_mm: float

    # ---- Servo horn pocket, inside the spigot -------------------------------
    horn_pocket_diameter_mm: float
    horn_pocket_depth_mm: float
    horn_bolt_circle_mm: float
    horn_bolt_count: int
    horn_bolt_clearance_diameter_mm: float
    horn_counterbore_diameter_mm: float
    horn_counterbore_depth_mm: float

    # ---- Shoulder bracket interface -----------------------------------------
    bracket_bolt_pattern_mm: Tuple[float, float]
    bracket_bolt_pilot_diameter_mm: float
    bracket_bolt_depth_mm: float

    # ---- Yaw-zero witness ---------------------------------------------------
    index_notch_width_mm: float
    index_notch_depth_mm: float
    index_notch_length_mm: float

    # ---- Carried through for validation -------------------------------------
    min_wall_thickness_mm: float

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_geometry(
        cls, hardware: Optional[HardwareSpec] = None
    ) -> "TurntableParameters":
        """
        Derive every dimension from the hardware singleton.

        Raises
        ------
        TurntableDesignError
            If the resulting design violates a clearance (see :meth:`validate`).
        """
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        spec = hardware.yaw_turntable
        bearing = hardware.thrust_bearing
        horn = hardware.servo_horn
        clearance = hardware.print_clearance_mm

        # The spigot fills the bore, less a slip fit so it goes in by hand.
        spigot_diameter = bearing.bore_diameter_mm - spec.spigot_bore_fit_mm
        # It reaches the full width of the bearing, and the horn takes up all
        # but the last millimetre of that. What is left is the cap the horn
        # bears on -- see validate(), which refuses a horn taller than the
        # bearing is wide.
        horn_pocket_depth = horn.total_height_mm

        params = cls(
            diameter_mm=spec.diameter_mm,
            thickness_mm=spec.thickness_mm,
            spigot_diameter_mm=spigot_diameter,
            spigot_depth_mm=bearing.width_mm,
            race_relief_inner_diameter_mm=bearing.inner_race_outer_diameter_mm,
            race_relief_outer_diameter_mm=(
                bearing.outer_diameter_mm + 2.0 * clearance
            ),
            race_relief_depth_mm=spec.bearing_race_recess_mm,
            bearing_proud_mm=bearing.proud_mm,
            inner_race_outer_diameter_mm=bearing.inner_race_outer_diameter_mm,
            horn_pocket_diameter_mm=horn.disc_diameter_mm + 2.0 * clearance,
            horn_pocket_depth_mm=horn_pocket_depth,
            horn_bolt_circle_mm=horn.bolt_circle_mm,
            horn_bolt_count=horn.bolt_count,
            horn_bolt_clearance_diameter_mm=horn.bolt_clearance_diameter_mm,
            horn_counterbore_diameter_mm=(
                horn.bolt_head_diameter_mm + 2.0 * clearance
            ),
            horn_counterbore_depth_mm=horn.bolt_head_height_mm + 0.5,
            bracket_bolt_pattern_mm=spec.bracket_bolt_pattern_mm,
            bracket_bolt_pilot_diameter_mm=spec.bracket_bolt_pilot_diameter_mm,
            bracket_bolt_depth_mm=spec.bracket_bolt_depth_mm,
            index_notch_width_mm=spec.index_notch_width_mm,
            index_notch_depth_mm=spec.index_notch_depth_mm,
            index_notch_length_mm=spec.index_notch_length_mm,
            min_wall_thickness_mm=hardware.min_wall_thickness_mm,
        )
        params.validate()
        return params

    # =====================================================================
    # Derived accessors
    # =====================================================================

    @property
    def spigot_cap_mm(self) -> float:
        """
        Material between the horn pocket's ceiling and the land plane, in mm.

        The face the horn's disc bears against, and the whole load path from
        the servo into the plate. Equal to the bearing's width less the horn's
        height, so a taller horn eats it.
        """
        return self.spigot_depth_mm - self.horn_pocket_depth_mm

    @property
    def spigot_ring_wall_mm(self) -> float:
        """Wall between the horn pocket and the spigot's outer surface."""
        return (self.spigot_diameter_mm - self.horn_pocket_diameter_mm) / 2.0

    @property
    def race_land_width_mm(self) -> float:
        """
        Radial width of the annulus that actually touches the inner ring, in mm.

        From the spigot's outer surface out to where the relief starts.
        """
        return (
            self.race_relief_inner_diameter_mm - self.spigot_diameter_mm
        ) / 2.0

    @property
    def bracket_bolt_radius_mm(self) -> float:
        """Distance from the yaw axis to a bracket bolt, in mm."""
        x_span, y_span = self.bracket_bolt_pattern_mm
        return float(math.hypot(x_span / 2.0, y_span / 2.0))

    @property
    def rim_outside_bracket_bolts_mm(self) -> float:
        """Material between a bracket bolt hole and the plate's edge."""
        return (
            self.diameter_mm / 2.0
            - self.bracket_bolt_radius_mm
            - self.bracket_bolt_pilot_diameter_mm / 2.0
        )

    @property
    def bracket_bolt_clearance_to_bearing_mm(self) -> float:
        """Gap between a bracket bolt hole and the bearing relief below it."""
        return (
            self.bracket_bolt_radius_mm
            - self.bracket_bolt_pilot_diameter_mm / 2.0
            - self.race_relief_outer_diameter_mm / 2.0
        )

    @property
    def bracket_bolt_floor_mm(self) -> float:
        """Material left under a blind bracket bolt hole."""
        return self.thickness_mm - self.bracket_bolt_depth_mm

    def bolt_positions(
        self, circle_diameter_mm: float, count: int, phase_deg: float = 45.0
    ) -> Tuple[Tuple[float, float], ...]:
        """(x, y) centres of ``count`` holes on a circle, in mm."""
        radius = circle_diameter_mm / 2.0
        return tuple(
            (
                radius * math.cos(math.radians(phase_deg + i * 360.0 / count)),
                radius * math.sin(math.radians(phase_deg + i * 360.0 / count)),
            )
            for i in range(count)
        )

    @property
    def horn_bolt_positions(self) -> Tuple[Tuple[float, float], ...]:
        return self.bolt_positions(self.horn_bolt_circle_mm, self.horn_bolt_count)

    @property
    def bracket_bolt_positions(self) -> Tuple[Tuple[float, float], ...]:
        x_span, y_span = self.bracket_bolt_pattern_mm
        return tuple(
            (sx * x_span / 2.0, sy * y_span / 2.0)
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
        )

    # =====================================================================
    # Validation
    # =====================================================================

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unprintable part.

        Returns :attr:`DesignStatus.OK` on success.

        Raises
        ------
        TurntableDesignError
            Naming the first violated constraint.
        """
        for name, value in (
            ("diameter_mm", self.diameter_mm),
            ("thickness_mm", self.thickness_mm),
            ("spigot_diameter_mm", self.spigot_diameter_mm),
            ("spigot_depth_mm", self.spigot_depth_mm),
            ("horn_pocket_diameter_mm", self.horn_pocket_diameter_mm),
            ("horn_pocket_depth_mm", self.horn_pocket_depth_mm),
            ("race_relief_depth_mm", self.race_relief_depth_mm),
            ("bracket_bolt_depth_mm", self.bracket_bolt_depth_mm),
        ):
            if value <= 0.0:
                raise TurntableDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )
        if self.horn_bolt_count < 3:
            raise TurntableDesignError(
                DesignStatus.INVALID_PARAMETER,
                "The horn pattern needs at least 3 holes to locate the plate "
                f"rotationally, got {self.horn_bolt_count}.",
            )

        # ---- The relief has to clear a bearing that stands proud ----------
        if self.race_relief_depth_mm <= self.bearing_proud_mm:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The outer-race relief is {self.race_relief_depth_mm:.2f} mm "
                f"deep but the bearing stands {self.bearing_proud_mm:.2f} mm "
                "proud of the turret, so the plate would rest on the outer "
                "ring -- or, worse, on the printed turret face -- instead of "
                "on the inner ring.",
            )
        if self.race_relief_inner_diameter_mm <= self.spigot_diameter_mm:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The relief starts at "
                f"{self.race_relief_inner_diameter_mm:.2f} mm, inside the "
                f"spigot ({self.spigot_diameter_mm:.2f} mm): there would be no "
                "land left to bear on the inner ring.",
            )
        if self.race_relief_outer_diameter_mm >= self.diameter_mm:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The plate ({self.diameter_mm:.2f} mm) does not cover the "
                f"bearing it relieves "
                f"({self.race_relief_outer_diameter_mm:.2f} mm).",
            )

        # ---- The horn has to fit inside the spigot ------------------------
        if self.spigot_cap_mm <= 0.0:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The horn is {self.horn_pocket_depth_mm:.2f} mm tall and the "
                f"spigot only reaches {self.spigot_depth_mm:.2f} mm, so there "
                "is no cap for it to bear on and the plate would ride the horn "
                "rather than the bearing.",
            )
        if self.spigot_ring_wall_mm < self.min_wall_thickness_mm / 2.0:
            raise TurntableDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {self.spigot_ring_wall_mm:.2f} mm of ring between the "
                f"horn pocket and the bearing bore. Half the minimum wall is "
                f"allowed here because the ring is confined by the steel race "
                f"over its whole height -- the same allowance the pressure "
                f"foot's rim takes -- but "
                f"{self.min_wall_thickness_mm / 2.0:.2f} mm is the floor.",
            )
        if self.horn_bolt_circle_mm >= self.horn_pocket_diameter_mm:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The horn's bolt circle ({self.horn_bolt_circle_mm:.2f} mm) "
                f"is not inside its pocket "
                f"({self.horn_pocket_diameter_mm:.2f} mm).",
            )
        counterbore_edge = (
            self.horn_bolt_circle_mm / 2.0 + self.horn_counterbore_diameter_mm / 2.0
        )
        if counterbore_edge > self.horn_pocket_diameter_mm / 2.0:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"A horn-screw counterbore reaches "
                f"{counterbore_edge:.2f} mm from the axis and would break out "
                f"through the spigot's ring, which ends at "
                f"{self.horn_pocket_diameter_mm / 2.0:.2f} mm.",
            )
        if (
            self.horn_counterbore_depth_mm
            >= self.thickness_mm + self.spigot_cap_mm
        ):
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"A {self.horn_counterbore_depth_mm:.2f} mm counterbore would "
                f"break straight through into the horn pocket "
                f"({self.thickness_mm + self.spigot_cap_mm:.2f} mm of material "
                "above it).",
            )

        # ---- The bracket's bolts have to land in material -----------------
        if self.bracket_bolt_clearance_to_bearing_mm <= 0.0:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The bracket's bolt pattern "
                f"(radius {self.bracket_bolt_radius_mm:.2f} mm) lands over the "
                f"bearing relief "
                f"({self.race_relief_outer_diameter_mm:.2f} mm), where the "
                "plate is thinned and there is nothing to thread into. Move "
                "the pattern outboard.",
            )
        if self.rim_outside_bracket_bolts_mm < self.min_wall_thickness_mm:
            raise TurntableDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {self.rim_outside_bracket_bolts_mm:.2f} mm of rim "
                f"outside the bracket's bolt pattern.",
            )
        if self.bracket_bolt_floor_mm <= 0.0:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"A {self.bracket_bolt_depth_mm:.2f} mm blind hole in a "
                f"{self.thickness_mm:.2f} mm plate breaks through onto the "
                "bearing.",
            )

        # ---- The witness notch must not cut the plate in half -------------
        if self.index_notch_depth_mm >= self.thickness_mm:
            raise TurntableDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The index notch ({self.index_notch_depth_mm:.2f} mm) is as "
                f"deep as the plate ({self.thickness_mm:.2f} mm).",
            )
        return DesignStatus.OK

    # =====================================================================
    # Reporting
    # =====================================================================

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        return (
            f"Yaw turntable parameters\n"
            f"------------------------\n"
            f"  Plate                  : {self.diameter_mm:.2f} mm dia x "
            f"{self.thickness_mm:.2f} mm (z 0 .. {self.thickness_mm:.2f})\n"
            f"  Spigot in bearing bore : {self.spigot_diameter_mm:.2f} mm dia x "
            f"{self.spigot_depth_mm:.2f} mm (z "
            f"{-self.spigot_depth_mm:.2f} .. 0)\n"
            f"  Horn pocket            : {self.horn_pocket_diameter_mm:.2f} mm "
            f"dia x {self.horn_pocket_depth_mm:.2f} mm deep\n"
            f"    ring wall            : {self.spigot_ring_wall_mm:.2f} mm "
            f"(floor {self.min_wall_thickness_mm / 2.0:.2f}, confined by the "
            f"race)\n"
            f"    cap above the horn   : {self.spigot_cap_mm:.2f} mm\n"
            f"  Outer-race relief      : "
            f"{self.race_relief_inner_diameter_mm:.2f} .. "
            f"{self.race_relief_outer_diameter_mm:.2f} mm dia x "
            f"{self.race_relief_depth_mm:.2f} mm deep\n"
            f"  Land on the inner ring : {self.race_land_width_mm:.2f} mm wide "
            f"(bearing stands {self.bearing_proud_mm:.2f} mm proud)\n"
            f"\n"
            f"  Horn screws            : {self.horn_bolt_count} x "
            f"{self.horn_bolt_clearance_diameter_mm:.2f} mm on a "
            f"{self.horn_bolt_circle_mm:.2f} mm circle, counterbored "
            f"{self.horn_counterbore_diameter_mm:.2f} x "
            f"{self.horn_counterbore_depth_mm:.2f} mm\n"
            f"  Bracket screws         : 4 x "
            f"{self.bracket_bolt_pilot_diameter_mm:.2f} mm blind on a "
            f"{self.bracket_bolt_pattern_mm[0]:.0f} x "
            f"{self.bracket_bolt_pattern_mm[1]:.0f} mm rectangle, "
            f"{self.bracket_bolt_depth_mm:.2f} mm deep\n"
            f"    clear of the bearing : "
            f"{self.bracket_bolt_clearance_to_bearing_mm:.2f} mm\n"
            f"    rim outside them     : "
            f"{self.rim_outside_bracket_bolts_mm:.2f} mm\n"
            f"    floor beneath them   : {self.bracket_bolt_floor_mm:.2f} mm\n"
            f"  Yaw-zero witness notch : {self.index_notch_length_mm:.2f} x "
            f"{self.index_notch_width_mm:.2f} x "
            f"{self.index_notch_depth_mm:.2f} mm in the rim at +X\n"
        )


def build_turntable(params: Optional[TurntableParameters] = None) -> Part:
    """
    Construct the turntable solid.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the yaw axis at the land plane -- the face
        that touches the bearing's inner ring.

    Raises
    ------
    TurntableDesignError
        If ``params`` fails :meth:`TurntableParameters.validate`.
    """
    params = TurntableParameters.from_geometry() if params is None else params
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Plate, sitting on the land plane ---------------------------------
    part = Cylinder(
        radius=params.diameter_mm / 2.0,
        height=params.thickness_mm,
        align=bottom,
    )

    # ---- Spigot hanging into the bearing bore -----------------------------
    part += Pos(0, 0, -params.spigot_depth_mm) * Cylinder(
        radius=params.spigot_diameter_mm / 2.0,
        height=params.spigot_depth_mm,
        align=bottom,
    )

    # ---- Horn pocket inside the spigot, open downward ---------------------
    part -= Pos(0, 0, -params.spigot_depth_mm) * Cylinder(
        radius=params.horn_pocket_diameter_mm / 2.0,
        height=params.horn_pocket_depth_mm,
        align=bottom,
    )

    # ---- Relief over the bearing's outer ring -----------------------------
    # An annulus: cut the outer disc, put the inner one back, so the land on
    # the inner ring survives.
    relief = Cylinder(
        radius=params.race_relief_outer_diameter_mm / 2.0,
        height=params.race_relief_depth_mm,
        align=bottom,
    ) - Cylinder(
        radius=params.race_relief_inner_diameter_mm / 2.0,
        height=params.race_relief_depth_mm,
        align=bottom,
    )
    part -= relief

    # ---- Horn screws: clearance up through the cap, counterbored on top ---
    for screw_x, screw_y in params.horn_bolt_positions:
        part -= Pos(screw_x, screw_y, -params.spigot_cap_mm) * Cylinder(
            radius=params.horn_bolt_clearance_diameter_mm / 2.0,
            height=params.spigot_cap_mm + params.thickness_mm,
            align=bottom,
        )
        part -= Pos(
            screw_x,
            screw_y,
            params.thickness_mm - params.horn_counterbore_depth_mm,
        ) * Cylinder(
            radius=params.horn_counterbore_diameter_mm / 2.0,
            height=params.horn_counterbore_depth_mm,
            align=bottom,
        )

    # ---- Blind pilot holes for the shoulder bracket -----------------------
    for screw_x, screw_y in params.bracket_bolt_positions:
        part -= Pos(
            screw_x, screw_y, params.thickness_mm - params.bracket_bolt_depth_mm
        ) * Cylinder(
            radius=params.bracket_bolt_pilot_diameter_mm / 2.0,
            height=params.bracket_bolt_depth_mm,
            align=bottom,
        )

    # ---- Yaw-zero witness notch in the rim at +X --------------------------
    notch_x = params.diameter_mm / 2.0 - params.index_notch_length_mm / 2.0
    part -= Pos(
        notch_x, 0, params.thickness_mm - params.index_notch_depth_mm
    ) * Box(
        params.index_notch_length_mm,
        params.index_notch_width_mm,
        params.index_notch_depth_mm,
        align=bottom,
    )
    return part


def export_turntable(
    output_path: Optional[Path] = None,
    params: Optional[TurntableParameters] = None,
) -> Path:
    """
    Build the turntable and write it to an STL, creating parent directories.

    Raises
    ------
    TurntableDesignError
        If the parameters are unbuildable.
    RuntimeError
        If build123d reports the export failed.
    """
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_turntable(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the yaw turntable STL from src/geometry.py."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = TurntableParameters.from_geometry()
    except TurntableDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    if args.report:
        return 0

    written = export_turntable(args.output, params)
    print(f"Wrote {written}  ({written.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
