"""
Whole-assembly preview -- every part placed on a desk, in one scene.

Run directly to write both artifacts::

    python3 -m cad.assembly_preview
    python3 -m cad.assembly_preview --report        # summary only
    python3 -m cad.assembly_preview --desk-thickness 35

Outputs:

- ``cad/output/assembly_preview.stl`` -- the whole scene as one STL, viewable
  in any slicer, mesh tool, or GitHub's built-in 3D preview.
- ``cad/output/assembly_preview.png`` -- an orthographic side elevation beside
  an isometric view.

What this is for
----------------
Individual parts each validate against their own design rules, but nothing
checks that they *fit together on a real desk*. This scene places the clamp at
``ArmGeometry``'s stated base position, drops a desk slab into its throat, and
extends the arm's links at their zero pose. Collisions, a misplaced yaw axis,
or links passing through the desk show up immediately -- and
``tests/test_cad.py`` asserts the same properties numerically.

The arm links are deliberately **placeholders**: plain cylinders at the link
lengths from ``ArmGeometry``, not the real (undesigned) parts. They answer
"does the arm clear the mount?", not "what will the arm look like?".

Coordinate frame
----------------
The **desk frame** from ``docs/PROOF_OF_CONCEPT.md`` section 3: origin at a
desk corner, ``+X`` along the 1200 mm edge, ``+Y`` along the 600 mm edge into
the desk, ``+Z`` up, with the desk's **top surface at z = 0**.

The clamp is modelled in its own frame, with ``+X`` pointing inward over the
desk. Placing it means rotating that frame 90 degrees about Z -- clamp ``+X``
becomes desk ``+Y`` -- and translating to
``(base_x_on_desk_mm, base_y_on_desk_mm)``. Both frames put their origin on the
yaw axis at the desk surface, so no vertical offset is needed.

.. note::
   Rendering is done with matplotlib over the tessellated mesh. build123d has
   no headless raster renderer of its own -- its viewers (ocp_vscode and
   friends) need a live GUI session, which is no use in a script or in CI.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from build123d import (
    Align,
    Box,
    Compound,
    Cylinder,
    Part,
    Pos,
    Rot,
    Sphere,
    export_stl,
)

from cad.base_pedestal import PedestalParameters, build_pedestal
from cad.desk_clamp_knob import KnobParameters, build_knob
from cad.desk_clamp_pressure_foot import (
    PressureFootParameters,
    build_pressure_foot,
)
from src.geometry import DEFAULT_ARM, DEFAULT_HARDWARE, ArmGeometry, HardwareSpec

__all__ = [
    "AssemblyPart",
    "Assembly",
    "build_assembly",
    "export_assembly_stl",
    "render_assembly_png",
    "DEFAULT_STL_PATH",
    "DEFAULT_PNG_PATH",
]

_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_STL_PATH = _OUTPUT_DIR / "assembly_preview.stl"
DEFAULT_PNG_PATH = _OUTPUT_DIR / "assembly_preview.png"

#: Density of PETG, g/cm^3. Used for the print-mass estimate.
PETG_DENSITY_G_CM3 = 1.27

#: Infill assumed for the mass estimate. Printed parts are not solid, so a
#: solid-volume figure would overstate filament use by roughly three times.
ASSUMED_INFILL_FRACTION = 0.35

#: Placeholder link radii, in mm. Cosmetic only -- chosen to taper from
#: shoulder to wrist so the links read as an arm rather than a pipe.
LINK_RADII_MM = (14.0, 11.0, 8.0)

#: Radius of the marker sphere at the tool centre point, in mm.
TCP_MARKER_RADIUS_MM = 12.0

#: Shoulder bracket placeholder footprint, as a fraction of the turntable's
#: diameter. Both placeholders are sized off the turret they stand on rather
#: than given fixed numbers: they stand in for parts Sessions D.2 and D.3 have
#: yet to design, and inventing dimensions for them would imply decisions that
#: have not been made.
BRACKET_PLACEHOLDER_FRACTION = 0.75


@dataclass(frozen=True)
class AssemblyPart:
    """
    One placed solid in the scene.

    Attributes
    ----------
    name:
        Label used in the summary and the render legend.
    solid:
        The positioned :class:`build123d.Part`, already in desk coordinates.
    color:
        Matplotlib colour for the PNG render.
    printed:
        Whether this is a part we actually print. The desk and the arm
        placeholders are scenery and are excluded from the mass estimate.
    """

    name: str
    solid: Part
    color: str
    printed: bool = True

    @property
    def volume_cm3(self) -> float:
        return float(self.solid.volume) / 1000.0

    @property
    def bounding_box_mm(self) -> Tuple[float, float, float]:
        box = self.solid.bounding_box()
        return (
            float(box.max.X - box.min.X),
            float(box.max.Y - box.min.Y),
            float(box.max.Z - box.min.Z),
        )


@dataclass(frozen=True)
class Assembly:
    """A placed scene, plus the numbers that describe it."""

    parts: Tuple[AssemblyPart, ...]
    desk_thickness_mm: float
    yaw_axis_xy_mm: Tuple[float, float]
    servo_shaft_output_z_mm: float
    shoulder_pivot_z_mm: float
    clamp_footprint_mm: Tuple[float, float]
    knob_drop_below_arm_mm: float

    def by_name(self, name: str) -> AssemblyPart:
        """Look up one placed part, raising KeyError if it is not in the scene."""
        for part in self.parts:
            if part.name == name:
                return part
        raise KeyError(f"No part named {name!r} in the assembly.")

    @property
    def printed_parts(self) -> Tuple[AssemblyPart, ...]:
        return tuple(part for part in self.parts if part.printed)

    @property
    def printed_volume_cm3(self) -> float:
        return sum(part.volume_cm3 for part in self.printed_parts)

    @property
    def estimated_print_mass_g(self) -> float:
        """
        Filament mass at :data:`ASSUMED_INFILL_FRACTION`, in grams.

        A rough figure: real slicer output depends on perimeters, top and
        bottom layers and support, none of which this models.
        """
        return (
            self.printed_volume_cm3 * ASSUMED_INFILL_FRACTION * PETG_DENSITY_G_CM3
        )

    def to_compound(self) -> Compound:
        """Every placed solid as one shape, for STL export."""
        return Compound(children=[part.solid for part in self.parts])

    def summary(self) -> str:
        """Human-readable scene summary, printed by ``--report``."""
        lines = [
            "Assembly preview",
            "----------------",
            f"  Desk                   : {DEFAULT_ARM.desk_width_mm:.0f} x "
            f"{DEFAULT_ARM.desk_depth_mm:.0f} x {self.desk_thickness_mm:.0f} mm "
            "(top surface at z = 0)",
            f"  Yaw axis on desk       : "
            f"({self.yaw_axis_xy_mm[0]:.1f}, {self.yaw_axis_xy_mm[1]:.1f}) mm",
            f"  Clamp footprint        : {self.clamp_footprint_mm[0]:.1f} (X) x "
            f"{self.clamp_footprint_mm[1]:.1f} (Y) mm on the desk surface",
            f"  Servo shaft output     : z = {self.servo_shaft_output_z_mm:.1f} mm",
            f"  Shoulder pivot         : z = {self.shoulder_pivot_z_mm:.1f} mm",
            f"  Knob hangs below arm   : {self.knob_drop_below_arm_mm:.1f} mm",
            "",
            "  Parts",
        ]
        for part in self.parts:
            width, depth, height = part.bounding_box_mm
            tag = "print" if part.printed else "scene"
            lines.append(
                f"    {part.name:<22} {tag}  "
                f"{width:7.1f} x {depth:7.1f} x {height:7.1f} mm"
                + (f"  {part.volume_cm3:7.1f} cm3" if part.printed else "")
            )
        lines += [
            "",
            f"  Printed volume         : {self.printed_volume_cm3:.1f} cm3 solid",
            f"  Est. filament mass     : {self.estimated_print_mass_g:.0f} g PETG "
            f"at {ASSUMED_INFILL_FRACTION * 100:.0f}% infill",
        ]
        return "\n".join(lines) + "\n"


def build_assembly(
    arm: Optional[ArmGeometry] = None,
    hardware: Optional[HardwareSpec] = None,
    desk_thickness_mm: Optional[float] = None,
) -> Assembly:
    """
    Place every part into desk coordinates.

    Parameters
    ----------
    arm, hardware:
        Sources of truth. Default to ``DEFAULT_ARM`` / ``DEFAULT_HARDWARE``.
    desk_thickness_mm:
        Desk to model. Defaults to the middle of the clamp's supported range.

    Returns
    -------
    Assembly
        Every solid positioned, with the derived scene numbers.

    Raises
    ------
    ValueError
        If ``desk_thickness_mm`` falls outside the clamp's supported range,
        which would make the scene meaningless.
    """
    arm = DEFAULT_ARM if arm is None else arm
    hardware = DEFAULT_HARDWARE if hardware is None else hardware
    clamp = hardware.desk_clamp

    if desk_thickness_mm is None:
        desk_thickness_mm = clamp.nominal_desk_thickness_mm
    if not (
        clamp.min_desk_thickness_mm
        <= desk_thickness_mm
        <= clamp.max_desk_thickness_mm
    ):
        raise ValueError(
            f"desk_thickness_mm {desk_thickness_mm} is outside the clamp's "
            f"supported range {clamp.desk_thickness_range_mm}."
        )

    pedestal_params = PedestalParameters.from_geometry(arm, hardware)
    foot_params = PressureFootParameters.from_geometry(hardware)
    knob_params = KnobParameters.from_geometry(hardware)

    base_x, base_y = arm.base_x_on_desk_mm, arm.base_y_on_desk_mm
    bottom = (Align.CENTER, Align.CENTER, Align.MIN)

    # ---- Desk slab, top surface on z = 0 ---------------------------------
    desk = Pos(
        arm.desk_width_mm / 2.0, arm.desk_depth_mm / 2.0, -desk_thickness_mm
    ) * Box(arm.desk_width_mm, arm.desk_depth_mm, desk_thickness_mm, align=bottom)

    # ---- U-clamp: its +X (inward) becomes the desk's +Y -------------------
    clamp_solid = Pos(base_x, base_y, 0) * Rot(0, 0, 90) * build_pedestal(
        pedestal_params
    )

    # ---- Pressure foot: contact face against the desk's underside --------
    foot_top_z = -desk_thickness_mm
    foot_solid = Pos(
        base_x,
        base_y + pedestal_params.bolt_axis_x_mm,
        foot_top_z - foot_params.height_mm,
    ) * build_pressure_foot(foot_params)

    # ---- Clamping screw ---------------------------------------------------
    # The tip stops at the crown of the foot's bore; the head is one bolt
    # length below that, wherever the desk's thickness happens to put it.
    screw_tip_z = foot_top_z - foot_params.rise_above_tip_mm
    screw_head_z = screw_tip_z - clamp.bolt_length_mm
    screw_solid = Pos(
        base_x, base_y + pedestal_params.bolt_axis_x_mm, screw_head_z
    ) * Cylinder(
        radius=clamp.bolt_nominal_diameter_mm / 2.0,
        height=clamp.bolt_length_mm,
        align=bottom,
    )

    # ---- Knob, flipped so its bearing face points up at the arm ----------
    knob_bearing_z = screw_head_z + (
        knob_params.total_height_mm - knob_params.socket_depth_mm
    )
    knob_solid = (
        Pos(base_x, base_y + pedestal_params.bolt_axis_x_mm, knob_bearing_z)
        * Rot(180, 0, 0)
        * build_knob(knob_params)
    )
    knob_drop = pedestal_params.bottom_arm_bottom_z_mm - knob_bearing_z

    # ---- The base stack between the turret and the shoulder pivot --------
    # These two parts are designed in Sessions D.2 and D.3 and do not exist
    # yet, but they occupy real height: BaseStack budgets for them, and
    # pedestal_height_mm is what is left over. Drawing them as placeholders is
    # what stops the arm appearing to float above the turret -- the gap is
    # hardware, not an error in the base frame.
    turntable_diameter = (
        pedestal_params.turret_x_max_mm - pedestal_params.turret_x_min_mm
    )
    turntable_bottom_z = pedestal_params.bearing_top_z_mm
    turntable_solid = Pos(base_x, base_y, turntable_bottom_z) * Cylinder(
        radius=turntable_diameter / 2.0,
        height=hardware.base_stack.turntable_plate_thickness_mm,
        align=bottom,
    )
    bracket_side = turntable_diameter * BRACKET_PLACEHOLDER_FRACTION
    bracket_bottom_z = (
        turntable_bottom_z + hardware.base_stack.turntable_plate_thickness_mm
    )
    bracket_solid = Pos(base_x, base_y, bracket_bottom_z) * Box(
        bracket_side,
        bracket_side,
        hardware.base_stack.shoulder_bracket_rise_mm,
        align=bottom,
    )

    # ---- Arm placeholders at the zero pose --------------------------------
    # All joints at zero puts the arm horizontal along the base frame's +X,
    # which is the desk's +Y. Links are drawn end to end from the shoulder.
    shoulder_z = arm.base_height_mm
    link_lengths = (
        arm.l1_upper_arm_mm,
        arm.l2_forearm_mm,
        arm.l3_wrist_to_tip_mm,
    )
    link_names = ("L1 upper arm", "L2 forearm", "L3 wrist-to-TCP")
    link_colors = ("#d95f02", "#7570b3", "#1b9e77")

    links: List[AssemblyPart] = []
    reach = 0.0
    for length, radius, name, color in zip(
        link_lengths, LINK_RADII_MM, link_names, link_colors
    ):
        # Cylinder's axis is +Z, so lay it along +Y with a -90 deg X rotation.
        links.append(
            AssemblyPart(
                name=name,
                solid=Pos(base_x, base_y + reach, shoulder_z)
                * Rot(-90, 0, 0)
                * Cylinder(radius=radius, height=length, align=bottom),
                color=color,
                printed=False,
            )
        )
        reach += length

    tcp = AssemblyPart(
        name="TCP marker",
        solid=Pos(base_x, base_y + reach, shoulder_z)
        * Sphere(radius=TCP_MARKER_RADIUS_MM),
        color="#e7298a",
        printed=False,
    )

    clamp_box = clamp_solid.bounding_box()
    parts = (
        AssemblyPart("desk", desk, "#9bb7d4", printed=False),
        AssemblyPart("base_pedestal", clamp_solid, "#b5651d"),
        AssemblyPart("pressure_foot", foot_solid, "#8c8c8c"),
        AssemblyPart("clamp_screw", screw_solid, "#4d4d4d", printed=False),
        AssemblyPart("knob", knob_solid, "#c49a6c"),
        AssemblyPart(
            "yaw turntable (D.2)", turntable_solid, "#a0a0a0", printed=False
        ),
        AssemblyPart(
            "shoulder bracket (D.3)", bracket_solid, "#8f8f8f", printed=False
        ),
        *links,
        tcp,
    )
    return Assembly(
        parts=parts,
        desk_thickness_mm=float(desk_thickness_mm),
        yaw_axis_xy_mm=(float(base_x), float(base_y)),
        servo_shaft_output_z_mm=float(pedestal_params.servo_shaft_output_z_mm),
        shoulder_pivot_z_mm=float(arm.base_height_mm),
        clamp_footprint_mm=(
            float(clamp_box.max.X - clamp_box.min.X),
            float(clamp_box.max.Y - clamp_box.min.Y),
        ),
        knob_drop_below_arm_mm=float(knob_drop),
    )


def export_assembly_stl(
    output_path: Optional[Path] = None, assembly: Optional[Assembly] = None
) -> Path:
    """
    Write the whole scene to a single STL.

    Every solid stays a separate closed shell inside the file; the export does
    not fuse them, so the result is a scene rather than a manufacturable part.

    Raises
    ------
    RuntimeError
        If build123d reports the export failed.
    """
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    assembly = build_assembly() if assembly is None else assembly

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(assembly.to_compound(), output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def _triangles(solid: Part) -> np.ndarray:
    """Tessellate one solid into an (N, 3, 3) array of triangle vertices."""
    vertices, faces = solid.tessellate(tolerance=0.5, angular_tolerance=0.3)
    points = np.array([[v.X, v.Y, v.Z] for v in vertices], dtype=np.float64)
    return points[np.array(faces, dtype=np.int64)]


def render_assembly_png(
    output_path: Optional[Path] = None,
    assembly: Optional[Assembly] = None,
) -> Path:
    """
    Render the scene to a PNG: side elevation beside an isometric view.

    Uses matplotlib over the tessellated mesh; build123d's own viewers need a
    live GUI session and are unusable from a script.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path = DEFAULT_PNG_PATH if output_path is None else Path(output_path)
    assembly = build_assembly() if assembly is None else assembly

    meshes: List[Tuple[AssemblyPart, np.ndarray]] = [
        (part, _triangles(part.solid)) for part in assembly.parts
    ]
    everything = np.vstack([tris.reshape(-1, 3) for _, tris in meshes])

    figure = plt.figure(figsize=(16, 7))
    # The desk is 1200 mm and the clamp is 80: at a shared scale the clamp is
    # a smudge. The elevation therefore zooms to the mount, and only the
    # isometric shows the whole scene.
    clamp_centre = np.array(
        [assembly.yaw_axis_xy_mm[0], assembly.yaw_axis_xy_mm[1], 0.0]
    )
    # azim=0 looks along -X, which puts the desk frame's Y-Z plane on screen --
    # the plane the C-profile lives in. azim=-90 would show the desk's width
    # instead, where the throat is hidden behind the clamp body.
    views = (
        (0, 0, "side elevation, zoomed on the mount", clamp_centre, 130.0),
        (24, -62, "isometric, whole scene", None, None),
    )
    for index, (elevation, azimuth, title, centre_override, span_override) in enumerate(
        views, start=1
    ):
        axes = figure.add_subplot(1, 2, index, projection="3d")
        direction = np.radians([elevation, azimuth])
        light = np.array(
            [
                np.cos(direction[0]) * np.cos(direction[1]),
                np.cos(direction[0]) * np.sin(direction[1]),
                np.sin(direction[0]),
            ]
        )
        for part, tris in meshes:
            normals = np.cross(
                tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
            )
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals = normals / np.where(lengths == 0.0, 1.0, lengths)
            shade = 0.45 + 0.55 * np.clip(np.abs(normals @ light), 0.0, 1.0)
            base = np.array(matplotlib.colors.to_rgb(part.color))
            facecolors = np.clip(shade[:, None] * base[None, :], 0.0, 1.0)
            axes.add_collection3d(
                Poly3DCollection(
                    tris,
                    facecolors=facecolors,
                    edgecolors="none",
                    alpha=0.35 if part.name == "desk" else 1.0,
                )
            )
        low, high = everything.min(axis=0), everything.max(axis=0)
        centre, span = (low + high) / 2.0, (high - low).max() / 2.0
        if centre_override is not None:
            centre, span = centre_override, span_override
        axes.set_xlim(centre[0] - span, centre[0] + span)
        axes.set_ylim(centre[1] - span, centre[1] + span)
        axes.set_zlim(centre[2] - span, centre[2] + span)
        axes.set_box_aspect((1, 1, 1))
        axes.view_init(elev=elevation, azim=azimuth)
        axes.set_title(title, fontsize=10)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_zticks([])
        axes.grid(False)
        for pane in (axes.xaxis, axes.yaxis, axes.zaxis):
            pane.pane.fill = False
            pane.pane.set_edgecolor("none")

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                   markerfacecolor=part.color, markeredgecolor="none",
                   label=part.name)
        for part in assembly.parts
    ]
    figure.legend(
        handles=handles, loc="lower center", ncol=len(handles),
        frameon=False, fontsize=8,
    )
    figure.suptitle(
        f"Desk-sorting arm - assembly preview on a "
        f"{assembly.desk_thickness_mm:.0f} mm desk, arm at zero pose",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(figure)
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the whole assembly on a desk."
    )
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG_PATH)
    parser.add_argument(
        "--desk-thickness", dest="desk_thickness", type=float, default=None,
        help="desk to model, in mm (default: middle of the supported range)",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="print the summary without writing any files",
    )
    args = parser.parse_args(argv)

    try:
        assembly = build_assembly(desk_thickness_mm=args.desk_thickness)
    except ValueError as exc:
        print(f"Cannot build the scene: {exc}", file=sys.stderr)
        return 2

    print(assembly.summary())
    if args.report:
        return 0

    stl_path = export_assembly_stl(args.stl, assembly)
    print(f"Wrote {stl_path}  ({stl_path.stat().st_size / 1024.0:.1f} KB)")
    png_path = render_assembly_png(args.png, assembly)
    print(f"Wrote {png_path}  ({png_path.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
