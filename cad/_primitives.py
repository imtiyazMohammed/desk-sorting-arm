"""
Shared solid primitives for the CAD package.

Kept in one place so that, for example, the hex pocket in the lower jaw and
the hex socket in the knob are provably the same construction rather than two
similar-looking snippets that can drift apart.
"""

from __future__ import annotations

from build123d import (
    Align,
    BuildPart,
    BuildSketch,
    Part,
    Plane,
    Polygon,
    RegularPolygon,
    extrude,
)

__all__ = ["hex_prism", "right_triangle_prism"]


def hex_prism(across_flats_mm: float, height_mm: float, rotation_deg: float = 0.0) -> Part:
    """
    A hexagonal prism sized by its across-flats dimension.

    Fasteners are specified across flats (a 13 mm spanner fits an M8 nut), but
    :class:`build123d.RegularPolygon` takes a radius. Passing
    ``major_radius=False`` makes that radius the inradius, which is exactly
    half the across-flats dimension.

    Parameters
    ----------
    across_flats_mm:
        Distance between opposite flats. Include fit clearance in this value.
    height_mm:
        Extrusion height. The prism's base sits on the XY plane, extruded +Z.
    rotation_deg:
        Rotation about Z, for orienting a flat toward a print bed or a wall.

    Raises
    ------
    ValueError
        If either dimension is not positive.
    """
    if across_flats_mm <= 0.0:
        raise ValueError(
            f"across_flats_mm must be positive, got {across_flats_mm}."
        )
    if height_mm <= 0.0:
        raise ValueError(f"height_mm must be positive, got {height_mm}.")

    with BuildPart() as builder:
        with BuildSketch(Plane.XY):
            RegularPolygon(
                radius=across_flats_mm / 2.0,
                side_count=6,
                major_radius=False,
                rotation=rotation_deg,
                align=(Align.CENTER, Align.CENTER),
            )
        extrude(amount=height_mm)
    return builder.part


def right_triangle_prism(
    leg_x_mm: float, leg_z_mm: float, width_mm: float
) -> Part:
    """
    A right-triangular prism, for gussets at an inner corner.

    The triangle lies in the XZ plane with its right angle at the origin, legs
    running along +X and +Z, and the hypotenuse joining their tips. It is
    extruded symmetrically about the XZ plane to ``width_mm`` along Y, so the
    caller positions it by its corner rather than by a centroid.

    Gussets rather than fillets throughout this package: a swept fillet is the
    most fragile operation to re-run when upstream dimensions change, and every
    dimension here is derived from ``src/geometry.py`` and expected to move.

    Parameters
    ----------
    leg_x_mm, leg_z_mm:
        Lengths of the two legs, in mm.
    width_mm:
        Extrusion depth along Y, centred on the XZ plane.

    Raises
    ------
    ValueError
        If any dimension is not positive.
    """
    for name, value in (
        ("leg_x_mm", leg_x_mm), ("leg_z_mm", leg_z_mm), ("width_mm", width_mm)
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")

    with BuildPart() as builder:
        with BuildSketch(Plane.XZ) as sketch:
            Polygon(
                (0.0, 0.0), (leg_x_mm, 0.0), (0.0, leg_z_mm),
                align=(Align.MIN, Align.MIN),
            )
        extrude(amount=width_mm / 2.0, both=True)
    return builder.part
