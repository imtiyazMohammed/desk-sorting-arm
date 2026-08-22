"""
Shared solid primitives for the CAD package.

Kept in one place so that, for example, the hex pocket in the lower jaw and
the hex socket in the knob are provably the same construction rather than two
similar-looking snippets that can drift apart.
"""

from __future__ import annotations

from build123d import Align, BuildPart, BuildSketch, Part, Plane, RegularPolygon, extrude

__all__ = ["hex_prism"]


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
