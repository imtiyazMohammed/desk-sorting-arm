"""
Upper arm (L1) -- the first structural link, shoulder pitch to elbow pitch.

Run directly to write ``cad/output/upper_arm.stl``::

    python3 -m cad.upper_arm
    python3 -m cad.upper_arm --report      # dimensions only, no export

What the part does
------------------
It spans ``ArmGeometry.l1_upper_arm_mm`` between two joint axes: a yoke at the
shoulder end that straddles the shoulder bracket, a hollow rectangular beam in
between, and a housing at the elbow end that carries the third DS3218. It is
the longest link and carries the largest bending moment in the arm.

Why the shoulder end is a yoke rather than a flange
---------------------------------------------------
A single flange bolted to the shoulder servo's horn would sit
``ShoulderBracketParameters.horn_face_y_mm`` -- 35.5 mm -- off the yaw axis,
and a 40 mm beam hung off it would centre nearly 50 mm out.
``forward_kinematics.py`` models no shoulder offset at all, so that distance
would become systematic error in every TCP position the software computes.
Straddling the bracket with two flanges puts the beam back on the axis: the
driven side bolts to the horn, the undriven side runs on a 608ZZ pressed into
its own flange, turning on the axle the bracket's idler plug carries. The joint
is then supported on both sides as well, which matters for a joint already
running past its torque rating.

Section sizing
--------------
The section is not strength-driven. At the worst-case shoulder moment the peak
bending stress is around 1.25 MPa against a 25 MPa allowable -- a factor of
about twenty. What the section costs is mass: see
:attr:`UpperArmParameters.beam_mass_g`. The 40 x 25 x 3 mm section specified for
D.2c is kept because it passes with room to spare and because a stiffer link
is worth more here than a lighter one, but the mass is the number to watch.

Why the cable trough sits on top of the beam rather than in it
--------------------------------------------------------------
An 8 mm channel sunk into a 3 mm top wall does not stay a channel: it cuts
straight through into the hollow and turns a closed box section into an open
one. Closed sections are roughly sixty times stiffer in torsion than open ones
of the same material, and that stiffness is free. So the trough is a raised
U-section standing on the beam's top face -- still an open 8 x 8 channel that
wires drop into, with the box below it left intact.

Coordinate frame
----------------
Origin on the **shoulder pitch axis**, ``+X`` toward the elbow, ``+Z`` up, and
``+Y`` along the pitch axis toward the driven side. At the arm's zero pose the
link lies along the base frame's ``+X``, so this frame and the base frame
coincide at the shoulder.
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
from cad.shoulder_bracket import ShoulderBracketParameters
from src.geometry import (
    DEFAULT_ARM,
    DEFAULT_HARDWARE,
    PETG_ALLOWABLE_STRESS_MPA,
    PETG_DENSITY_G_CM3,
    ArmGeometry,
    HardwareSpec,
    LinkSpec,
)

__all__ = [
    "UpperArmDesignError",
    "UpperArmParameters",
    "build_upper_arm",
    "export_upper_arm",
    "DEFAULT_STL_PATH",
]

DEFAULT_STL_PATH = Path(__file__).resolve().parent / "output" / "upper_arm.stl"


class UpperArmDesignError(DesignRuleError):
    """Raised when a parameter set would produce an unbuildable link."""


@dataclass(frozen=True)
class UpperArmParameters:
    """Fully resolved upper-arm dimensions, in millimetres."""

    # ---- Overall ------------------------------------------------------------
    length_mm: float
    beam_width_mm: float
    beam_height_mm: float
    wall_thickness_mm: float
    beam_start_x_mm: float

    # ---- Shoulder yoke ------------------------------------------------------
    flange_diameter_mm: float
    driven_flange_inner_y_mm: float
    driven_flange_thickness_mm: float
    idler_flange_inner_y_mm: float
    idler_flange_thickness_mm: float
    horn_recess_diameter_mm: float
    horn_recess_depth_mm: float
    horn_bolt_positions: Tuple[Tuple[float, float], ...]
    horn_bolt_clearance_diameter_mm: float
    horn_counterbore_diameter_mm: float
    horn_counterbore_depth_mm: float
    horn_access_bore_mm: float
    idler_seat_diameter_mm: float
    idler_seat_depth_mm: float
    yoke_block_length_mm: float
    yoke_taper_length_mm: float

    # ---- Elbow housing ------------------------------------------------------
    elbow_axis_x_mm: float
    housing_x_min_mm: float
    housing_x_max_mm: float
    housing_half_y_mm: float
    housing_z_min_mm: float
    housing_z_max_mm: float
    elbow_cavity_x_min_mm: float
    elbow_cavity_x_max_mm: float
    elbow_cavity_z_min_mm: float
    elbow_cavity_z_max_mm: float
    elbow_wall_inner_y_mm: float
    elbow_wall_thickness_mm: float
    elbow_screw_positions: Tuple[Tuple[float, float], ...]
    elbow_screw_diameter_mm: float
    elbow_screw_depth_mm: float

    # ---- Cable trough -------------------------------------------------------
    channel_width_mm: float
    channel_depth_mm: float
    channel_wall_mm: float
    strain_relief_pitch_mm: float
    strain_relief_tab_width_mm: float

    # ---- Carried through ----------------------------------------------------
    bracket_max_radius_mm: float
    shoulder_moment_nm: float
    section_modulus_mm3: float
    cross_section_area_mm2: float
    min_wall_thickness_mm: float

    # =====================================================================
    # Construction
    # =====================================================================

    @classmethod
    def from_geometry(
        cls,
        arm: Optional[ArmGeometry] = None,
        hardware: Optional[HardwareSpec] = None,
        link: Optional[LinkSpec] = None,
    ) -> "UpperArmParameters":
        """
        Derive every dimension from the geometry and hardware singletons.

        The shoulder end is placed from ``ShoulderBracketParameters``, so the
        yoke's flanges land on the horn the bracket actually presents rather
        than on an assumed position.

        Raises
        ------
        UpperArmDesignError
            If the resulting design violates a clearance.
        """
        arm = DEFAULT_ARM if arm is None else arm
        hardware = DEFAULT_HARDWARE if hardware is None else hardware
        link = hardware.upper_arm_link if link is None else link
        horn = hardware.servo_horn
        idler = hardware.shoulder_idler_bearing
        servo = hardware.elbow_pitch_servo
        clearance = hardware.print_clearance_mm
        wall = hardware.min_wall_thickness_mm

        bracket = ShoulderBracketParameters.from_geometry(arm, hardware)

        flange_diameter, flange_thickness = link.end_flange_mm
        # The undriven flange has to swallow a bearing, so it is deeper.
        idler_flange_thickness = idler.width_mm + wall / 2.0 + 1.0

        # The beam may only begin outside everything the bracket occupies, or
        # the joint cannot swing. Measured as a full circle about the pitch
        # axis, which is conservative -- the joint's travel does not reach the
        # bracket's rear-bottom corner -- but robust to a limit change.
        beam_half_diagonal = (
            (link.cross_section_height_mm / 2.0) ** 2
        ) ** 0.5
        beam_start_x = (
            bracket.max_radius_from_pitch_axis_mm**2 - beam_half_diagonal**2
        ) ** 0.5 + wall

        # ---- Elbow servo, laid out exactly like the shoulder's ------------
        elbow_axis_x = link.length_mm
        body_x_centre = elbow_axis_x - servo.body_offset_from_shaft_axis_mm
        cavity_x_half = (servo.body_length_mm + 2.0 * clearance) / 2.0
        cavity_z_half = (servo.body_width_mm + 2.0 * clearance) / 2.0
        elbow_wall_inner_y = servo.body_depth_behind_ears_mm / 2.0 + clearance

        elbow_screws = tuple(
            (
                body_x_centre + sx * servo.flange_hole_spacing_long_mm / 2.0,
                sz * servo.flange_hole_spacing_short_mm / 2.0,
            )
            for sx in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        )

        params = cls(
            length_mm=link.length_mm,
            beam_width_mm=link.cross_section_width_mm,
            beam_height_mm=link.cross_section_height_mm,
            wall_thickness_mm=link.wall_thickness_mm,
            beam_start_x_mm=beam_start_x,
            flange_diameter_mm=flange_diameter,
            driven_flange_inner_y_mm=bracket.horn_face_y_mm,
            driven_flange_thickness_mm=flange_thickness,
            idler_flange_inner_y_mm=-bracket.horn_face_y_mm,
            idler_flange_thickness_mm=idler_flange_thickness,
            horn_recess_diameter_mm=horn.disc_diameter_mm + 2.0 * clearance,
            horn_recess_depth_mm=1.0,
            horn_bolt_positions=horn.bolt_positions(),
            horn_bolt_clearance_diameter_mm=horn.bolt_clearance_diameter_mm,
            horn_counterbore_diameter_mm=(
                horn.bolt_head_diameter_mm + 2.0 * clearance
            ),
            horn_counterbore_depth_mm=horn.bolt_head_height_mm + 0.5,
            horn_access_bore_mm=horn.hub_diameter_mm - 5.0,
            idler_seat_diameter_mm=idler.seat_diameter_mm,
            idler_seat_depth_mm=idler.width_mm,
            yoke_block_length_mm=2.0 * wall,
            yoke_taper_length_mm=6.0 * wall,
            elbow_axis_x_mm=elbow_axis_x,
            housing_x_min_mm=body_x_centre - cavity_x_half - wall,
            housing_x_max_mm=body_x_centre + cavity_x_half + wall,
            housing_half_y_mm=elbow_wall_inner_y + wall + 2.0,
            housing_z_min_mm=-cavity_z_half - wall,
            housing_z_max_mm=cavity_z_half + wall,
            elbow_cavity_x_min_mm=body_x_centre - cavity_x_half,
            elbow_cavity_x_max_mm=body_x_centre + cavity_x_half,
            elbow_cavity_z_min_mm=-cavity_z_half,
            elbow_cavity_z_max_mm=cavity_z_half,
            elbow_wall_inner_y_mm=elbow_wall_inner_y,
            elbow_wall_thickness_mm=wall + 2.0,
            elbow_screw_positions=elbow_screws,
            elbow_screw_diameter_mm=servo.flange_hole_diameter_mm,
            elbow_screw_depth_mm=wall,
            channel_width_mm=link.cable_channel_mm[0],
            channel_depth_mm=link.cable_channel_mm[1],
            channel_wall_mm=link.wall_thickness_mm,
            strain_relief_pitch_mm=link.strain_relief_pitch_mm,
            strain_relief_tab_width_mm=link.strain_relief_tab_width_mm,
            bracket_max_radius_mm=bracket.max_radius_from_pitch_axis_mm,
            shoulder_moment_nm=arm.shoulder_moment_nm(),
            section_modulus_mm3=link.section_modulus_mm3,
            cross_section_area_mm2=link.cross_section_area_mm2,
            min_wall_thickness_mm=wall,
        )
        params.validate()
        return params

    # =====================================================================
    # Derived accessors
    # =====================================================================

    @property
    def bending_stress_mpa(self) -> float:
        """Peak bending stress at the shoulder end, in MPa."""
        return self.shoulder_moment_nm * 1000.0 / self.section_modulus_mm3

    @property
    def stress_margin(self) -> float:
        """How many times over the allowable stress the section is."""
        return PETG_ALLOWABLE_STRESS_MPA / self.bending_stress_mpa

    @property
    def beam_mass_g(self) -> float:
        """Mass of the constant-section beam run alone, in grams."""
        run = self.housing_x_min_mm - self.beam_start_x_mm
        return self.cross_section_area_mm2 * run / 1000.0 * PETG_DENSITY_G_CM3

    @property
    def beam_run_mm(self) -> float:
        """Length of the constant-section beam between yoke and housing."""
        return self.housing_x_min_mm - self.beam_start_x_mm

    @property
    def yoke_outer_width_mm(self) -> float:
        """Overall width across the yoke's two flanges, in mm."""
        return (
            self.driven_flange_inner_y_mm
            + self.driven_flange_thickness_mm
            + abs(self.idler_flange_inner_y_mm)
            + self.idler_flange_thickness_mm
        )

    @property
    def idler_seat_floor_mm(self) -> float:
        """Material left behind the idler bearing's seat."""
        return self.idler_flange_thickness_mm - self.idler_seat_depth_mm

    @property
    def strain_relief_positions(self) -> Tuple[float, ...]:
        """X centres of the tabs that bridge the cable trough."""
        start = self.beam_start_x_mm + self.yoke_block_length_mm
        end = self.housing_x_min_mm
        positions = []
        x = start + self.strain_relief_pitch_mm
        while x < end - self.strain_relief_tab_width_mm:
            positions.append(x)
            x += self.strain_relief_pitch_mm
        return tuple(positions)

    @property
    def channel_top_z_mm(self) -> float:
        """Top of the cable trough's side walls, in mm."""
        return self.beam_height_mm / 2.0 + self.channel_depth_mm

    # =====================================================================
    # Validation
    # =====================================================================

    def validate(self) -> DesignStatus:
        """
        Re-derive every clearance and refuse an unbuildable link.

        Raises
        ------
        UpperArmDesignError
            Naming the first violated constraint.
        """
        for name, value in (
            ("length_mm", self.length_mm),
            ("beam_width_mm", self.beam_width_mm),
            ("beam_height_mm", self.beam_height_mm),
            ("wall_thickness_mm", self.wall_thickness_mm),
            ("flange_diameter_mm", self.flange_diameter_mm),
        ):
            if value <= 0.0:
                raise UpperArmDesignError(
                    DesignStatus.INVALID_PARAMETER,
                    f"{name} must be positive, got {value}.",
                )

        # ---- Strength -----------------------------------------------------
        if self.bending_stress_mpa > PETG_ALLOWABLE_STRESS_MPA:
            raise UpperArmDesignError(
                DesignStatus.INVALID_PARAMETER,
                f"The section carries {self.bending_stress_mpa:.2f} MPa at the "
                f"shoulder end under {self.shoulder_moment_nm:.2f} N.m, past "
                f"the {PETG_ALLOWABLE_STRESS_MPA:.0f} MPa allowable. Deepen "
                "the section: bending capacity goes as the square of its "
                "height.",
            )

        # ---- The yoke must clear the bracket it straddles -----------------
        if self.beam_start_x_mm <= self.bracket_max_radius_mm:
            raise UpperArmDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The beam starts {self.beam_start_x_mm:.2f} mm from the pitch "
                f"axis but the shoulder bracket reaches "
                f"{self.bracket_max_radius_mm:.2f} mm. The joint could not "
                "rotate without the beam striking the bracket.",
            )
        if self.driven_flange_inner_y_mm <= self.beam_width_mm / 2.0:
            raise UpperArmDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The yoke's flanges sit "
                f"{self.driven_flange_inner_y_mm:.2f} mm out but the beam is "
                f"{self.beam_width_mm:.2f} mm wide, so its side would foul "
                "them.",
            )
        if self.flange_diameter_mm <= self.horn_recess_diameter_mm + 2.0 * self.min_wall_thickness_mm:
            raise UpperArmDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"A {self.flange_diameter_mm:.2f} mm flange leaves less than "
                f"{self.min_wall_thickness_mm:.2f} mm around a "
                f"{self.horn_recess_diameter_mm:.2f} mm horn recess.",
            )
        if self.flange_diameter_mm <= self.idler_seat_diameter_mm + 2.0 * self.min_wall_thickness_mm:
            raise UpperArmDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"A {self.flange_diameter_mm:.2f} mm flange leaves less than "
                f"{self.min_wall_thickness_mm:.2f} mm around a "
                f"{self.idler_seat_diameter_mm:.2f} mm bearing seat.",
            )
        if self.idler_seat_floor_mm < self.min_wall_thickness_mm / 2.0:
            raise UpperArmDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"Only {self.idler_seat_floor_mm:.2f} mm behind the idler "
                "bearing's seat.",
            )
        counterbore_edge = (
            max(abs(x) for x, _ in self.horn_bolt_positions) ** 2
            + max(abs(y) for _, y in self.horn_bolt_positions) ** 2
        ) ** 0.5 + self.horn_counterbore_diameter_mm / 2.0
        if counterbore_edge > self.flange_diameter_mm / 2.0:
            raise UpperArmDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"A horn-screw counterbore reaches {counterbore_edge:.2f} mm "
                f"from the axis, off a "
                f"{self.flange_diameter_mm / 2.0:.2f} mm flange.",
            )
        if self.horn_counterbore_depth_mm >= self.driven_flange_thickness_mm:
            raise UpperArmDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"A {self.horn_counterbore_depth_mm:.2f} mm counterbore in a "
                f"{self.driven_flange_thickness_mm:.2f} mm flange breaks "
                "through.",
            )

        # ---- The elbow servo has to fit ------------------------------------
        if self.housing_x_min_mm <= self.beam_start_x_mm:
            raise UpperArmDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The elbow housing starts at "
                f"{self.housing_x_min_mm:.2f} mm and the yoke ends at "
                f"{self.beam_start_x_mm:.2f} mm; there is no beam between "
                "them.",
            )
        if self.elbow_cavity_z_max_mm - self.elbow_cavity_z_min_mm <= 0.0:
            raise UpperArmDesignError(
                DesignStatus.INVALID_PARAMETER,
                "The elbow cavity has no height.",
            )
        if self.housing_half_y_mm <= self.elbow_wall_inner_y_mm:
            raise UpperArmDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"The elbow housing reaches "
                f"{self.housing_half_y_mm:.2f} mm but its servo slot's walls "
                f"start at {self.elbow_wall_inner_y_mm:.2f} mm.",
            )
        if self.elbow_screw_depth_mm >= self.elbow_wall_thickness_mm:
            raise UpperArmDesignError(
                DesignStatus.WALL_TOO_THIN,
                f"A {self.elbow_screw_depth_mm:.2f} mm blind hole in a "
                f"{self.elbow_wall_thickness_mm:.2f} mm wall leaves no floor.",
            )

        # ---- The trough must not eat the beam ------------------------------
        if self.channel_width_mm + 2.0 * self.channel_wall_mm > self.beam_width_mm:
            raise UpperArmDesignError(
                DesignStatus.FEATURE_COLLISION,
                f"The cable trough is "
                f"{self.channel_width_mm + 2.0 * self.channel_wall_mm:.2f} mm "
                f"across, wider than the "
                f"{self.beam_width_mm:.2f} mm beam it stands on.",
            )
        return DesignStatus.OK

    # =====================================================================
    # Reporting
    # =====================================================================

    def report(self) -> str:
        """Human-readable dimension summary, printed by ``--report``."""
        return (
            f"Upper arm (L1) parameters\n"
            f"-------------------------\n"
            f"  Axis to axis           : {self.length_mm:.2f} mm "
            f"(shoulder at x = 0, elbow at x = {self.elbow_axis_x_mm:.2f})\n"
            f"  Beam section           : {self.beam_width_mm:.2f} (Y) x "
            f"{self.beam_height_mm:.2f} (Z) mm, "
            f"{self.wall_thickness_mm:.2f} mm walls\n"
            f"  Beam run               : x {self.beam_start_x_mm:.2f} .. "
            f"{self.housing_x_min_mm:.2f} ({self.beam_run_mm:.2f} mm)\n"
            f"\n"
            f"  Shoulder yoke\n"
            f"    driven flange        : y {self.driven_flange_inner_y_mm:.2f} "
            f".. "
            f"{self.driven_flange_inner_y_mm + self.driven_flange_thickness_mm:.2f}"
            f", {self.flange_diameter_mm:.2f} mm dia\n"
            f"    idler flange         : y "
            f"{self.idler_flange_inner_y_mm - self.idler_flange_thickness_mm:.2f}"
            f" .. {self.idler_flange_inner_y_mm:.2f}, "
            f"{self.idler_seat_diameter_mm:.2f} mm seat x "
            f"{self.idler_seat_depth_mm:.2f} mm deep "
            f"({self.idler_seat_floor_mm:.2f} mm floor)\n"
            f"    overall width        : {self.yoke_outer_width_mm:.2f} mm\n"
            f"    clears the bracket   : beam starts at "
            f"{self.beam_start_x_mm:.2f} mm vs its "
            f"{self.bracket_max_radius_mm:.2f} mm reach\n"
            f"\n"
            f"  Elbow housing          : x {self.housing_x_min_mm:.2f} .. "
            f"{self.housing_x_max_mm:.2f}, y "
            f"+/-{self.housing_half_y_mm:.2f}, z "
            f"{self.housing_z_min_mm:.2f} .. {self.housing_z_max_mm:.2f}\n"
            f"    servo screws         : 4 x "
            f"{self.elbow_screw_diameter_mm:.2f} mm blind, "
            f"{self.elbow_screw_depth_mm:.2f} mm deep\n"
            f"\n"
            f"  Cable trough           : {self.channel_width_mm:.2f} x "
            f"{self.channel_depth_mm:.2f} mm open channel on the top face, "
            f"walls {self.channel_wall_mm:.2f} mm\n"
            f"    strain-relief tabs   : {len(self.strain_relief_positions)} "
            f"at {self.strain_relief_pitch_mm:.0f} mm pitch\n"
            f"\n"
            f"  Structure\n"
            f"    shoulder moment      : {self.shoulder_moment_nm:.2f} N.m "
            f"(arm horizontal, full payload at full reach)\n"
            f"    section modulus      : {self.section_modulus_mm3:.0f} mm3\n"
            f"    bending stress       : {self.bending_stress_mpa:.2f} MPa vs "
            f"{PETG_ALLOWABLE_STRESS_MPA:.0f} allowable "
            f"({self.stress_margin:.1f}x margin)\n"
            f"    beam run mass        : {self.beam_mass_g:.0f} g of PETG "
            f"(section alone)\n"
        )


def build_upper_arm(params: Optional[UpperArmParameters] = None) -> Part:
    """
    Construct the upper arm solid.

    Returns
    -------
    build123d.Part
        A single solid. Origin on the shoulder pitch axis, ``+X`` to the elbow.

    Raises
    ------
    UpperArmDesignError
        If ``params`` fails :meth:`UpperArmParameters.validate`.
    """
    params = UpperArmParameters.from_geometry() if params is None else params
    params.validate()

    bottom = (Align.CENTER, Align.CENTER, Align.MIN)
    half_h = params.beam_height_mm / 2.0
    block_end_x = params.beam_start_x_mm + params.yoke_block_length_mm

    def y_slab(y_min, y_max, x_min, x_max, z_min, z_max):
        """A box spanning the given ranges in all three axes."""
        return Pos(
            (x_min + x_max) / 2.0, (y_min + y_max) / 2.0, z_min
        ) * Box(x_max - x_min, y_max - y_min, z_max - z_min, align=bottom)

    # ---- Shoulder yoke: two flange plates straddling the bracket ----------
    part = None
    flanges = (
        (
            params.driven_flange_inner_y_mm,
            params.driven_flange_inner_y_mm + params.driven_flange_thickness_mm,
        ),
        (
            params.idler_flange_inner_y_mm - params.idler_flange_thickness_mm,
            params.idler_flange_inner_y_mm,
        ),
    )
    for y_min, y_max in flanges:
        disc = Pos(0, (y_min + y_max) / 2.0, 0) * Rot(-90, 0, 0) * Cylinder(
            radius=params.flange_diameter_mm / 2.0,
            height=y_max - y_min,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        plate = y_slab(y_min, y_max, 0.0, block_end_x, -half_h, half_h)
        part = disc + plate if part is None else part + disc + plate

    # ---- Transition block joining both flanges to the beam ---------------
    part += y_slab(
        params.idler_flange_inner_y_mm - params.idler_flange_thickness_mm,
        params.driven_flange_inner_y_mm + params.driven_flange_thickness_mm,
        params.beam_start_x_mm,
        block_end_x,
        -half_h,
        half_h,
    )

    # ---- Taper from the block's full width down to the beam --------------
    # Gussets rather than a swept fillet, for the reason cad/README.md gives.
    half_w = params.beam_width_mm / 2.0
    for sign, rotation in ((1.0, -90.0), (-1.0, 90.0)):
        part += (
            Pos(block_end_x, sign * half_w, 0)
            * Rot(rotation, 0, 0)
            * right_triangle_prism(
                params.yoke_taper_length_mm,
                params.driven_flange_inner_y_mm - half_w,
                params.beam_height_mm,
            )
        )

    # ---- Hollow rectangular beam -----------------------------------------
    part += y_slab(
        -half_w, half_w, params.beam_start_x_mm, params.housing_x_min_mm,
        -half_h, half_h,
    )
    part -= y_slab(
        -half_w + params.wall_thickness_mm,
        half_w - params.wall_thickness_mm,
        block_end_x + params.yoke_taper_length_mm,
        params.housing_x_min_mm,
        -half_h + params.wall_thickness_mm,
        half_h - params.wall_thickness_mm,
    )

    # ---- Cable trough standing on the beam's top face --------------------
    trough_half = params.channel_width_mm / 2.0 + params.channel_wall_mm
    trough_x_min = block_end_x
    part += y_slab(
        -trough_half, trough_half, trough_x_min, params.housing_x_min_mm,
        half_h, params.channel_top_z_mm,
    )
    part -= y_slab(
        -params.channel_width_mm / 2.0,
        params.channel_width_mm / 2.0,
        trough_x_min,
        params.housing_x_min_mm,
        half_h,
        params.channel_top_z_mm,
    )
    # Retention tabs bridging the channel: wires push under them, and the
    # channel stays open everywhere else so they can be laid in rather than
    # threaded.
    for tab_x in params.strain_relief_positions:
        part += y_slab(
            -trough_half, trough_half,
            tab_x, tab_x + params.strain_relief_tab_width_mm,
            params.channel_top_z_mm,
            params.channel_top_z_mm + params.channel_wall_mm,
        )

    # ---- Elbow servo housing ---------------------------------------------
    part += y_slab(
        -params.housing_half_y_mm, params.housing_half_y_mm,
        params.housing_x_min_mm, params.housing_x_max_mm,
        params.housing_z_min_mm, params.housing_z_max_mm,
    )
    # Slot for the servo, open at the top so its 54.5 mm ears can drop past.
    part -= y_slab(
        -params.elbow_wall_inner_y_mm, params.elbow_wall_inner_y_mm,
        params.elbow_cavity_x_min_mm, params.elbow_cavity_x_max_mm,
        params.elbow_cavity_z_min_mm, params.housing_z_max_mm,
    )
    # The same slot through each wall, so the case's protruding top section and
    # its shaft pass out to the horn on the driven side.
    for sign in (1.0, -1.0):
        inner = sign * params.elbow_wall_inner_y_mm
        outer = sign * params.housing_half_y_mm
        part -= y_slab(
            min(inner, outer), max(inner, outer),
            params.elbow_cavity_x_min_mm, params.elbow_cavity_x_max_mm,
            params.elbow_cavity_z_min_mm, params.housing_z_max_mm,
        )
    # Blind retention holes in both walls.
    for sign in (1.0, -1.0):
        inner = sign * params.elbow_wall_inner_y_mm
        for screw_x, screw_z in params.elbow_screw_positions:
            part -= Pos(screw_x, inner, screw_z) * Rot(90, 0, 0) * Cylinder(
                radius=params.elbow_screw_diameter_mm / 2.0,
                height=params.elbow_screw_depth_mm,
                align=(
                    Align.CENTER,
                    Align.CENTER,
                    Align.MIN if sign < 0 else Align.MAX,
                ),
            )

    # ---- Horn interface in the driven flange ------------------------------
    driven_inner = params.driven_flange_inner_y_mm
    driven_outer = driven_inner + params.driven_flange_thickness_mm
    part -= Pos(0, driven_inner, 0) * Rot(-90, 0, 0) * Cylinder(
        radius=params.horn_recess_diameter_mm / 2.0,
        height=params.horn_recess_depth_mm,
        align=bottom,
    )
    part -= Pos(0, driven_inner, 0) * Rot(-90, 0, 0) * Cylinder(
        radius=params.horn_access_bore_mm / 2.0,
        height=params.driven_flange_thickness_mm,
        align=bottom,
    )
    for screw_y, screw_z in params.horn_bolt_positions:
        part -= Pos(screw_y, driven_inner, screw_z) * Rot(-90, 0, 0) * Cylinder(
            radius=params.horn_bolt_clearance_diameter_mm / 2.0,
            height=params.driven_flange_thickness_mm,
            align=bottom,
        )
        part -= Pos(
            screw_y,
            driven_outer - params.horn_counterbore_depth_mm,
            screw_z,
        ) * Rot(-90, 0, 0) * Cylinder(
            radius=params.horn_counterbore_diameter_mm / 2.0,
            height=params.horn_counterbore_depth_mm,
            align=bottom,
        )

    # ---- Bearing seat in the idler flange ---------------------------------
    part -= Pos(
        0, params.idler_flange_inner_y_mm - params.idler_seat_depth_mm, 0
    ) * Rot(-90, 0, 0) * Cylinder(
        radius=params.idler_seat_diameter_mm / 2.0,
        height=params.idler_seat_depth_mm,
        align=bottom,
    )
    return part


def export_upper_arm(
    output_path: Optional[Path] = None,
    params: Optional[UpperArmParameters] = None,
) -> Path:
    """
    Build the upper arm and write it to an STL, creating parent directories.

    Raises
    ------
    UpperArmDesignError
        If the parameters are unbuildable.
    RuntimeError
        If build123d reports the export failed.
    """
    output_path = DEFAULT_STL_PATH if output_path is None else Path(output_path)
    part = build_upper_arm(params)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not export_stl(part, output_path):
        raise RuntimeError(f"build123d failed to write STL to {output_path}.")
    return output_path


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the upper arm STL from src/geometry.py."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_STL_PATH)
    parser.add_argument(
        "--report", action="store_true",
        help="print resolved dimensions without exporting",
    )
    args = parser.parse_args(argv)

    try:
        params = UpperArmParameters.from_geometry()
    except UpperArmDesignError as exc:
        print(f"Design rule check failed: {exc}", file=sys.stderr)
        return 1

    print(params.report())
    if args.report:
        return 0

    written = export_upper_arm(args.output, params)
    print(f"Wrote {written}  ({written.stat().st_size / 1024.0:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
