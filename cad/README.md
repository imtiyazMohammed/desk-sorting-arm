# Parametric CAD

Mechanical parts for the desk-sorting arm, modelled in
[build123d](https://build123d.readthedocs.io/) (a Python CAD library over the
OpenCASCADE kernel).

## The rule: no dimensions live here

Every part in this package derives its dimensions from `src/geometry.py`:

- **`DEFAULT_ARM`** (`ArmGeometry`) — kinematic lengths and the base height
  budget.
- **`DEFAULT_HARDWARE`** (`HardwareSpec`) — off-the-shelf component envelopes:
  the DS3218 servo, the 608ZZ bearing, M4 fasteners, print clearances.

No module in `cad/` declares a physical dimension of its own. This is the same
discipline that keeps `arm_chain.py` building its ikpy chain programmatically
instead of from a URDF file: one number, one place, and a change propagates
atomically instead of drifting.

The practical consequence: to make the pedestal fit a different servo, edit
`ServoSpec` in `src/geometry.py`. Do not edit `cad/base_pedestal.py`.

## Regenerating STLs

```bash
python3 -m cad.base_pedestal                      # -> cad/output/base_pedestal.stl
python3 -m cad.base_pedestal --report             # dimensions only, no export
python3 -m cad.base_pedestal --output /tmp/x.stl  # somewhere else
```

`--report` prints every resolved dimension along with the outstanding
verification warnings. Run it after changing anything in `src/geometry.py` to
see what moved.

## Design rule checks

`PedestalParameters.validate()` is not a formality. It re-derives every
clearance and refuses to build a part that would print badly:

| Check | `DesignStatus` on failure |
|---|---|
| Base height budget leaves room for a pedestal | `NEGATIVE_HEIGHT` |
| Vertical layout ordered: flange < ear shelf < cavity ceiling < bearing seat | `FEATURE_COLLISION` |
| Ceiling between cavity and bearing seat ≥ `min_wall_thickness_mm` | `WALL_TOO_THIN` |
| Radial wall outside the ear slot ≥ `min_wall_thickness_mm` | `WALL_TOO_THIN` |
| Bearing seat does not undercut the body wall | `WALL_TOO_THIN` |
| Shaft bore narrower than the bearing seat (so the seat has a floor) | `FEATURE_COLLISION` |
| Bolt access channels clear of the servo cavity | `FEATURE_COLLISION` |
| M3 pilot holes land in shelf material and do not break through | `FEATURE_COLLISION` |
| Cable slot fits between flange top and ear shelf | `FEATURE_COLLISION` |

This is what makes the module safe to drive from swept parameters: a violating
combination fails loudly instead of quietly emitting a part with a 0.2 mm wall.

## Parts

### `base_pedestal.py`

Bolts to the desk and carries the base yaw joint. Bottom to top:

```
                    ┌───────────────┐  z = 70.00  top face
                    │   ▓▓▓▓▓▓▓▓▓   │             608ZZ seat, Ø21.90 × 6.50 deep
                    ├───┬───────┬───┤  z = 63.50  seat floor (4.00 mm ceiling)
                    │   │  ███  │   │             shaft bore, Ø13.50
                    ├───┴───────┴───┤  z = 59.50  cavity ceiling
                    │  ░░░░░░░░░░   │             servo body pocket, 40.50 × 20.50
       M3 pilots →  ├──┬─────────┬──┤  z = 49.50  ear shelf  ← servo ears seat here
                    │  ░░░░░░░░░░░  │             ear slot, 54.50 × 20.50
                    │  ░░░░░░░░░░░  │             (cable slot exits +X)
                    ├───────────────┤  z =  5.00  flange top
                    │               │             flange, Ø93.27
                    └───────────────┘  z =  0.00  desk surface
                         Ø85.27 body
```

**Resolved dimensions** (from the current `src/geometry.py`):

| | |
|---|---|
| Total height | 70.00 mm (`base_height_mm` 100 − `BaseStack.allowance_mm` 30) |
| Body OD | 85.27 mm (derived) |
| Flange OD | 93.27 mm (derived) |
| Servo cavity | 40.50 × 20.50 mm body, 54.50 × 20.50 mm ear slot |
| Cavity X offset | −10.00 mm |
| Bearing seat | Ø21.90 × 6.50 mm deep |
| Desk bolts | 4 × M4 (Ø4.50 clearance) on a 60 mm bolt circle at 45° |
| Volume | ≈ 324 cm³ |

#### Why the cavity is offset

The DS3218's output shaft is not centred on its body. For the *shaft* to sit on
the yaw axis — which it must, since that shaft defines the axis — the body has
to sit 10 mm off-centre. That offset, not the servo's size, is what drives the
pedestal's diameter: it pushes the far corner of the ear slot out to r ≈ 38.6 mm,
and one 4 mm wall beyond that gives the 85 mm body.

#### Why the bolts have access channels

The specified 60 mm bolt circle puts the M4 holes at r = 30 mm, with their outer
edge at r = 32.25 mm — comfortably **inside** the 42.6 mm body wall. Bolt heads
there would be unreachable. Each bolt therefore gets an 8 mm vertical counterbore
running from the top face down to the flange, for a long hex key. The pattern is
rotated 45° so the channels miss the offset cavity; `validate()` proves the
clearance (10.7 mm against 8 mm needed) rather than assuming it.

**This makes assembly order load-bearing**: bolt the pedestal to the desk
*first*, then fit the servo. Once the servo is in, the channels are still open
but the work is harder.

#### Why the bearing stands proud

The 608ZZ's outer race presses into the top seat; the yaw turntable above
clamps its **inner** race. The seat is cut 0.5 mm shallower than the bearing is
wide so the bearing stands proud of the printed top face — otherwise the
turntable would scrub on the plastic instead of turning on the bearing, and
the whole point of fitting a thrust bearing (taking axial load off the servo's
output shaft) is lost.

#### Servo insertion

The cavity is open at the bottom and stepped: a 54.50 mm ear slot from the
underside up to z = 49.50, then a 40.50 mm body pocket above it. The servo goes
in **from below** and is pushed up until its mounting ears meet the step. Four
M3 screws are then driven upward through the ears into blind pilot holes in the
shelf. A radial slot on the +X side takes the servo lead out.

## ⚠ Unverified dimensions

The DS3218 body envelope (40 × 20 × 40.5 mm), mass, gear ratio, torque and speed
are confirmed against the DSServo product datasheet. **The mounting-flange
geometry and output-shaft placement are not.** The datasheet's dimensioned
drawing is a raster image with no extractable numbers, and suppliers ship
visually similar variants.

These fields carry standard-size-servo placeholders and are enumerated in
`ServoSpec.UNVERIFIED_FIELDS`:

```
flange_span_mm  flange_thickness_mm  flange_hole_spacing_long_mm
flange_hole_spacing_short_mm  flange_hole_diameter_mm
shaft_offset_from_body_end_mm  shaft_boss_diameter_mm
shaft_boss_height_mm  output_shaft_diameter_mm  travel_deg
```

`python3 -m src.geometry` and `python3 -m cad.base_pedestal` both print this
list every run. The generated STL is geometrically valid and the design rule
checks all pass, but **measure a real DS3218 before printing for final
assembly** — `shaft_offset_from_body_end_mm` in particular shifts the entire
cavity, and `flange_span_mm` sets the body diameter.

Also confirm which **travel variant** was ordered. The DS3218 ships in 180° and
270° versions that are otherwise near-identical; `ArmGeometry`'s base yaw limit
is ±135° (a 270° span), which the 180° variant cannot deliver.

## Known future work

- **The pedestal is over-material.** At ≈ 324 cm³ it is mostly solid; a lattice
  or shelled interior would cut print time and filament substantially. Deferred
  because wall topology should be decided once the shoulder loads from Session
  D.3 are known.
- **`BaseStack` is provisional.** The 6 mm turntable plate and 24 mm shoulder
  bracket rise are estimates. Sessions D.2 and D.3 replace them with measured
  values; the pedestal height then follows automatically.
- **No fillets.** Sharp internal corners are stress risers and print with
  elephant's foot. Filleting is deliberately deferred until the geometry stops
  moving, since fillet operations are the most fragile part of a build123d
  model to re-run against changed dimensions.

## Testing

`tests/test_cad.py` covers hardware-spec validation, parameter derivation, every
design rule check, solid construction, and STL integrity. The mesh checks parse
the exported STL directly rather than trusting the kernel's own report, because
"watertight" is a property of the tessellation a slicer will read:

- every **undirected** edge used by exactly two faces (closed surface)
- every **directed** edge traversed exactly once (consistent winding)
- enclosed volume positive, and within 1% of the kernel's B-rep volume

```bash
python3 -m pytest tests/test_cad.py -v
```
