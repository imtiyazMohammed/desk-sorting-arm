# Parametric CAD

Mechanical parts for the desk-sorting arm, modelled in
[build123d](https://build123d.readthedocs.io/) (a Python CAD library over the
OpenCASCADE kernel).

## The rule: no dimensions live here

Every part in this package derives its dimensions from `src/geometry.py`:

- **`DEFAULT_ARM`** (`ArmGeometry`) — kinematic lengths, the base height
  budget, and the mass estimates the mount is sized against.
- **`DEFAULT_HARDWARE`** (`HardwareSpec`) — off-the-shelf component envelopes:
  the DS3218 servo, the 608ZZ bearing, the M8 desk clamp, print clearances.

No module in `cad/` declares a physical dimension of its own. This is the same
discipline that keeps `arm_chain.py` building its ikpy chain programmatically
instead of from a URDF file: one number, one place, and a change propagates
atomically instead of drifting.

The practical consequence: to make the pedestal fit a different servo, edit
`ServoSpec` in `src/geometry.py`. Do not edit `cad/base_pedestal.py`.

## Regenerating STLs

```bash
python3 -m cad.base_pedestal            # -> cad/output/base_pedestal.stl
python3 -m cad.desk_clamp_lower_jaw     # -> cad/output/desk_clamp_lower_jaw.stl
python3 -m cad.desk_clamp_knob          # -> cad/output/desk_clamp_knob.stl
```

Each module takes `--report` (print resolved dimensions, export nothing) and
`--output PATH`. Run `--report` after changing anything in `src/geometry.py` to
see what moved.

## Design rule checks

Every part's `validate()` is not a formality. It re-derives each clearance and
refuses to build something that would print badly, raising a
`DesignRuleError` subclass carrying a `DesignStatus` from `cad/_design.py`.

| Check | `DesignStatus` on failure |
|---|---|
| **Pedestal** | |
| Base height budget leaves room for a pedestal | `NEGATIVE_HEIGHT` |
| Vertical layout ordered: jaw < ear shelf < cavity ceiling < bearing seat | `FEATURE_COLLISION` |
| Ceiling between cavity and bearing seat ≥ `min_wall_thickness_mm` | `WALL_TOO_THIN` |
| Radial wall outside the ear slot ≥ `min_wall_thickness_mm` | `WALL_TOO_THIN` |
| Bearing seat does not undercut the body wall | `WALL_TOO_THIN` |
| Shaft bore narrower than the bearing seat (so the seat has a floor) | `FEATURE_COLLISION` |
| Pad recess clear of the open servo cavity | `FEATURE_COLLISION` |
| Clamp screw hole outboard of the pad, inside the jaw, clear of its edges | `WALL_TOO_THIN` |
| Knob clears the pedestal body | `FEATURE_COLLISION` |
| M3 pilot holes land in shelf material and do not break through | `FEATURE_COLLISION` |
| Clamp screw long enough for the whole stack at max throat | `FASTENER_TOO_SHORT` |
| **Lower jaw** | |
| Nut pocket and pad recess do not meet | `FEATURE_COLLISION` |
| Load-bearing web ≥ `jaw_thickness_mm` | `WALL_TOO_THIN` |
| Bolt hole narrower than the nut pocket (so the nut has a shoulder) | `FEATURE_COLLISION` |
| Plate wraps the hex pocket's *corners*, not just the bolt hole | `WALL_TOO_THIN` |
| Pad clears the nut pocket | `FEATURE_COLLISION` |
| **Knob** | |
| Hex socket does not pierce the knob | `FEATURE_COLLISION` |
| Bore narrower than the socket (so the head has a shoulder) | `FEATURE_COLLISION` |
| Bearing boss leaves a wall around the bore | `WALL_TOO_THIN` |
| Flutes do not cut into the hex socket or reach the boss | `WALL_TOO_THIN` |

This is what makes the modules safe to drive from swept parameters: a violating
combination fails loudly instead of quietly emitting a part with a 0.2 mm wall.
It also caught two real errors during Session D.1b — an M8 × 80 screw that
could not span the clamp stack, and a bearing boss too small to keep 4 mm of
wall around its bore.

---

## Why a clamp instead of bolts

Session D.1 bolted a flange to the desk through four M4 holes on a 60 mm bolt
circle. Session D.1b replaced that entirely:

- **No drilling.** The desk is not modified in any way.
- **Repositionable.** The arm can be moved along the desk edge, taken off, or
  transferred to another desk, in about ten seconds.
- **Simpler part.** The old design needed four vertical hex-key access channels
  bored through the body, because the 60 mm bolt circle fell *underneath* the
  85 mm body wall where no driver could reach. All of that is gone.

The pedestal's internals — servo cavity, retention shelf, shaft bore, bearing
seat — are **unchanged**. That geometry was good; only the mounting changed.
`test_pedestal_internals_survived_the_clamp_redesign` pins those dimensions so
a future edit cannot quietly disturb them.

The cable slot moved from +X to −X so the servo lead exits away from the clamp
rather than over it.

---

## Parts

### `base_pedestal.py` — pedestal and clamp upper jaw

```
                    ┌───────────────┐  z = 70.00  top face
                    │   ▓▓▓▓▓▓▓▓▓   │             608ZZ seat, Ø21.90 × 6.50 deep
                    ├───┬───────┬───┤  z = 63.50  seat floor (4.00 mm ceiling)
                    │   │  ███  │   │             shaft bore, Ø13.50
                    ├───┴───────┴───┤  z = 59.50  cavity ceiling
                    │  ░░░░░░░░░░   │             servo body pocket, 40.50 × 20.50
       M3 pilots →  ├──┬─────────┬──┤  z = 49.50  ear shelf  ← servo ears seat here
                    │  ░░░░░░░░░░░  │             ear slot, 54.50 × 20.50
       cable ←──────┤  ░░░░░░░░░░░  │             (slot exits −X)
                    │  ░░░░░░░░░░░  │
        ╔═══════════╧═══════════════╧══════════════╗  z = 12.00  upper jaw top
        ║      ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒               ○   ║             ▒ = pad recess
        ╚══════════════════════════════════════════╝  z =  0.00  desk surface
         x=21.25 ────── pad 50×50 ────── 71.25   85.75 ← M8 hole
                                    └ desk edge ┘
                          Ø85.27 body      jaw reach 94.25
```

**Resolved dimensions** (from the current `src/geometry.py`):

| | |
|---|---|
| Total height | 70.00 mm (`base_height_mm` 100 − `BaseStack.allowance_mm` 30) |
| Body OD | 85.27 mm (derived) |
| Upper jaw | 94.25 mm reach × 50.00 wide × 12.00 thick |
| Pad recess | 50 × 50 × 2.0 mm deep, x = 21.25 … 71.25 |
| Clamp screw hole | Ø9.00 mm at x = 85.75 |
| Desk edge window | x = 71.25 … 81.25 (10 mm wide) |
| Servo cavity | 40.50 × 20.50 mm body, 54.50 × 20.50 mm ear slot |
| Bearing seat | Ø21.90 × 6.50 mm deep |
| Volume | ≈ 359 cm³ |

#### The desk edge window

The desk edge must fall in the 10 mm band between the pad's outer edge and the
clamp screw hole. That band is the positioning tolerance you get when siting
the arm — wider would be friendlier, but it lengthens the jaw's unsupported
overhang, which is exactly what limits how hard the knob may be tightened.

#### Why the cavity is offset

The DS3218's output shaft is not centred on its body. For the *shaft* to sit on
the yaw axis — which it must, since that shaft defines the axis — the body has
to sit 10 mm off-centre. That offset, not the servo's size, is what drives the
pedestal's diameter: it pushes the far corner of the ear slot out to r ≈ 38.6 mm,
and one 4 mm wall beyond that gives the 85 mm body.

#### Why the bearing stands proud

The 608ZZ's outer race presses into the top seat; the yaw turntable above
clamps its **inner** race. The seat is cut 0.5 mm shallower than the bearing is
wide so the bearing stands proud of the printed top face — otherwise the
turntable would scrub on the plastic instead of turning on the bearing, and
the whole point of fitting a thrust bearing (taking axial load off the servo's
output shaft) is lost.

### `desk_clamp_lower_jaw.py`

A plate that goes under the desk, 70.29 × 40.00 × 18.80 mm. Its origin is on
the clamp screw axis; it extends in −X, mirroring the pedestal's upper jaw.

| | |
|---|---|
| Nut pocket (underside) | 13.50 mm across flats × 6.80 mm deep |
| Pad recess (top) | 40 × 40 × 2.0 mm deep, starting 14.50 mm from the bolt |
| Bolt hole | Ø9.00 mm, full height |
| Load-bearing web | 10.00 mm |

The pad stands back 14.50 mm from the bolt — the *widest* offset the desk edge
can take — so it sits under solid desk wherever in the window you site the arm,
not only when you place it perfectly.

**Why the plate is 18.80 mm and not 10 mm.** `jaw_thickness_mm = 10.0` is the
*structural* thickness. A 2 mm pad recess and a 6.8 mm nut pocket cut into
opposite faces of a 10 mm plate would leave a 1.2 mm web carrying the entire
clamp load, so the recesses are additional: 10 + 2 + 6.8 = 18.8.
`test_load_bearing_web_meets_the_structural_thickness` holds that invariant.

### `desk_clamp_knob.py`

A 50 mm fluted disc, 22.00 mm tall including its bearing boss.

| | |
|---|---|
| Body | Ø50.00 × 20.00 mm |
| Bearing boss | Ø18.00 × 2.00 mm |
| Hex socket (top) | 13.10 mm across flats × 7.80 mm deep |
| Shank bore | Ø9.00 × 14.20 mm |
| Grip flutes | 12 × Ø8.00 mm scallops, cutting to r = 21.00 mm |

**The boss exists for friction control.** Torque spent rubbing the knob's face
against the jaw never becomes clamping force, and that loss scales with contact
radius. A full 50 mm face would swallow most of the hand torque; confining
contact to an 18 mm collar keeps it useful. This is modelled explicitly in
`DeskClampSpec.torque_to_preload_factor_m`, and
`test_larger_boss_wastes_torque_on_collar_friction` pins the relationship.

Print **socket-face down**: the hex socket then needs no support and its
gripping surfaces come out crisp.

---

## ⚠ Tighten by hand only — do not use a wrench

The upper jaw is a cantilever overhanging the desk edge, and it is the weakest
link in the clamp:

| | |
|---|---|
| Preload needed to hold the arm down | **105 N** (0.28 N·m at the knob) |
| Jaw's allowable preload (PETG, safety factor 2) | **1437 N** |
| **Maximum safe knob torque** | **≈ 3.9 N·m** |
| Preload a firm 5 N·m tighten would produce | 1849 N — **over the limit** |

So there is roughly a 14× margin for the job the clamp actually has to do, and
the only real exposure is over-tightening. A 50 mm knob gives good leverage;
snug is enough, and past snug you are working against the part rather than the
desk. `test_max_safe_torque_is_documented_and_below_five_newton_metres` pins
this finding — if a future change makes the jaw strong enough for 5 N·m, that
test fails and this warning should be revisited.

---

## Assembly

### 1. Print all three parts

| Part | Orientation | Notes |
|---|---|---|
| `base_pedestal` | As modelled, underside on the bed | Servo cavity and pad recess print as bridges; no supports needed |
| `desk_clamp_lower_jaw` | Nut pocket **down** on the bed | Pocket needs no support this way |
| `desk_clamp_knob` | Hex socket **down** on the bed | Crisp socket, no supports |

PETG throughout, ≥ 4 perimeters. The design assumes 4 × 0.4 mm walls
(`min_wall_thickness_mm = 4.0`); printing thinner invalidates the torque limit
above.

### 2. Glue in the anti-slip pads

Cut two pads from 2 mm rubber sheet:

- **50 × 50 mm** for the pedestal's upper jaw (recess on the underside)
- **40 × 40 mm** for the lower jaw (recess on the top face)

Degrease both the rubber and the printed recess with isopropyl alcohol, then
bond with contact adhesive or cyanoacrylate. Press flat and let cure fully
before clamping anything.

The recesses are 2.0 mm deep and the sheet is 2.0 mm, so a correctly seated pad
sits **flush**. This matters: a proud pad would make the jaw rock on rubber
instead of bearing on the desk near its edge, which is exactly the support the
torque limit above assumes. `DeskClampSpec` refuses a pad thicker than its
recess for this reason.

### 3. Insert the M8 nut into the lower jaw

Press the nut into the hex pocket on the lower jaw's underside. It should need
a firm push. If it spins, a drop of epoxy locks it — it only ever needs to
resist the screw's thread friction.

### 4. Press the knob onto the screw head

Push the M8 × 90 screw's hex head into the socket in the knob's top face until
it bottoms out, 7.80 mm down. FDM prints internal features slightly undersize,
so this is normally a firm press fit. If it is loose, epoxy the head in.

### 5. Thread the assembly through the pedestal

Pass the screw down through the Ø9 hole in the pedestal's upper jaw, then
through the lower jaw, and start it into the captive nut. Leave the throat wide.

### 6. Mount to the desk

1. Open the throat past your desk's thickness (up to 45 mm).
2. Slide the assembly onto the desk edge until the pedestal's pad sits fully on
   the desk top and the desk edge falls inside the 10 mm window — between the
   pad's outer edge and the clamp screw.
3. Check the lower jaw's pad is under solid desk, not overhanging the edge.
4. Tighten the knob **by hand until snug**. See the torque warning above.

---

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
list every run. The generated STLs are geometrically valid and the design rule
checks all pass, but **measure a real DS3218 before printing for final
assembly** — `shaft_offset_from_body_end_mm` in particular shifts the entire
cavity, and `flange_span_mm` sets the body diameter.

Also confirm which **travel variant** was ordered. The DS3218 ships in 180° and
270° versions that are otherwise near-identical; `ArmGeometry`'s base yaw limit
is ±135° (a 270° span), which the 180° variant cannot deliver.

### Clamp fastener dimensions — verified, with one caveat

| | Value | Source |
|---|---|---|
| M8 coarse thread pitch | 1.25 mm | ISO 261 |
| Hex head across flats / height | 13.00 / 5.30 mm | DIN 933 / ISO 4017 |
| Clearance hole (medium series) | 9.0 mm | ISO 273 |
| Nut across flats | 13.00 mm max | DIN 934 |
| **Nut thickness** | **6.80 mm max** | **DIN EN ISO 4032** |

The nut thickness is version-dependent and worth knowing about. Legacy DIN 934
tables give m = 6.5 mm max for M8; the current DIN EN ISO 4032 revision gives
6.80 max / 6.44 min. **Both are sold as "DIN 934."** A pocket cut to 6.5 mm
would not seat a modern nut, so `nut_pocket_depth_mm` derives from the larger
figure. The 6.5 value is retained as `nut_thickness_nominal_mm` for reference
only, and `test_nut_pocket_uses_the_max_not_nominal_thickness` pins the choice.

---

## Known future work

- **The pedestal is over-material.** At ≈ 359 cm³ it is mostly solid; a lattice
  or shelled interior would cut print time and filament substantially. Deferred
  because wall topology should be decided once the shoulder loads from Session
  D.3 are known.
- **`BaseStack` is provisional.** The 6 mm turntable plate and 24 mm shoulder
  bracket rise are estimates. Sessions D.2 and D.3 replace them with measured
  values; the pedestal height then follows automatically.
- **No fillets.** Sharp internal corners are stress risers and print with
  elephant's foot. Filleting is deliberately deferred until the geometry stops
  moving, since fillet operations are the most fragile part of a build123d
  model to re-run against changed dimensions. The upper jaw's root would
  benefit most.
- **The jaw stress model is a hand calculation.** A rectangular-section
  cantilever with a point load is a reasonable first approximation, but it
  ignores the pedestal's stiffening effect at the root and PETG's anisotropy
  between layers. If the clamp is ever asked to carry more, this deserves FEA
  rather than a thicker safety factor.

## Testing

`tests/test_cad.py` covers hardware-spec validation, clamp physics, parameter
derivation for all three parts, every design rule check, solid construction,
and STL integrity. The mesh checks parse each exported STL directly rather than
trusting the kernel's own report, because "watertight" is a property of the
tessellation a slicer will read:

- every **undirected** edge used by exactly two faces (closed surface)
- every **directed** edge traversed exactly once (consistent winding)
- enclosed volume positive, and within 1% of the kernel's B-rep volume

```bash
python3 -m pytest tests/test_cad.py -v
```
