# Parametric CAD

Mechanical parts for the desk-sorting arm, modelled in
[build123d](https://build123d.readthedocs.io/) (a Python CAD library over the
OpenCASCADE kernel).

## The rule: no dimensions live here

Every part in this package derives its dimensions from `src/geometry.py`:

- **`DEFAULT_ARM`** (`ArmGeometry`) — kinematic lengths, the base height
  budget, the base's position on the desk, and the mass estimates the mount is
  sized against.
- **`DEFAULT_HARDWARE`** (`HardwareSpec`) — off-the-shelf component envelopes:
  the DS3218 servo, the 608ZZ bearing, the M8 desk clamp, print clearances.

No module in `cad/` declares a physical dimension of its own. This is the same
discipline that keeps `arm_chain.py` building its ikpy chain programmatically
instead of from a URDF file: one number, one place, and a change propagates
atomically instead of drifting.

The practical consequence: to make the clamp fit a different servo or a thicker
desk, edit `src/geometry.py`. Do not edit `cad/base_pedestal.py`.

## Regenerating STLs

```bash
python3 -m cad.base_pedestal               # -> cad/output/base_pedestal.stl
python3 -m cad.desk_clamp_knob             # -> cad/output/desk_clamp_knob.stl
python3 -m cad.desk_clamp_pressure_foot    # -> cad/output/desk_clamp_pressure_foot.stl
python3 -m cad.assembly_preview            # -> assembly_preview.stl + .png
```

Each module takes `--report` (print resolved dimensions, export nothing) and
`--output PATH`. Run `--report` after changing anything in `src/geometry.py` to
see what moved.

---

## Why a monolithic U-clamp

Session D.1 bolted a flange to the desk through four M4 holes — which meant
drilling the desk. D.1b replaced that with a clamp, but built it as a cylinder
with a small horizontal wing bolted down beside it: a tower with a side arm,
not a clamp. **Session D.1c is a full rewrite as a single C-profile body**, the
shape a monitor-arm clamp actually uses:

```
                      +----------+  z = +70   bearing seat in the top face
                      |  servo   |
                      |  turret  |
      +---------------+----------+---------+  z = +15   top arm
      |///|    pads       cavity opening   |  z =   0   desk surface
      |///+-------------------------------+
      |///|                                |
      |sp |          throat (desk)         |
      |ine|                                |
      |///+-------------------------------+  z = -45   bottom arm top
      |///|         nut pocket    O screw  |
      +---+-------------------------------+  z = -60   bottom arm underside
       -50  -35    -30                   +30
                    ^ desk seats here
```

Everything is one solid: top arm, spine, bottom arm, servo turret and both
gussets. There is no bolted joint anywhere in the load path.

The servo/bearing internals — cavity, retention shelf, shaft bore, seat — are
carried over from D.1 unchanged; `test_pedestal_internals_survived_the_u_clamp_rewrite`
pins those dimensions so a future edit cannot quietly disturb them.

---

## Three specified numbers that could not be built

The design-rule checks and a cross-section render caught three physical
impossibilities in the D.1c brief. All are documented here rather than
silently worked around.

### 1. A 15 mm top arm cannot contain a 40.5 mm servo

`top_arm_thickness_mm = 15.0` was specified as including the servo cavity. A
DS3218 is 40.5 mm tall and needs **55.5 mm** of housing once print clearance,
the shaft boss, a ceiling and the bearing seat are counted.

**Resolved** by keeping the arm at 15 mm and raising a **turret** from it to
the full 70 mm pedestal height. This is what the brief's own sketch showed
(the servo box is drawn above the arm line), it preserves the thin C-profile,
and the 70 mm budget from `base_height_mm − BaseStack.allowance_mm` accepts it
exactly. Thickening the arm instead would have made a slab and tripled its
material.

### 2. The servo cavity and the desk pad both wanted the same face

The cavity has to open on the top arm's underside so the servo can be inserted
from below onto its retention shelf. But that face is also what rests on the
desk and carries the anti-slip pad. A 40 × 40 mm pad has nowhere to go: the
opening lands mid-face, leaving 15.75 mm strips.

**Resolved** with **two pad strips flanking the opening**, 15.75 × 78.50 mm
each — 2473 mm² total, more contact than the single 40 × 40 pad (1600 mm²) it
replaced, and better placed, since it straddles the yaw axis instead of sitting
to one side.

### 3. A 20 mm pad recess cannot sit on an 8 mm screw tip

**Resolved** with a small printed **pressure foot** (`desk_clamp_pressure_foot.py`)
that threads onto the screw's tip and carries the pad. See below.

### And one the cross-section caught

The upper gusset hangs into the throat beside the spine, so the desk's edge
comes to rest **against the gusset, not the spine**. With the spine placed at
the nominal offset, the yaw axis would have ended up 25 mm from the desk edge
rather than the specified 30 mm.

**Resolved** by treating the gusset line as the real desk seating plane and
placing the spine one gusset further out (`spine_inner_x = desk_seat_x −
gusset_size`). `test_the_gusset_defines_where_the_desk_stops` holds it.

---

## Parts

### `base_pedestal.py` — the U-clamp

| | |
|---|---|
| Overall envelope | 80.00 (X) × 82.50 (Y) × 130.00 (Z) mm |
| Desk seats at | x = −30.00 mm (yaw axis at x = 0) |
| Top arm | x −35 … +30, z 0 … 15 |
| Spine | x −50 … −35, z −60 … +15 |
| Bottom arm | x −35 … +30, z −60 … −45 |
| Throat | 45.0 mm, for 15–35 mm desks |
| Servo turret | x ±14.95, y −41.25 … +21.25, z 15 … 70 |
| Servo cavity | 20.50 (X) × 40.50 (Y) body, 54.50 (Y) ear slot |
| Bearing seat | Ø21.90 × 6.50 mm deep |
| Anti-slip pads | 2 × 15.75 × 78.50 mm, 2.0 mm deep (2473 mm²) |
| Clamp screw | Ø9.00 through the bottom arm at x = −16.00 |
| Volume | ≈ 285 cm³ |

**Coordinate frame.** The origin is on the yaw axis at the **desk's top
surface**, so `z = 0` is the desk plane and `+X` points inward over the desk.
This is deliberate: `ArmGeometry.base_height_mm` is measured from the same
datum, so the clamp, the kinematics and the assembly preview share one
coordinate system with no conversion step. The exported STL therefore straddles
z = 0 rather than sitting on the bed — slicers drop it on import.

**Why the servo's long axis runs across the clamp.** Along the arm it does not
fit: the ear slot is 54.5 mm and the body sits 10 mm off the shaft, so it would
reach 37.25 mm from the axis and break out past a spine only 30 mm away. Across
the clamp, the arm needs to be just 20.5 mm deep for it. This is what sets the
clamp's 82.5 mm width.

**Why the screw sits outboard, not under the yaw axis.** Placing it as far
toward the desk edge as the pressure foot allows shortens the bottom arm's
cantilever (19 mm instead of 30) *and* lengthens the lever it resists tipping
with (46 mm instead of 30) — both improve at once.

**Why gussets rather than fillets.** A swept fillet is the most fragile
operation to re-run when an upstream dimension moves, and every dimension here
is expected to move. Triangular gussets give most of the stress relief and
always regenerate.

### `desk_clamp_pressure_foot.py`

| | |
|---|---|
| Foot | Ø24.00 × 6.02 mm |
| Swivel seat (underside) | Ø9.00 mouth → Ø2.00 flat, 2.02 mm deep, 120° included |
| Web | 2.00 mm (compression only) |
| Pad recess (top) | Ø20.00 × 2.00 mm deep |
| Contact area | 314 mm², **6.2× a bare M8 tip** |

Rests on the screw's tip and bears on the desk's underside. Its height is not a
free choice: at the thickest supported desk the throat leaves exactly 10 mm
between the bottom arm and the desk, and the foot has to fit inside that.
`validate()` enforces it.

**Revised in D.1d: it swivels, it does not thread on.** The first version had a
tapping bore, so the foot turned with the screw. That drags the rubber pad
across the desk as you tighten — scuffing the finish — and moves the friction
that resists tightening out to the pad's radius, which roughly halves the
preload a given hand torque produces. Seated on a truncated cone the foot is
free to stay still while the screw turns inside it. The cone's apex is
truncated because a sharp internal point is unprintable and tessellates to
degenerate triangles.

The foot is not captive: it rests on the tip and drops off if the assembly is
inverted. Fit it as the clamp goes on; once the pad meets the desk it stays put.

### `desk_clamp_knob.py`

| | |
|---|---|
| Body | Ø50.00 × 20.00 mm |
| Hex socket (top) | 13.10 mm across flats × 7.80 mm deep |
| Shank bore | Ø9.00 × 12.20 mm |
| Grip flutes | 12 × Ø8.00 mm scallops |

**Why it stayed 50 mm.** The knob was scheduled to shrink to 30 mm, on the
grounds that a large knob lets a hand exceed the printed jaw's 3.9 N·m limit.
That limit belonged to D.1b's cantilever wing. The U-clamp's bottom arm takes
**11.62 N·m** and a 50 mm knob reaches only 1.00 N·m — 11.6× of headroom — so
there is nothing left to protect against, while shrinking would have cut the
grip margin from 2.39× to 1.43×. The shrink was cancelled.

**The bearing boss is gone (D.1d).** It was added in D.1b, where the knob bore
against the upper jaw and collar friction there ate much of the hand torque. In
the U-clamp the knob touches nothing but the screw — the assembly preview shows
it hanging 28 mm below the bottom arm — so the boss was inert material and the
friction term it justified was modelling a contact that does not exist.

Print **socket-face down**: no support needed and the gripping surfaces come
out crisp.

---

## Clamp mechanics

| | | revised in D.1d |
|---|---|---|
| Arm tipping moment (worst case) | 6.76 N·m | unchanged |
| Tipping lever (top arm edge → screw) | 46.00 mm | unchanged |
| Preload needed | **147 N** | unchanged (0.28 N·m at the knob, was 0.42) |
| Hand torque available, Ø50 knob | 1.00 N·m | unchanged |
| Preload it produces | **516 N** | ← was 350 N |
| **Grip margin** | **3.51×** | ← was 2.39× |
| Bottom arm structural limit | 7.89 N·m | ← was 11.62 N·m |

**Why those three numbers moved.** The friction model charged collar friction
at the knob boss's radius, about 6.75 mm, for a contact the U-clamp does not
have. D.1d moved it to where the rubbing actually happens — the screw's tip
turning in the pressure foot's cone, contact radius 3.20 mm — which roughly
halves the friction lever and so raises the preload a given torque produces.
The structural limit falls in the same proportion, because it is expressed as a
*torque* and the same torque now delivers more force; the *force* the bottom
arm can carry is unchanged at 4834 N. Hand torque still cannot reach it.

**The U-profile inverts the old design's weakness.** D.1b's side wing yielded
at 3.9 N·m — below what a hand could apply — so `cad/README.md` had to carry a
"do not use a wrench" warning. A 15 mm bottom arm spanning the full 82.5 mm
width takes 11.62 N·m, which no hand can reach through a knob this size. That
is a structural guarantee rather than an instruction, and
`test_hand_cannot_overstress_the_u_clamp` holds it.

The tipping moment above is deliberately pessimistic: it puts the whole arm
mass plus a full payload at maximum reach. `docs/PROOF_OF_CONCEPT.md` §2.2
computes ≈ 3.4 N·m on a real centre-of-mass basis, so the true grip margin is
roughly double the figure quoted.

---

## Design rule checks

Every part's `validate()` re-derives each clearance and refuses to build
something that would print badly, raising a `DesignRuleError` subclass carrying
a `DesignStatus` from `cad/_design.py`.

| Check | `DesignStatus` on failure |
|---|---|
| **U-clamp** | |
| Base height budget leaves room for the turret | `NEGATIVE_HEIGHT` |
| Vertical order: top arm < ear shelf < cavity ceiling < bearing seat | `FEATURE_COLLISION` |
| Ceiling between cavity and bearing seat ≥ `min_wall_thickness_mm` | `WALL_TOO_THIN` |
| Turret walls clear the cavity *and* the bearing seat | `WALL_TOO_THIN` |
| Turret sits within the top arm, not overhanging it | `FEATURE_COLLISION` |
| Spine outer < spine inner ≤ desk seating plane | `INVALID_PARAMETER` |
| Gussets do not close the throat from both sides | `FEATURE_COLLISION` |
| Pad strips have area left between cavity and gusset | `FEATURE_COLLISION` |
| Screw is inboard of the desk seat; foot stays under the desk | `FEATURE_COLLISION` |
| Floor above the nut pocket ≥ `min_wall_thickness_mm` | `WALL_TOO_THIN` |
| Nut pocket clears the spine and the clamp's sides | `WALL_TOO_THIN` |
| M3 pilot holes land in shelf material | `FEATURE_COLLISION` |
| Screw long enough to reach the *thinnest* supported desk | `FASTENER_TOO_SHORT` |
| **Pressure foot** | |
| Height matches bore + web + pad recess | `INVALID_PARAMETER` |
| Foot fits the throat at the thickest desk | `FEATURE_COLLISION` |
| Bore inside pad recess inside foot | `FEATURE_COLLISION` |
| **Knob** | |
| Hex socket does not pierce the knob | `FEATURE_COLLISION` |
| Bearing boss leaves a wall around the bore | `WALL_TOO_THIN` |
| Flutes do not cut into the socket or reach the boss | `WALL_TOO_THIN` |

Note the screw-length check keys off the **thinnest** desk, not the thickest: a
thin desk sits high in the throat, so its underside is furthest from the bottom
arm and the screw has to reach hardest.

---

## Assembly

### 1. Print the three parts

| Part | Orientation | Notes |
|---|---|---|
| `base_pedestal` | Spine face down on the bed | The U's opening faces sideways; no supports needed for the throat |
| `desk_clamp_knob` | Hex socket **down** | Crisp socket, no supports |
| `desk_clamp_pressure_foot` | Pad recess **up** | Bore bridges cleanly |

PETG throughout, ≥ 4 perimeters. The design assumes 4 × 0.4 mm walls
(`min_wall_thickness_mm = 4.0`).

### 2. Glue in the anti-slip pads

Cut from 2 mm rubber sheet:

- **Two 15.75 × 78.50 mm strips** for the top arm's underside
- **One Ø20 mm disc** for the pressure foot's top face

Degrease both rubber and printed recess with isopropyl alcohol, then bond with
contact adhesive or cyanoacrylate. The recesses are 2.0 mm deep and the sheet
is 2.0 mm, so a correctly seated pad sits **flush** — `DeskClampSpec` refuses a
pad thicker than its recess for exactly this reason.

### 3. Press the M8 nut into the bottom arm

The hex pocket is in the bottom arm's **underside**. It should need a firm
push. If it spins, a drop of epoxy locks it; it only ever resists the screw's
thread friction.

### 4. Fit the knob and the foot to the screw

Press the M8 × 70 screw's hex head into the knob's socket until it bottoms out,
7.80 mm down. The pressure foot simply **rests** on the other end: its conical
seat centres on the screw's tip. It is not captive — hold it in place while
positioning the clamp, and it stays put once the pad meets the desk.

### 5. Thread the screw up through the bottom arm

From below: knob at the bottom, screw rising through the captive nut, foot on
top pointing at the desk. Wind it down so the foot sits low in the throat.

### 6. Clamp to the desk

1. Slide the clamp onto the desk edge until the desk stops against the gussets.
2. Check both pad strips are fully on the desk and the foot is under solid desk.
3. Turn the knob until firm. The bottom arm cannot be over-stressed by hand, so
   "firm" is enough — there is no torque warning to observe on this design.

---

## Assembly preview

```bash
python3 -m cad.assembly_preview                      # STL + PNG
python3 -m cad.assembly_preview --report             # summary only
python3 -m cad.assembly_preview --desk-thickness 35  # any desk in range
```

![Assembly preview](output/assembly_preview.png)

Every part placed on a 1200 × 600 mm desk at the position `ArmGeometry`
actually specifies, with the arm's links drawn as placeholder cylinders at
their zero pose. Individual parts each pass their own design rules, but
nothing else checks that they *fit together on a real desk* — this does, and
`tests/test_cad.py` asserts the same properties numerically rather than
relying on the picture.

| | |
|---|---|
| Yaw axis on desk | (600.0, 30.0) mm |
| Clamp footprint | 82.5 (X) × 80.0 (Y) mm |
| Knob hangs below the bottom arm | 24.8 mm (on a 25 mm desk) |
| Printed volume | 320.6 cm³ solid |
| Estimated filament | ≈ 142 g PETG at 35% infill |

The arm at zero pose reaches 980 mm along +Y, well past the desk's 600 mm
depth. That is correct rather than alarming: the zero pose is a fully extended
horizontal reference posture, not a working position.

The scene is exported as a single STL with each solid left as its own closed
shell — a scene, not a manufacturable part. Rendering uses matplotlib over the
tessellated mesh, because build123d's own viewers need a live GUI session and
are unusable from a script or in CI.

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
list every run. **Measure a real DS3218 before printing for final assembly** —
`flange_span_mm` now sets the clamp's whole width, and
`shaft_offset_from_body_end_mm` shifts the cavity within it.

Also confirm which **travel variant** was ordered. The DS3218 ships in 180° and
270° versions; `ArmGeometry`'s base yaw limit is ±135° (a 270° span), which the
180° variant cannot deliver.

### Clamp fastener dimensions — verified, with one caveat

| | Value | Source |
|---|---|---|
| M8 coarse thread pitch | 1.25 mm | ISO 261 |
| Hex head across flats / height | 13.00 / 5.30 mm | DIN 933 / ISO 4017 |
| Clearance hole (medium series) | 9.0 mm | ISO 273 |
| Nut across flats | 13.00 mm max | DIN 934 |
| **Nut thickness** | **6.80 mm max** | **DIN EN ISO 4032** |

Legacy DIN 934 tables give m = 6.5 mm max for M8; the current DIN EN ISO 4032
revision gives 6.80 max / 6.44 min. **Both are sold as "DIN 934."** A pocket cut
to 6.5 mm would not seat a modern nut, so `nut_pocket_depth_mm` derives from the
larger figure.

---

## Known reach caveat

Moving the base 30 mm inward improves the worst desk corner from 848.5 mm to
**827.6 mm**. That still exceeds `ArmGeometry.safe_reach_mm` (807.5 mm, a
conservative 85% of full extension), but `docs/PROOF_OF_CONCEPT.md` §2.1 sized
the links against 89%, and 827.6 mm is 87.1% — inside the envelope the arm was
actually designed for. `coverage_report()` therefore prints
"Full-desk reachable? False" while the arm is fine.
`test_worst_corner_sits_between_the_two_reach_thresholds` documents the
disagreement and fails if a future change pushes the corner past 89%.

---

## Known future work

- **The clamp is over-material.** At ≈ 285 cm³ it is mostly solid; lightening
  pockets in the spine and arms would cut print time substantially. Deferred
  until the shoulder loads from Session D.3 are known.
- **`BaseStack` is provisional.** The 6 mm turntable plate and 24 mm shoulder
  bracket rise are estimates. Sessions D.2 and D.3 replace them; the turret
  height then follows automatically.
- **The stress model is a hand calculation.** A rectangular-section cantilever
  with a point load is a reasonable first approximation, but it ignores the
  spine's restraint and PETG's anisotropy between layers. If the clamp is ever
  asked to carry more, this deserves FEA rather than a bigger safety factor.

## Testing

`tests/test_cad.py` covers hardware-spec validation, clamp physics, parameter
derivation for all three parts, every design rule check, solid construction,
and STL integrity. The mesh checks parse each exported STL directly rather than
trusting the kernel, because "watertight" is a property of the tessellation a
slicer will read:

- every **undirected** edge used by exactly two faces (closed surface)
- every **directed** edge traversed exactly once (consistent winding)
- enclosed volume positive, and within 1% of the kernel's B-rep volume

```bash
python3 -m pytest tests/test_cad.py -v
```
