# Carltographer v2 - Original Design Document

> **Note**: This is the original design document from early development,
> preserved for historical context. The project has since been fully
> implemented — see the [README](../README.md) for current state and
> [CLAUDE.md](../CLAUDE.md) for development details. The "Open Questions"
> at the bottom have all been resolved.

## Original Notes (verbatim)

Some time ago, I started working on a random terrain layout generator for
Warhammer 40k. I did this in Lua, so that it could run inside Tabletop
Simulator, and create the terrain layouts directly in-game. But (1) I hate
working with Lua, and (2) the performance of the Lua interpreter in Tabletop
Simulator is atrocious, so the mildly compute-intensive algorithm was way too
slow. So, I would like to create a new project, primarily in Python (because I
understand the language well), and eventually have you translate the core bits
into rust for more performance if needed. I have the old code in the v1/
directory, and would like almost all the new work to be in the v2/ directory. I
would like the overall structure of the python code to be something like ... a
library that manages the state of a terrain layout and the iterative mutation of
it as we explore the space of possible terrain layouts, and a distinct blob of
python for things like organizing which pieces of terrain are usable (e.g.
because a person only owns a particular collection of terrain pieces) and the
UI. I would like the mutation engine section to be entirely deterministic (even
if it can use a prng), so that if I ask you to translate it into rust, we can
have separate scripts that can run the same procedure through the rust and
python versions with the same seed and verify that we get the same output. The
UI or whatever then could use either the pure python or compiled rust as its
engine, to get identical results (just performance differences). There will need
to be a specification, perhaps in JSON, describing terrain pieces and terrain
layouts, so that this can be passed around between programs (including
potentially a future Lua program that can load things into Tabletop Simulator).

## Thoughts on the v1 Code

The v1 implementation is a genetic algorithm for terrain placement. It maintains
a population of candidate maps, scores them on fitness criteria (piece counts in
desired ranges, no overlaps, minimum gap enforcement), and evolves them through
mutation (add/remove/move/rotate/modify pieces) and selection. Key domain
concepts:

- **Terrain features** are composite objects: each has a footprint for collision
  detection (TerrainAbstraction rectangles with properties like blocksLoS,
  blocksMovement), visual models (mesh references for TTS), and a transform.
- **Ruins** are procedurally generated: random rectangular footprints with
  randomly configured walls (doors, windows, solid walls, decorations) per
  floor.
- **Containers** come in fixed configurations (single, L, J, I shapes).
- **Validation** checks boundary encroachment, inter-feature gaps (1" general,
  6.2" for movement-blocking pairs), and self-collision for symmetric maps.
- **Scoring** penalizes exponentially for being outside desired piece-count
  ranges, rewards more pieces with diminishing returns.

The code is monolithic (~1900 lines) and tightly coupled to TTS (asset URLs,
spawn logic, UI XML). The core algorithm is buried in TTS-specific scaffolding.

## Architecture Sketch

### Two main boundaries

```
+------------------------------------------+
|  Engine (deterministic, portable)        |
|                                          |
|  - Terrain data model                    |
|  - Collision / validation                |
|  - Genetic algorithm (mutation,          |
|    scoring, selection)                   |
|  - Seeded PRNG only                      |
|  - JSON spec in, JSON layout out         |
|                                          |
|  Python now, Rust later (same results)   |
+------------------------------------------+
          |  JSON specs  |
          v              v
+------------------------------------------+
|  Frontend / Orchestration (Python only)  |
|                                          |
|  - Terrain collection management         |
|    (what pieces does a player own?)      |
|  - UI (TBD: CLI? TUI? web?)             |
|  - Calls engine, displays results        |
|  - Could swap in Rust engine via FFI     |
+------------------------------------------+
```

The engine consumes and produces JSON. It has no opinions about UI, asset URLs,
or what terrain a player owns. It just takes a specification of what terrain
types are available and their geometric properties, plus algorithm parameters,
and produces a layout.

### JSON Specifications

Two distinct specs seem necessary:

1. **Terrain Catalog**: Describes what terrain pieces exist and their geometric
   properties (footprint rectangles, LoS/movement blocking, possible
   configurations like the L/J/I container variants). This is the "what could go
   on a table" spec. It does NOT include visual/asset information -- that's a
   frontend concern.

2. **Layout**: Describes a specific arrangement of pieces on a table. Each entry
   references a piece from the catalog (by ID) and gives its transform. Plus
   any per-instance state like ruin wall configurations. This is the "what is on
   this table right now" spec.

A possible third spec or section of the catalog: **generation parameters**
(population size, iteration count, min/max piece counts, gap sizes, table
dimensions, symmetry). But this might just be function arguments rather than a
spec.

### Determinism Strategy

The engine must produce bit-identical results given the same seed. This means:

- All randomness comes from a single seeded PRNG (Python's `random.Random`,
  Rust's equivalent seeded with the same algorithm).
- No floating-point ambiguity: either use integer arithmetic for positions
  (e.g. positions in 0.5" increments represented as integers) or be very
  careful about float operations. The v1 code uses float inches. Integer
  grid-snapping might actually be a feature -- terrain placement precision
  beyond ~0.5" doesn't matter on a physical table.
- No hash-order dependence, no set iteration, no dict ordering surprises.
- No parallelism inside the engine (or deterministic parallelism, which is
  harder).

Cross-language determinism (Python vs Rust) is the hard part. The PRNG
algorithm itself needs to match. We will use **PCG32 (PCG-XSH-RR)**, a
permuted congruential generator, implemented from scratch in both languages.

#### Why PCG32

- **Statistical quality**: Passes TestU01 BigCrush. More than sufficient for
  terrain generation.
- **Simplicity**: The core is ~5 lines. 64-bit state, 32-bit output. One
  multiply, one add, a couple of shifts and a rotate.
- **Cross-language reproducibility**: The algorithm is fully specified with no
  platform-dependent behavior. Python needs explicit 64-bit masking (since its
  ints are arbitrary-precision), but that's just `& 0xFFFFFFFFFFFFFFFF` after
  each operation. Rust's `u64::wrapping_mul` / `wrapping_add` handles it
  natively.
- **Well-documented**: The original paper (O'Neill 2014) and reference
  implementation at https://www.pcg-random.org/ are clear and unambiguous.

#### PCG32 Specification

```
State:    uint64 (the evolving state)
Sequence: uint64 (the increment, must be odd; set once at init)

Multiplier constant: 6364136223846793005

advance(state):
    state = state * 6364136223846793005 + sequence
    return state

output(state):
    xorshifted = uint32(((state >> 18) ^ state) >> 27)
    rot = uint32(state >> 59)
    return (xorshifted >> rot) | (xorshifted << (32 - rot))    # 32-bit rotate

next():
    state = advance(state)
    return output(state)

seed(init_state, init_seq):
    sequence = (init_seq << 1) | 1    # must be odd
    state = 0
    state = advance(state)
    state = state + init_state
    state = advance(state)
```

For our purposes, we can fix the sequence to a constant (e.g. the default
`1442695040888963407`) and only vary the seed, simplifying the API to
`seed(uint64) -> generator`. The engine_params schema takes a single integer
seed.

### On Ruin Procedural Generation

The v1 ruin generation is interesting but complex. It lives inside the engine
because the genetic algorithm needs to be able to mutate ruins (re-randomize
their shape). The ruin's geometric footprint matters for collision, but the
visual details (which wall segments are doors vs windows) might be separable.

One option: the engine generates abstract ruin shapes (dimensions, which
corners have walls, opacity), and a separate "decorator" step (outside the
engine, or as a post-process) fills in the visual details. This would simplify
the engine and make ruin generation more modular. But it means the engine's
output is less complete.

Alternatively, keep ruin generation fully in the engine since wall
configuration affects LoS blocking (a gameplay-relevant property that the
scoring function might care about). This is what v1 does.

Probably keep it in the engine for now. Can refactor later.

### What the Engine API Looks Like (Roughly)

```python
# Very hand-wavy

catalog = TerrainCatalog.from_json("catalog.json")
params = GenerationParams(
    seed=42,
    table_width_inches=44,
    table_depth_inches=30,
    population_size=50,
    iterations=100,
    symmetric=True,
    ...
)

engine = Engine(catalog, params)
# Run the full GA, get the best layout
layout = engine.run()

# Or step through it
engine.init_population()
for i in range(100):
    engine.step()
best = engine.best_layout()

# Serialize
layout.to_json("layout.json")
```

### Open Questions (all resolved)

- **UI**: What form does the frontend take? CLI that dumps a JSON file? A
  little web viewer? A TUI with ASCII art? Probably start with CLI + JSON and
  add visualization later.
  > **Resolved**: Tkinter GUI with 2D top-down battlefield viewer, deployment
  > zone overlay, objective markers, and save/load. No CLI mode.

- **Terrain catalog scope**: v1 only has ruins and containers. Is the catalog
  format general enough for future terrain types (craters, barricades, woods)?
  Probably yes if we keep it abstract (rectangles with properties).
  > **Resolved**: Yes — the catalog format is general. Features have a
  > `feature_type` (obstacle, ruin, etc.) and are composed of objects with
  > rectangular prism shapes. Extensible to any terrain type.

- **Ruin detail level**: How much of the wall-by-wall configuration belongs in
  the engine vs. a post-processing step?
  > **Resolved**: Ruins are defined in the catalog as multi-component features
  > (objects with offsets). The engine treats them as collections of shapes for
  > collision/visibility. No procedural ruin generation in the engine.

- **Scoring extensibility**: The scoring function is very domain-specific (40k
  gap rules, desired piece counts). Should it be configurable/pluggable, or
  just hardcoded for now?
  > **Resolved**: Configurable. Two-phase scoring with user-settable visibility
  > targets (overall %, DZ %, cross-DZ %, objective hidability %) and per-metric
  > weights. Feature count preferences are also configurable per type.

- **Table coordinate system**: v1 uses float inches with the origin at center.
  Keep that, or switch to a grid? Grid simplifies determinism but loses
  placement precision. A fine grid (0.25" or 0.5") might be the best of both.
  > **Resolved**: Float inches with origin at center, quantized to 0.1"
  > increments. Rotations quantized to 15° increments. Determinism achieved
  > via PCG32 PRNG with explicit 64-bit masking, verified across Python and
  > Rust implementations.
