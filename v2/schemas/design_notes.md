# Terrain Data Model - Design Notes

## User's Requirements (verbatim)

In terms of "types" I think we would want one thing for physical terrain
objects (this could be a physical building, or a physical crate, or a
physical wall panel from a modular terrain kit, or an acrylic rectangle that is
sometimes used for a ruin's base). There would probably also need to be another
type for a more granular geometrical representation, typically just dimensions
of e.g. a rectangular prism, but in future versions maybe more freeform shapes
like oblong blobs; a physical terrain object might have several of these or
just one. Then we also need a "type" for a "terrain feature" -- this is a
technical term in Warhammer 40k representing e.g. a Ruin, a Crater, a Woods,
etc. It might be composed of multiple physical objects combined together, such
as one of those objects atop an acrylic ruin base, or several modular ruin
bits, or etc. We need a "type" for a terrain layout, which contains zero or
more terrain features. We need a "type" for a terrain catalogue, indicating the
set of terrain objects owned by someone (potentially a huge collection, each
initialized to infinite quantity, if we're in tabletop simulator).

## Type Hierarchy

```
GeometricShape          (collision volume: a rectangular prism for now)
    ↑ has many
TerrainObject        (a physical object you can hold: building, crate, wall panel, base)
    ↑ has many
TerrainFeature          (a 40k rules concept: Ruin, Obstacle, Woods, Crater, etc.)
    ↑ has many
TerrainLayout           (a complete table: placed features with positions)

TerrainCatalog          (what someone owns: objects with quantities)
```

## How This Maps to v1

| v2 concept        | v1 equivalent                                      |
| ----------------- | -------------------------------------------------- |
| GeometricShape    | TerrainAbstraction (rect with blocksLoS, etc.)     |
| TerrainObject  | (implicit -- e.g. a single container, a wall tile) |
| TerrainFeature    | TerrainFeature subclasses (Ruin, Container, etc.)  |
| TerrainLayout     | Map (list of terrain features + table dimensions)  |
| TerrainCatalog    | (hardcoded -- only Eddie's ruins + Strix containers)|

In v1, TerrainObject wasn't an explicit concept. A ContainerFeature just
*was* a container (or stack of two). The object/feature distinction matters
when you have modular kits where the same wall panels can form different ruins.

## Design Considerations

### Where do game-rule properties live?

GeometricShapes carry `opacity_height_inches`: the height up to which the
volume blocks line of sight. This is distinct from the shape's physical height
-- a ruin wall might be 6" tall physically but only opaque up to 3" (broken
upper section). `null` or `0` means fully transparent (acrylic bases, open
floors). A value >= physical height means fully opaque (solid containers,
intact walls).

Movement blocking is derived from the shape's physical height rather than
being a separate property -- anything with sufficient height blocks movement.
The threshold for "blocks movement" is an engine/rules concern, not a
per-shape declaration.

"Obscuring" (blocks LoS only from outside the feature, per 40k Ruins rules)
is a feature-level rules concept determined by `feature_type`, not a
per-shape property. If `feature_type` is `"ruin"`, the engine knows to apply
obscuring semantics. This keeps shapes purely geometric.

### Object vs. Feature: who owns the shapes?

A TerrainObject has its own shapes (e.g. a container is a 2.5" x 5" x 5"
prism). When an object is placed inside a feature, its shapes are positioned
relative to the feature's origin via the component's transform. The engine
computes the feature's total collision footprint by combining all its
component objects' shapes.

This means the engine doesn't need a separate "feature shape" -- it derives
the feature's geometry from its objects. (v1 did it the other way: features
directly declared their abstractions, and there was no object concept.)

### Procedurally generated features (ruins)

v1 generates ruin shapes procedurally: random length/width, random wall
configuration. This is an engine concern, not a JSON-spec concern. A
procedurally generated ruin, once generated, can be serialized as a
TerrainFeature with specific objects placed at specific offsets. The
randomness is in how the engine assembles the feature, not in the data format.

For a modular ruin kit, the catalog would list the individual wall panels and
floor tiles as objects. The engine (or a feature-assembly step) combines
them into a ruin feature. For a non-modular ruin (a single cast piece), the
object IS the feature (one component, identity transform).

### What about visual / asset information?

The schema deliberately excludes visual info (mesh URLs, textures). The engine
doesn't need it. A separate "asset mapping" file could map object IDs to
visual assets for rendering in TTS or a viewer. This keeps the engine schema
clean and focused on geometry and game rules.

### Feature templates / recipes

The schema as drafted describes *instances* -- a specific feature with specific
objects at specific positions. It doesn't describe *templates* (e.g. "a
Ruin can be assembled from these objects in these ways"). Templates are an
engine/configuration concern and might be better expressed in code or a
separate config format. The JSON schema handles the output: what was generated.

### Catalog quantity semantics

`quantity: null` means unlimited (TTS mode). `quantity: 0` means "I own this
but it's all in use" (maybe useful for multi-layout scenarios). The engine
should respect quantities when selecting objects to place.

### Coordinate system

The schema uses inches (matching 40k conventions). Origin is at center of
table for layouts, center of feature for objects/shapes. Y is vertical
(height), X and Z are horizontal. Rotation is Y-axis only (pieces sit flat
on the table), in degrees.

Open question from PLAN.md: whether to use floats or fixed-point integers
for positions. The schema uses `"type": "number"` which allows both.
