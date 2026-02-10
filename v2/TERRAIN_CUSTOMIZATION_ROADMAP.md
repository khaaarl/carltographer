# Terrain Customization & TTS Integration Roadmap

This document organizes the planned features for terrain catalog customization, dynamic assembly, manual placement, and Tabletop Simulator integration. Each feature has a unique ID for dependency tracking.

This is a living document — update the **Status** fields as features progress (states: **Not Started**, **In Progress**, **Done**). When a blocking feature is completed, annotate it as `(Done)` in downstream **Blocked by** fields.

---

## A. Catalog Persistence & Management

### A1. Catalog Persistence (Save/Load)

**Status:** Not Started
**Blocked by:** —

Externalize terrain catalogs as JSON files instead of hardcoding them in Python. The JSON schema already exists (`schemas/carltographer.schema.json`); this feature makes it the primary storage format.

- Save catalogs to user-accessible JSON files
- Load catalogs from JSON files at startup or on demand
- File location convention (e.g., `~/.carltographer/catalogs/` or a configurable path)
- Validation against the JSON schema on load

### A2. Built-in Starter Catalogs

**Status:** Not Started
**Blocked by:** A1

Ship the current hardcoded catalogs (WTC set, GW misc, etc.) as bundled JSON files that come with the application. Users can clone and customize them without losing the originals.

- Extract current Python-dict catalogs into JSON files
- Mark built-in catalogs as read-only (or clone-on-edit)
- UI for selecting which catalog(s) to use for generation

### A3. Catalog Import/Export

**Status:** Not Started
**Blocked by:** A1

Allow users to share catalog files with each other.

- Export a catalog to a standalone JSON file
- Import a catalog from a JSON file (with duplicate-ID detection and resolution)
- Potential for a simple catalog-sharing convention (e.g., post a `.json` file)

---

## B. Authoring & Tagging

### B1. Tag System

**Status:** Done
**Blocked by:** —

Add a flexible tagging system to terrain objects and features, complementing (not replacing) the existing `feature_type` field. Tags enable edition-neutral rules modeling — e.g., a feature tagged `obscuring` behaves correctly in both 10th and future editions without hardcoding edition-specific feature types.

- Tags on `TerrainObject` (e.g., `tall`, `dense`, `scalable`)
- Tags on `TerrainFeature` (e.g., `obscuring`, `defensible`, `unstable-ground`)
- Engine scoring and constraints can reference tags in addition to `feature_type`
- UI for managing tags (add/remove, autocomplete from known tags)

### B2. Tag-based Scoring & Constraints

**Status:** Not Started
**Blocked by:** B1 (Done)

Extend the engine's feature count preferences and visibility scoring to work with tags. For example, "at least 4 features tagged `obscuring`" rather than (or in addition to) "at least 4 features of type `ruins`."

- `FeatureCountPreference` accepts tag-based predicates
- Scoring targets can reference tags
- Backward-compatible: `feature_type`-based preferences still work

### B3. Custom Object Authoring

**Status:** Not Started
**Blocked by:** A1

UI for users to define their own terrain objects, representing physical pieces they own.

- Define objects as compositions of axis-aligned rectangular shapes (matching current `Shape` model)
- Set dimensions (width, depth, height) and offsets for multi-shape objects
- Name, ID, and tag assignment
- Quantity tracking ("I own 6 of these wall sections")

### B4. Custom Feature Authoring

**Status:** Not Started
**Blocked by:** B3

UI for users to compose terrain features from terrain objects.

- Select objects from the catalog and position them relative to each other
- Set feature type and tags
- Assign quantities
- Save to the active catalog

### B5. Piece Preview / Visualization

**Status:** Not Started
**Blocked by:** —

Show a top-down silhouette or outline of terrain objects and features in the catalog editor and selection UI, so users can see what they're building or choosing.

- Render object shapes as 2D outlines (top-down projection)
- Render multi-component features showing all objects in position
- Color-code by height or feature type
- Useful in both the authoring UI and the catalog browser

Not a hard dependency for anything, but significantly improves the UX of B3, B4, and D1. Can be developed at any point.

---

## C. Dynamic Feature Assembly

### C1. Assembly Template Schema

**Status:** Not Started
**Blocked by:** A1

Extend the catalog JSON format so that a terrain feature can declare **variable components** — slots where the engine picks from a set of compatible objects during generation.

Example concept:
```json
{
  "id": "tall_ruin_template",
  "feature_type": "ruins",
  "components": [
    {
      "object_id": "ruin_base_6x4",
      "transform": { "x": 0, "z": 0 }
    }
  ],
  "variable_components": [
    {
      "slot": "walls",
      "options": [
        { "object_id": "walls_L_shaped", "transform": { "x": 0, "z": 0 } },
        { "object_id": "walls_U_shaped", "transform": { "x": 0, "z": 0 } },
        { "object_id": "walls_full_surround", "transform": { "x": 0, "z": 0 } }
      ]
    }
  ]
}
```

The exact schema will evolve through iteration, but the core idea: features can have fixed components and variable slots.

### C2. Assembly in Engine (Mutation Integration)

**Status:** Not Started
**Blocked by:** C1

The engine uses assembly templates during generation:

- When instantiating a feature from an assembly template, randomly select one option per variable slot
- New mutation action: "swap component" — pick a placed feature that has variable slots, change one slot's selection
- Alternatively, extend the existing "replace" mutation to handle sub-component swaps
- Both Python and Rust engines need this (TDD workflow per CLAUDE.md)

### C3. Object-level Quantity Tracking

**Status:** Not Started
**Blocked by:** C1

Currently, quantity limits exist at both the object and feature level in the schema, but only feature-level quantities are enforced during generation. With dynamic assembly, object-level tracking becomes essential — the engine must ensure it doesn't use more wall sections than the user owns.

- Track per-object usage counts during generation
- A layout is valid only if both object-level and feature-level quantities are satisfied
- Mutation actions check object availability before selecting components

### C4. Procedural Segment Assembly (Lower Priority)

**Status:** Not Started
**Blocked by:** C2, C3

Rather than choosing from pre-defined wall configurations, the engine places individual wall segments along footprint edges. This is a more advanced form of dynamic assembly.

- Define a footprint polygon and compatible wall segment objects
- Engine procedurally places segments along edges (with gap/opening constraints)
- Significantly more complex; defer until the simpler slot-based system (C1-C3) is proven

---

## D. Manual Placement & Pinning

### D1. Manual Feature Placement

**Status:** Not Started
**Blocked by:** A1

UI for users to select a terrain feature from the catalog and place it on the battlefield by hand. Should share logic/behavior with the existing move and copy UI interactions.

- Catalog browser panel (with preview from B5 when available)
- Click-to-place or drag-to-place interaction
- Snap to the same quantization grid the engine uses (0.1" position, 15° rotation)
- Respect table bounds and optionally gap constraints
- Purely manual layouts (no engine) are valid output

### D2. Pinning / Locking

**Status:** Done
**Blocked by:** —

Mark manually-placed (or engine-placed) features as "locked" so the engine treats them as immovable constraints during generation.

- Lock/unlock toggle per feature in the UI context menu (gold button)
- Locked features are excluded from move/delete/replace/rotate mutations
- Locked features still count toward feature count preferences and visibility scoring
- Data model: `locked: bool` field on `PlacedFeature` (Python + Rust, with JSON schema)
- Visual indicator: padlock icon drawn at center of locked features
- Parity: both engines produce identical output with locked features

### D3. Manual Object Placement & Feature Crafting (Lower Priority)

**Status:** Not Started
**Blocked by:** D1, B3

Extend manual placement to individual terrain objects. Users place objects on the board, then group/attach them to create a new custom terrain feature and add it to the catalog.

- Place individual objects (not just features)
- Select multiple objects and "group as feature"
- Assign feature type, tags, and ID to the new group
- Save the composed feature back to the catalog

---

## E. Tabletop Simulator Integration

### E1. TTS Reference Save System

**Status:** Not Started
**Blocked by:** —

Load one or more TTS save files as "reference libraries." Carltographer scans their objects for matching tags (e.g., `carltographer_id=ruin_base_6x4` in the object's name, description, or GM notes) to build an index of available TTS objects.

- Parse TTS save file JSON format (well-understood from Caverns of Carl)
- Scan `ObjectStates` (including nested bags) for tagged objects
- Build an index: `carltographer_id` → TTS object definition (deep-copyable template)
- Ship one or more default reference save files with common terrain pieces
- Handle GUID refresh on copy (same pattern as Caverns of Carl)

### E2. TTS Object Associations

**Status:** Not Started
**Blocked by:** E1, A1

Link Carltographer terrain objects and features to their TTS representations. Associations are stored in the catalog alongside the terrain definitions.

- Per `TerrainObject`: optional TTS object ID + transform offset (for origin mismatch)
- Per `TerrainFeature`: optional TTS object ID + transform offset
- Fallback chain for rendering a feature in TTS:
  1. Feature-level TTS object (if available) — a single pre-assembled TTS model
  2. Assemble from constituent object-level TTS objects
  3. Generate primitive stand-ins (stretched cubes matching shape dimensions)
- UI for browsing available TTS objects (from loaded references) and linking them

### E3. TTS Save File Generation

**Status:** Not Started
**Blocked by:** E2

Generate a complete TTS save file from a Carltographer layout, following the Caverns of Carl pattern.

- Coordinate mapping: Carltographer inches → TTS units (1:1), centered on (x=0, z=0) for standard 40k table positions
- Table surface height: configurable, default y=0.96
- For each placed feature, resolve its TTS representation via the fallback chain (E2)
- Apply transforms: Carltographer position/rotation → TTS Transform (posX/posY/posZ, rotY)
- Tag all spawned objects (e.g., `carltographer_spawned` in GM notes) for batch management
- Write to TTS saves directory (OS-aware path detection)
- Include metadata: layout seed, generation params, catalog name

### E4. Custom TTS Reference Saves

**Status:** Not Started
**Blocked by:** E1

Users can point Carltographer at their own additional TTS reference save files to provide TTS objects for custom terrain.

- Configuration: list of reference save file paths (default + user-added)
- Scan all configured saves and merge into a unified TTS object index
- Conflict resolution if multiple saves define the same `carltographer_id`
- Documentation/guide for users: how to tag their TTS objects for Carltographer

### E5. TTS Lua Scripting (Optional)

**Status:** Not Started
**Blocked by:** E3

Embed Lua scripts in generated TTS saves for interactive features.

- Measurement helpers (ruler overlays)
- Feature info on hover (name, type, tags)
- Batch select/delete Carltographer-spawned objects
- Lower priority; basic save generation (E3) is useful without scripting

---

## F. Geometry Enhancements

### F1. Polygonal Shape Outlines

**Status:** Not Started
**Blocked by:** —

Extend the shape model beyond axis-aligned rectangles to support arbitrary convex (or simple) polygons. Needed for woods areas, irregularly-shaped terrain, and precise custom objects.

- New shape type: polygon (list of 2D vertices + height)
- Update collision detection (OBB → polygon intersection)
- Update visibility ray casting (polygon-based LOS blocking)
- Update both Python and Rust engines
- Significant engine change; defer until needed (woods feature or user demand)

---

## G. Physical Play Enhancements (Low Priority)

### G1. Enhanced Layout Diagrams

**Status:** Not Started
**Blocked by:** —

Improve the layout diagram output for physical tabletop play.

- Dimension annotations (arrows with inch measurements from table edges to terrain corners)
- Feature name overlays on the diagram
- Optional grid overlay
- Printable format (A4/letter PDF export)
- Low priority: current diagram is sufficient for most IRL play needs

---

## Dependency Graph

### Summary

Independent roots that can start immediately: **A1**, **B5**, **E1**, **F1**, **G1**. Also unblocked: **B2** (via B1 Done), **D2** (Done).

A1 is the most critical — it unblocks the most downstream work (A2, A3, B3→B4, C1→C2/C3, D1→D3, and E2→E3).

### Dependencies (DAG Edges)

| Feature | Depends On | Unlocks |
|---------|-----------|---------|
| **A1** (Catalog Persistence) | — | A2, A3, B3, C1, D1, E2 |
| A2 (Starter Catalogs) | A1 | — |
| A3 (Import/Export) | A1 | — |
| **B1** (Tag System) — Done | — | B2 |
| B2 (Tag-based Scoring) | B1 | — |
| B3 (Custom Object Authoring) | A1 | B4, D4 |
| B4 (Custom Feature Authoring) | B3 | — |
| **B5** (Piece Preview) | — | — (enhances B3, B4, D1) |
| C1 (Assembly Template Schema) | A1 | C2, C3 |
| C2 (Assembly in Engine) | C1 | C4 |
| C3 (Object-level Quantity) | C1 | C4 |
| C4 (Procedural Segments) | C2, C3 | — |
| D1 (Manual Placement) | A1 | D2 (Done), D3 |
| D2 (Pinning/Locking) | — | — |
| D3 (Object Placement + Crafting) | D1, B3 | — |
| **E1** (TTS Reference System) | — | E2, E4 |
| E2 (TTS Associations) | E1, A1 | E3 |
| E3 (TTS Save Generation) | E2 | E5 |
| E4 (Custom TTS References) | E1 | — |
| E5 (TTS Lua Scripting) | E3 | — |
| **F1** (Polygonal Shapes) | — | — |
| **G1** (Enhanced Diagrams) | — | — |

---

## Recommended Development Phases

### Phase 1: Foundation
**Goal**: Catalogs are data files, not hardcoded Python.

- **A1** — Catalog Persistence
- **B1** — Tag System
- **A2** — Built-in Starter Catalogs

These unblock almost everything else. Relatively contained changes: move existing catalog definitions to JSON, add load/save logic, add tag fields to the data model.

### Phase 2: TTS Integration (Basic)
**Goal**: Generate a TTS save file from a layout.

- **E1** — TTS Reference Save System
- **E2** — TTS Object Associations
- **E3** — TTS Save File Generation

Can be developed in parallel with Phase 3. High user value — this is the primary use case for many users. The Caverns of Carl pattern provides a clear implementation template.

### Phase 3: Authoring
**Goal**: Users can define their own terrain pieces.

- **B3** — Custom Object Authoring
- **B4** — Custom Feature Authoring
- **B5** — Piece Preview / Visualization
- **A3** — Catalog Import/Export

Depends on Phase 1. Gives users the tools to represent their physical terrain collections.

### Phase 4: Manual Placement & Pinning
**Goal**: Users can hand-place terrain and have the engine work around it.

- **D1** — Manual Feature Placement
- **D2** — Pinning / Locking (Done)

D1 shares UI interaction patterns with existing move/copy. D2 required engine changes (both Python and Rust, TDD workflow) and is complete.

### Phase 5: Dynamic Assembly
**Goal**: The engine can mix and match components within features.

- **C1** — Assembly Template Schema
- **C2** — Assembly in Engine
- **C3** — Object-level Quantity Tracking
- **B2** — Tag-based Scoring

Most architecturally complex phase. Engine changes in both languages. Benefits from having the authoring UI (Phase 3) available for testing and iteration.

### Phase 6: Advanced / Deferred
**Goal**: Polish and extend.

- **E4** — Custom TTS Reference Saves
- **E5** — TTS Lua Scripting
- **D3** — Manual Object Placement & Feature Crafting
- **C4** — Procedural Segment Assembly
- **F1** — Polygonal Shape Outlines
- **G1** — Enhanced Layout Diagrams

These are lower priority, higher complexity, or both. Order within this phase is flexible and driven by user need.

---

## Notes

- **Engine parity**: Any feature that touches the engine (C2, C3, B2) requires the TDD workflow from CLAUDE.md — Python first, then parity comparison, then Rust.
- **Schema evolution**: Features A1, B1, C1, and E2 all modify the catalog/layout JSON schema. Plan schema changes together to minimize breaking changes.
- **Phases 2 and 3 are parallel**: TTS integration and authoring are independent tracks. Work on whichever has more immediate user value.
- **The tag system (B1) is small but strategic**: Adding tag fields early means the schema doesn't need to change again when B2 is implemented.
