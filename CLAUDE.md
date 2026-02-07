# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Carltographer generates random terrain layouts for Warhammer 40k using a genetic algorithm. It evolves a population of candidate table layouts through mutation (add/remove/move/rotate/modify terrain pieces), scores them on fitness criteria (piece counts, gap enforcement, no overlaps), and selects the best.

**v1/** contains the original Lua implementation (~1900 lines) that ran inside Tabletop Simulator. It is kept as reference but is not actively developed.

**v2/** is the active rewrite. See `v2/PLAN.md` for detailed architecture and design notes.

## v2 Architecture

The v2 code is split into two layers:

- **Engine** (deterministic, portable): Terrain data model, collision/validation, genetic algorithm. Takes a terrain catalog + generation params, produces a layout. All randomness from a seeded PRNG. This layer must be translatable to Rust with bit-identical output for the same seed.

- **Frontend** (Python only): Terrain collection management (what pieces a player owns), UI, orchestration. Calls the engine, can swap between Python and Rust engine implementations.

JSON specifications are the interchange format: a **terrain catalog** describes available piece types and their geometry, and a **layout** describes a specific arrangement of pieces on a table.

## Development Environment

Python virtual environment: `v2/.env/` (Python 3.12).

Activate: `source v2/.env/bin/activate`

Intended toolchain (not all configured yet):
- **pytest** for tests: `python -m pytest v2/`
- **black** for formatting: `python -m black v2/`
- **ruff** for linting: `ruff check v2/`
- **isort** for import sorting: `python -m isort v2/`

## Key Constraints

- **Determinism**: The engine must produce identical results given the same seed. No hash-order dependence, no set iteration, no stdlib PRNG (use a portable PRNG like PCG/xoshiro implemented from scratch). This enables cross-language verification between Python and future Rust implementations.

- **No floats for positions** (under consideration): Integer or fixed-point grid coordinates avoid cross-language floating-point divergence. See PLAN.md open questions.

- **Engine purity**: The engine has no UI concerns, no asset URLs, no TTS-specific logic. It works purely with geometric abstractions.
