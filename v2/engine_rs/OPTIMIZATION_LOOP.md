# Rust Engine Optimization Loop — Runbook

This document defines the protocol for one optimization attempt. Each invocation of the `rust-optimizer` agent follows this protocol exactly.

## Overview

You are attempting **one** optimization to the Rust engine in `v2/engine_rs/`. You will pick a target, implement it, benchmark it, and record the result — whether it succeeded or failed. Both outcomes are valuable and get committed.

## Phase 1: Orient

1. **Read `OPTIMIZATION_NOTES.md`** (in this directory). Understand:
   - What optimizations have already been committed (don't redo these)
   - What has been attempted and abandoned (don't retry unless the notes say "worth retrying")
   - What ideas are in the "Future Optimization Ideas" section
   - What the current profiling data shows
   - What the "Recommended next steps" suggest

2. **Read `.optimization_status`** (if it exists). This tells you the state from the previous attempt — what was tried, what's promising, any suggestions for next steps.

3. **Pick exactly one optimization target.** Criteria for selection:
   - Prefer ideas already listed in the "Future" section, roughly by tier (lower tier = higher expected impact)
   - If the notes suggest profiling first, do that as your "attempt" (profiling is a valid single-attempt outcome)
   - Be open-minded: visibility is historically the bottleneck, but collision, mutation, allocation, and other paths may also have opportunities
   - Avoid Tier 5 (architectural/high-complexity) changes — these risk parity breakage and are too large for a single attempt
   - If no promising ideas remain (all reasonable-complexity ideas have been tried), write `stop` to the status file and skip to Phase 5

4. **State your plan** briefly (in your own reasoning). What are you changing, why, and what improvement do you expect?

## Phase 2: Benchmark Baseline

Before changing any code, establish the current baseline:

```bash
cd v2/engine_rs && cargo bench
```

Record the numbers for the key benchmarks. The main ones (as of writing):
- `visibility_50` — pure visibility, no DZs (60×44, crates only)
- `visibility_100` — pure visibility, no DZs, more features (60×44, crates only)
- `mission_hna` — full mission workload with DZs and objectives (60×44, crates only)
- `mission_ruins` — mission workload with mixed terrain (44×30, crates + ruins + walls)
- `basic_100` / `all_features` — mutation-heavy, no visibility

If your optimization targets a path not covered by existing benchmarks, consider adding a benchmark. But keep it simple — don't spend your whole attempt on benchmark infrastructure.

If you want to profile to find bottlenecks, `cargo bench` output plus any instrumentation you add temporarily is fine. Remove temporary instrumentation before committing.

## Phase 3: Implement and Test

1. **Make the code change.** Keep it focused — one optimization, not a grab bag of tweaks.

2. **Run `cargo fmt`:**
   ```bash
   cd v2/engine_rs && cargo fmt
   ```

3. **Run Rust tests:**
   ```bash
   cd v2/engine_rs && cargo test
   ```
   All tests must pass. If they don't, fix the issue or abandon the attempt.

4. **Run parity tests** to verify the optimization doesn't change engine output:
   ```bash
   cd v2 && python scripts/build_rust_engine.py
   ```
   This builds the Rust engine and runs all parity comparison tests. ALL scenarios must pass. If any fail, the optimization changes engine behavior and must be fixed or abandoned.

5. **Run benchmarks:**
   ```bash
   cd v2/engine_rs && cargo bench
   ```
   Compare against the baseline from Phase 2.

## Phase 4: Evaluate and Record

### If the optimization improved any benchmark by ~5% or more:

**SUCCESS.** Update `OPTIMIZATION_NOTES.md`:
1. Add a new entry under "Committed Optimizations" with:
   - Description of what you did and why
   - Before/after benchmark numbers for ALL key benchmarks (not just the one that improved)
   - Update the cumulative improvement table
2. Remove the idea from the "Future" section if it was listed there
3. Update "Profiling Results" if you gathered new profiling data
4. Update "Recommended next steps" — what looks promising now?
5. If you have new optimization ideas based on what you learned, add them to the "Future" section in the appropriate tier

### If the optimization did NOT help (< ~5% improvement or regression):

**FAILURE (still valuable).** Revert the code changes:
```bash
cd v2/engine_rs && git checkout -- src/
```

Update `OPTIMIZATION_NOTES.md`:
1. Add an entry under "Attempted But Abandoned" with:
   - What you tried
   - What the benchmark results were
   - Why it didn't work (your analysis of why)
   - Whether it's worth retrying under different conditions
2. Remove the idea from the "Future" section if it was listed there
3. Update "Recommended next steps" based on what you learned

### In either case:

**Add any new optimization ideas** you discovered during this attempt to the "Future" section. Categorize them by expected impact tier.

## Phase 5: Status File and Commit

### Write `.optimization_status`

Write a single short status to `v2/engine_rs/.optimization_status`. This file must be **very brief** (1-2 sentences max). Format:

```
continue: <what looks promising next>
```
or
```
stop: <reason>
```

**Write `stop` when ANY of these are true:**
- No remaining ideas in the "Future" section that are below Tier 5
- The last 3+ consecutive attempts (check "Attempted But Abandoned" recency) all failed
- Profiling shows no single phase accounting for >10% of any benchmark's runtime
- All remaining ideas have complexity disproportionate to expected gain
- You genuinely believe further optimization isn't worthwhile at this time

**Write `continue` otherwise**, with a brief note on what seems most promising next.

### Commit

Stage and commit everything (notes + status file + any code changes for successful optimizations):

```bash
cd v2
git add engine_rs/OPTIMIZATION_NOTES.md engine_rs/.optimization_status
# If optimization was successful, also add the changed source files:
git add engine_rs/src/ engine_rs/benches/
git commit -m "perf(engine_rs): <brief description of what was attempted and outcome>"
```

Use a commit message like:
- `perf(engine_rs): slab decomposition for PIP — 15% improvement on mission_hna`
- `perf(engine_rs): attempted pseudoangle hybrid — no improvement, documented`
- `perf(engine_rs): profiled post-zsorted bottlenecks, updated optimization notes`

## Guidelines

### Benchmarking
- Benchmark noise is real. If results are within ±3%, treat it as "no change."
- If unsure, run `cargo bench` twice and compare. Criterion's built-in comparison helps.
- Record the numbers you actually observed, not theoretical expectations.

### Parity
- The Rust engine must produce **bit-identical** output to the Python engine for the same seed.
- `python scripts/build_rust_engine.py` is the authoritative parity check. Always run it.
- Internal-only changes (data structures, algorithms, caching) that produce the same output are fine.
- If you're unsure whether a change affects output, it probably does. Test it.

### Scope
- One attempt per invocation. Don't chain "well, while I'm here I'll also..."
- Profiling-only is a valid attempt. If the notes say "re-profile," that's useful work.
- Adding a new benchmark fixture is a valid attempt if the notes suggest one is needed.
- Keep changes minimal and focused. Don't refactor surrounding code.

### What NOT to do
- Don't modify anything in `v2/engine/` (Python engine)
- Don't modify parity comparison tests in `v2/engine_cmp/` (unless adding a new scenario to cover your change, which should be rare for pure optimizations)
- Don't attempt Tier 5 (architectural) changes — too complex for one attempt, too risky for parity
- Don't add external dependencies without strong justification (one crate for a big win is OK; don't add five)
