#!/usr/bin/env python3
"""Random-search hyperparameter tuning across diverse game scenarios.

Samples random TuningParams combinations from a JSON search space and
evaluates each one across 24 scenarios built from an all-pairs covering
array (see tuning_scenarios.py for details on the scenario design).

Workflow:
  1. Phase 1 — Screen many random trials:
       python scripts/tune_hyperparams_multi.py \\
           --search-space scripts/example_search_space.json \\
           --num-trials 100 --seed 42 \\
           --output tuning_results.csv --include-baseline

  2. Examine the CSV to identify the best candidates (sort by mean_score
     or by other aggregated columns).

  3. Phase 2 — Re-run the best candidates with different seeds to confirm
     they generalize (not yet automated; use --num-trials 1 with manually
     specified params, or write a small wrapper script).

Each trial runs all 24 scenarios with the same fixed per-scenario seeds, so
differences between trials are purely from TuningParams. Per-scenario seeds
are derived from the --seed flag, making results fully reproducible.

Output CSV columns:
  trial_name, 11 tuning params, mean_score, std_score, min_score,
  mean_steps_{10,20,50,100} (mean score at each step budget),
  mean_{wtc,gw,omnium} (per-catalog), mean_sym_{true,false} (per-symmetry),
  elapsed_s, score_<scenario_name> (all 24 per-scenario scores)

Requires the Rust engine (engine_rs). Build with:
  python scripts/build_rust_engine.py
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(V2_DIR))

from scripts.tuning_scenarios import (  # noqa: E402
    CATALOGS,
    COVERING_ARRAY,
    STEP_COUNTS,
    build_scenarios,
)

# Tuning param names and defaults (duplicated from tune_hyperparams.py
# to keep this script self-contained)
TUNING_PARAM_NAMES = [
    "max_retries",
    "retry_decay",
    "min_move_range",
    "max_extra_mutations",
    "tile_size",
    "delete_weight_last",
    "rotate_on_move_prob",
    "shortage_boost",
    "excess_boost",
    "penalty_factor",
    "temp_ladder_min_ratio",
]

# phase2_base is intentionally excluded — it controls the boundary between
# phase 1 (feature count) and phase 2 (visibility) scoring and should not
# be tuned. Changing it just shifts scores without improving layout quality.
TUNING_DEFAULTS = {
    "max_retries": 100,
    "retry_decay": 0.95,
    "min_move_range": 2.0,
    "max_extra_mutations": 3,
    "tile_size": 2.0,
    "delete_weight_last": 0.25,
    "rotate_on_move_prob": 0.5,
    "shortage_boost": 2.0,
    "excess_boost": 2.0,
    "penalty_factor": 0.1,
    "temp_ladder_min_ratio": 0.01,
}

INTEGER_PARAMS = {"max_retries", "max_extra_mutations"}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def sample_tuning_params(search_space: dict, rng: random.Random) -> dict:
    """Sample a set of tuning params from the search space."""
    tuning = dict(TUNING_DEFAULTS)
    for param, bounds in search_space.items():
        if param not in TUNING_DEFAULTS:
            print(f"Warning: unknown tuning param '{param}', skipping")
            continue
        lo, hi = bounds
        value = rng.uniform(lo, hi)
        if param in INTEGER_PARAMS:
            value = round(value)
        tuning[param] = value
    return tuning


def run_single(params_json: str) -> dict:
    """Run the Rust engine on a single params JSON string."""
    import engine_rs

    result_json = engine_rs.generate_json(params_json)  # type: ignore[unresolved-attribute]
    return json.loads(result_json)


def evaluate_trial(
    scenarios: list[dict],
    scenario_seeds: list[int],
    tuning: dict,
) -> dict:
    """Evaluate one tuning param set across all 24 scenarios.

    Each scenario already includes its num_steps from the covering array.
    Returns dict with per-scenario scores and aggregated stats.
    """
    scores = []
    for scenario, seed in zip(scenarios, scenario_seeds):
        params = dict(scenario)
        params.pop("_scenario_name", None)
        params["seed"] = seed
        params["tuning"] = tuning

        result = run_single(json.dumps(params))
        scores.append(result.get("score", 0.0))

    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    std = variance**0.5
    min_score = min(scores)

    # Per-step-count means
    step_means = {}
    for ns in STEP_COUNTS:
        step_scores = [
            scores[i]
            for i, (_, _, _, n) in enumerate(COVERING_ARRAY)
            if n == ns
        ]
        step_means[ns] = sum(step_scores) / len(step_scores)

    # Per-catalog means
    catalog_means = {}
    for cat_name in CATALOGS:
        cat_scores = [
            scores[i]
            for i, (cn, _, _, _) in enumerate(COVERING_ARRAY)
            if cn == cat_name
        ]
        catalog_means[cat_name] = sum(cat_scores) / len(cat_scores)

    # Per-symmetry means
    sym_true_scores = [
        scores[i] for i, (_, sym, _, _) in enumerate(COVERING_ARRAY) if sym
    ]
    sym_false_scores = [
        scores[i] for i, (_, sym, _, _) in enumerate(COVERING_ARRAY) if not sym
    ]

    return {
        "scores": scores,
        "mean_score": mean,
        "std_score": std,
        "min_score": min_score,
        "step_means": step_means,
        "mean_wtc": catalog_means["WTC Set"],
        "mean_gw": catalog_means["GW Misc"],
        "mean_omnium": catalog_means["Omnium Gatherum"],
        "mean_sym_true": sum(sym_true_scores) / len(sym_true_scores),
        "mean_sym_false": sum(sym_false_scores) / len(sym_false_scores),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-scenario hyperparameter tuning for terrain generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--search-space",
        required=True,
        help="Path to search space JSON: {param: [min, max], ...}",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=20,
        help="Number of random parameter combinations to try (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the search itself (for reproducibility)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV path (default: stdout)",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include a trial with default tuning params for comparison",
    )

    args = parser.parse_args()

    # Verify Rust engine is available
    try:
        import engine_rs  # noqa: F401
    except ImportError:
        print(
            "Error: Rust engine not available. Build with:\n"
            "  cd v2 && python scripts/build_rust_engine.py",
            file=sys.stderr,
        )
        sys.exit(1)

    search_space = load_json(args.search_space)

    # Validate search space
    for param in search_space:
        if param not in TUNING_DEFAULTS:
            print(f"Error: unknown tuning param '{param}'", file=sys.stderr)
            sys.exit(1)
        bounds = search_space[param]
        if not isinstance(bounds, list) or len(bounds) != 2:
            print(
                f"Error: search space for '{param}' must be [min, max]",
                file=sys.stderr,
            )
            sys.exit(1)

    # Build scenarios (24 rows from all-pairs covering array)
    scenarios = build_scenarios()
    num_scenarios = len(scenarios)

    # Generate deterministic per-scenario seeds
    rng = random.Random(args.seed)
    scenario_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_scenarios)]

    # Build trial list
    trials = []
    if args.include_baseline:
        trials.append(("baseline", dict(TUNING_DEFAULTS)))
    for i in range(args.num_trials):
        tuning = sample_tuning_params(search_space, rng)
        trials.append((f"trial_{i}", tuning))

    # CSV columns
    scenario_names = [s["_scenario_name"] for s in scenarios]
    csv_fields = (
        ["trial_name"]
        + TUNING_PARAM_NAMES
        + [
            "mean_score",
            "std_score",
            "min_score",
        ]
        + [f"mean_steps_{n}" for n in STEP_COUNTS]
        + [
            "mean_wtc",
            "mean_gw",
            "mean_omnium",
            "mean_sym_true",
            "mean_sym_false",
            "elapsed_s",
        ]
        + [f"score_{name}" for name in scenario_names]
    )

    if args.output:
        out_file = open(args.output, "w", newline="")
    else:
        out_file = sys.stdout

    writer = csv.DictWriter(out_file, fieldnames=csv_fields)
    writer.writeheader()

    total_runs = len(trials) * num_scenarios
    steps_desc = ", ".join(str(s) for s in STEP_COUNTS)
    print(
        f"Running {len(trials)} trials x {num_scenarios} scenarios "
        f"(all-pairs over catalogs/symmetry/missions/steps [{steps_desc}]) "
        f"= {total_runs} engine runs",
        file=sys.stderr,
    )
    print(
        f"Search space: {len(search_space)} params "
        f"({', '.join(search_space.keys())})",
        file=sys.stderr,
    )

    best_trial = None
    best_score = float("-inf")

    for idx, (trial_name, tuning) in enumerate(trials):
        t0 = time.perf_counter()
        result = evaluate_trial(scenarios, scenario_seeds, tuning)
        elapsed = time.perf_counter() - t0

        row: dict = {"trial_name": trial_name}
        for param in TUNING_PARAM_NAMES:
            row[param] = tuning[param]
        row["mean_score"] = f"{result['mean_score']:.6f}"
        row["std_score"] = f"{result['std_score']:.6f}"
        row["min_score"] = f"{result['min_score']:.6f}"
        for n in STEP_COUNTS:
            row[f"mean_steps_{n}"] = f"{result['step_means'][n]:.6f}"
        row["mean_wtc"] = f"{result['mean_wtc']:.6f}"
        row["mean_gw"] = f"{result['mean_gw']:.6f}"
        row["mean_omnium"] = f"{result['mean_omnium']:.6f}"
        row["mean_sym_true"] = f"{result['mean_sym_true']:.6f}"
        row["mean_sym_false"] = f"{result['mean_sym_false']:.6f}"
        row["elapsed_s"] = f"{elapsed:.2f}"
        for name, score in zip(scenario_names, result["scores"]):
            row[f"score_{name}"] = f"{score:.6f}"

        writer.writerow(row)

        if args.output:
            out_file.flush()

        if result["mean_score"] > best_score:
            best_score = result["mean_score"]
            best_trial = trial_name

        step_detail = "  ".join(
            f"s{n}={result['step_means'][n]:.1f}" for n in STEP_COUNTS
        )
        print(
            f"  [{idx + 1}/{len(trials)}] {trial_name}: "
            f"mean={result['mean_score']:.4f} std={result['std_score']:.4f} "
            f"min={result['min_score']:.4f}  [{step_detail}]  ({elapsed:.1f}s)",
            file=sys.stderr,
        )

    if args.output:
        out_file.close()
        print(f"\nResults written to {args.output}", file=sys.stderr)

    print(
        f"\nBest: {best_trial} (mean_score={best_score:.4f})", file=sys.stderr
    )


if __name__ == "__main__":
    main()
