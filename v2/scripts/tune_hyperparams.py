#!/usr/bin/env python3
"""Hyperparameter tuning for the terrain generation engine.

Performs random search over TuningParams, running multiple seeds per trial
to evaluate each parameter combination. Uses the Rust engine via PyO3.

Usage (from v2/):
    # Basic run: 20 trials, 5 seeds each, 200 steps
    python scripts/tune_hyperparams.py \\
        --base-params scripts/example_base_params.json \\
        --search-space scripts/example_search_space.json

    # Customized run
    python scripts/tune_hyperparams.py \\
        --base-params scripts/example_base_params.json \\
        --search-space scripts/example_search_space.json \\
        --num-trials 50 --seeds-per-trial 10 --num-steps 500 \\
        --output results.csv

    # Include a baseline (default tuning) for comparison
    python scripts/tune_hyperparams.py \\
        --base-params scripts/example_base_params.json \\
        --search-space scripts/example_search_space.json \\
        --include-baseline
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
    "phase2_base",
    "temp_ladder_min_ratio",
]

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
    "phase2_base": 1000.0,
    "temp_ladder_min_ratio": 0.01,
}

# Parameters that should be sampled as integers
INTEGER_PARAMS = {"max_retries", "max_extra_mutations"}


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def sample_tuning_params(search_space: dict, rng: random.Random) -> dict:
    """Sample a set of tuning params from the search space.

    Search space format: {"param_name": [min, max], ...}
    Parameters not in the search space use defaults.
    """
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
    """Run the Rust engine on a single params JSON string, return result dict."""
    import engine_rs

    result_json = engine_rs.generate_json(params_json)  # type: ignore[attr-defined]
    return json.loads(result_json)


def evaluate_trial(
    base_params: dict,
    tuning: dict,
    seeds: list[int],
    num_steps: int,
) -> dict:
    """Run one trial: evaluate a set of tuning params across multiple seeds.

    Returns dict with mean_score, std_score, scores list, and elapsed time.
    """
    scores = []
    for seed in seeds:
        params = dict(base_params)
        params["seed"] = seed
        params["num_steps"] = num_steps
        params["tuning"] = tuning
        params_json = json.dumps(params)

        result = run_single(params_json)
        scores.append(result.get("score", 0.0))

    mean = sum(scores) / len(scores) if scores else 0.0
    variance = (
        sum((s - mean) ** 2 for s in scores) / len(scores)
        if len(scores) > 1
        else 0.0
    )
    std = variance**0.5

    return {
        "mean_score": mean,
        "std_score": std,
        "scores": scores,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for terrain generation engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--base-params",
        required=True,
        help="Path to base engine params JSON (catalog, table size, mission, etc.)",
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
        "--seeds-per-trial",
        type=int,
        default=5,
        help="Number of seeds to average over per trial (default: 5)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=200,
        help="Number of engine steps per run (default: 200)",
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

    base_params = load_json(args.base_params)
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

    rng = random.Random(args.seed)

    # Generate deterministic per-trial seeds
    trial_seeds = [
        rng.randint(0, 2**32 - 1) for _ in range(args.seeds_per_trial)
    ]

    # Build trial list
    trials = []
    if args.include_baseline:
        trials.append(("baseline", dict(TUNING_DEFAULTS)))
    for i in range(args.num_trials):
        tuning = sample_tuning_params(search_space, rng)
        trials.append((f"trial_{i}", tuning))

    # CSV output
    csv_fields = (
        ["trial_name"]
        + TUNING_PARAM_NAMES
        + ["mean_score", "std_score", "elapsed_s"]
        + [f"seed_{s}" for s in trial_seeds]
    )

    if args.output:
        out_file = open(args.output, "w", newline="")
    else:
        out_file = sys.stdout

    writer = csv.DictWriter(out_file, fieldnames=csv_fields)
    writer.writeheader()

    total_runs = len(trials) * args.seeds_per_trial
    print(
        f"Running {len(trials)} trials x {args.seeds_per_trial} seeds "
        f"x {args.num_steps} steps = {total_runs} engine runs",
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
        result = evaluate_trial(
            base_params, tuning, trial_seeds, args.num_steps
        )
        elapsed = time.perf_counter() - t0

        row = {"trial_name": trial_name}
        for param in TUNING_PARAM_NAMES:
            row[param] = tuning[param]
        row["mean_score"] = f"{result['mean_score']:.6f}"
        row["std_score"] = f"{result['std_score']:.6f}"
        row["elapsed_s"] = f"{elapsed:.2f}"
        for seed, score in zip(trial_seeds, result["scores"]):
            row[f"seed_{seed}"] = f"{score:.6f}"

        writer.writerow(row)

        # Flush after each row so partial results are visible
        if args.output:
            out_file.flush()

        # Track best
        if result["mean_score"] > best_score:
            best_score = result["mean_score"]
            best_trial = trial_name

        print(
            f"  [{idx + 1}/{len(trials)}] {trial_name}: "
            f"mean={result['mean_score']:.4f} std={result['std_score']:.4f} "
            f"({elapsed:.1f}s)",
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
