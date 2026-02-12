#!/usr/bin/env python3
"""Benchmark Python visibility engine performance.

Usage (from v2/):
    python scripts/bench_python_visibility.py          # default: 3 iterations, 100 steps
    python scripts/bench_python_visibility.py -n 5     # 5 iterations
    python scripts/bench_python_visibility.py -s 50    # 50 steps
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

# Add v2/ to path
SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(V2_DIR))

from engine.generate import generate as py_generate  # noqa: E402
from engine_cmp.compare import TEST_SCENARIOS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Python visibility engine"
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations (default: 3)",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=100,
        help="Number of generation steps (default: 100)",
    )
    args = parser.parse_args()

    # Use the basic_10_steps scenario as a template (has visibility enabled)
    scenario = next(s for s in TEST_SCENARIOS if s.name == "basic_10_steps")
    params = scenario.make_params()
    params.num_steps = args.steps

    print(f"Benchmark: Python engine, {args.steps} steps, seed={params.seed}")
    print(f"Iterations: {args.iterations}")
    print()

    # Warmup
    print("Warmup...", end=" ", flush=True)
    py_generate(params)
    print("done")

    # Timed runs
    times_ms = []
    for i in range(args.iterations):
        start = time.perf_counter()
        py_generate(params)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.1f} ms")

    median = statistics.median(times_ms)
    mean = statistics.mean(times_ms)
    print()
    print(f"Median: {median:.1f} ms")
    print(f"Mean:   {mean:.1f} ms")
    if len(times_ms) > 1:
        stdev = statistics.stdev(times_ms)
        print(f"Stdev:  {stdev:.1f} ms")


if __name__ == "__main__":
    main()
