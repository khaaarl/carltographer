#!/usr/bin/env python3
"""Profile engine performance using test scenarios as workloads.

Usage (from v2/):
    python scripts/profile_engine.py --profile                       # cProfile top 30 functions
    python scripts/profile_engine.py --profile --scenario basic_50_steps
    python scripts/profile_engine.py --compare-time                  # Python vs Rust timing
    python scripts/profile_engine.py --compare-time --heavy          # only visibility-enabled scenarios
    python scripts/profile_engine.py --dump-json                     # print scenario params as JSON
    python scripts/profile_engine.py --dump-json --scenario basic_100_steps
"""

import argparse
import cProfile
import json
import pstats
import statistics
import sys
import time
from pathlib import Path

# Add v2/ to path so we can import engine and engine_cmp
SCRIPT_DIR = Path(__file__).resolve().parent
V2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(V2_DIR))

from engine.generate import generate as py_generate  # noqa: E402
from engine_cmp.compare import (  # noqa: E402
    TEST_SCENARIOS,
    params_to_json_dict,
)


def _select_scenarios(args):
    """Filter scenarios based on CLI args."""
    scenarios = list(TEST_SCENARIOS)
    if args.scenario:
        scenarios = [s for s in scenarios if s.name == args.scenario]
        if not scenarios:
            names = [s.name for s in TEST_SCENARIOS]
            print(f"Unknown scenario: {args.scenario}")
            print(f"Available: {', '.join(names)}")
            sys.exit(1)
    if args.heavy:
        scenarios = [s for s in scenarios if not s.skip_visibility]
        if not scenarios:
            print("No heavy (visibility-enabled) scenarios found.")
            sys.exit(1)
    return scenarios


def cmd_profile(args):
    """Run cProfile on Python engine."""
    scenarios = _select_scenarios(args)
    top_n = args.top or 30

    profiler = cProfile.Profile()
    for scenario in scenarios:
        params = scenario.make_params()
        print(f"Profiling: {scenario.name} ({scenario.num_steps} steps)...")
        profiler.enable()
        py_generate(params)
        profiler.disable()

    print(f"\n{'=' * 70}")
    print(f"Top {top_n} functions by cumulative time")
    print(f"Scenarios: {', '.join(s.name for s in scenarios)}")
    print(f"{'=' * 70}\n")

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n)

    if args.output:
        profiler.dump_stats(args.output)
        print(f"\nProfile data written to {args.output}")
        print("Visualize with: snakeviz " + args.output)


def cmd_compare_time(args):
    """Time Python vs Rust engine on each scenario."""
    scenarios = _select_scenarios(args)
    iterations = args.iterations or 3

    try:
        import engine_rs

        has_rust = True
    except ImportError:
        has_rust = False
        print("Warning: Rust engine not available, showing Python times only")
        print("Build with: cd v2/engine_rs && maturin develop --release\n")

    header = f"{'Scenario':<35} {'Python (ms)':>12}"
    divider_len = 49
    if has_rust:
        header += f" {'Rust (ms)':>12} {'Speedup':>10}"
        divider_len = 71
    print(header)
    print("-" * divider_len)

    for scenario in scenarios:
        params = scenario.make_params()

        # Time Python
        py_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            py_generate(params)
            elapsed = (time.perf_counter() - start) * 1000
            py_times.append(elapsed)
        py_median = statistics.median(py_times)

        # Time Rust
        if has_rust:
            params_json = json.dumps(params_to_json_dict(params))
            rs_times = []
            for _ in range(iterations):
                start = time.perf_counter()
                engine_rs.generate_json(params_json)  # type: ignore[unresolved-attribute]
                elapsed = (time.perf_counter() - start) * 1000
                rs_times.append(elapsed)
            rs_median = statistics.median(rs_times)
            speedup = py_median / rs_median if rs_median > 0 else float("inf")
            print(
                f"{scenario.name:<35} {py_median:>12.1f} {rs_median:>12.1f} {speedup:>9.1f}x"
            )
        else:
            print(f"{scenario.name:<35} {py_median:>12.1f}")

    print(f"\n({iterations} iterations each, median reported)")


def cmd_dump_json(args):
    """Print scenario params as JSON for Rust benchmark fixtures."""
    scenarios = _select_scenarios(args)
    for scenario in scenarios:
        params = scenario.make_params()
        params_dict = params_to_json_dict(params)
        print(f"// Scenario: {scenario.name}")
        print(json.dumps(params_dict, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Profile Carltographer engine performance"
    )
    sub = parser.add_subparsers(dest="command")

    # Common args added to each subparser
    def add_common(p):
        p.add_argument("--scenario", help="Run only this scenario (by name)")
        p.add_argument(
            "--heavy",
            action="store_true",
            help="Only visibility-enabled scenarios",
        )

    # --profile
    p_profile = sub.add_parser("profile", help="cProfile the Python engine")
    add_common(p_profile)
    p_profile.add_argument(
        "--top", type=int, help="Number of top functions to show (default: 30)"
    )
    p_profile.add_argument(
        "--output", "-o", help="Write cProfile binary data to file"
    )

    # --compare-time
    p_time = sub.add_parser("compare-time", help="Time Python vs Rust engine")
    add_common(p_time)
    p_time.add_argument(
        "--iterations",
        type=int,
        help="Iterations per scenario (default: 3)",
    )

    # --dump-json
    p_dump = sub.add_parser("dump-json", help="Print scenario params as JSON")
    add_common(p_dump)

    args = parser.parse_args()

    if args.command == "profile":
        cmd_profile(args)
    elif args.command == "compare-time":
        cmd_compare_time(args)
    elif args.command == "dump-json":
        cmd_dump_json(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
