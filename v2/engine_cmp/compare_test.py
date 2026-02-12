"""Pytest integration for engine comparison tests."""

import pytest

from .compare import TEST_SCENARIOS, run_comparison


@pytest.mark.parametrize(
    "scenario", TEST_SCENARIOS, ids=[s.name for s in TEST_SCENARIOS]
)
def test_parity(scenario):
    """Test that Python and Rust engines produce identical results."""
    params = scenario.make_params()
    success, diffs, timing = run_comparison(
        params, verbose=True, validate_fn=scenario.validate_fn
    )

    if timing:
        print(
            f"\n  timing: py {timing.python_secs:.2f}s"
            f", rs {timing.rust_secs:.2f}s"
            f", total {timing.total_secs:.2f}s"
        )

    if not success:
        error_msg = "Engine outputs differ:\n"
        for diff in diffs:
            error_msg += f"  - {diff}\n"
        pytest.fail(error_msg)

    assert success, f"Parity check failed for {scenario.name}"
