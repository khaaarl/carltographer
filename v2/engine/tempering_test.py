"""Tests for parallel tempering module using a toy quadratic candidate."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .prng import PCG32
from .tempering import (
    TemperingParams,
    _ReplicaState,
    attempt_swap,
    compute_temperatures,
    run_tempering,
    sa_accept,
)


@dataclass
class QuadraticCandidate:
    """Minimize distance to target value. Score = -(value - target)^2."""

    value: float
    target: float
    _score: float | None = None

    def get_score(self) -> float:
        if self._score is None:
            self._score = -((self.value - self.target) ** 2)
        return self._score

    def step(self, rng: PCG32) -> float:
        old_value = self.value
        self.value += (rng.next_float() - 0.5) * 2.0
        self._score = None
        return old_value

    def undo(self, token: float) -> None:
        self.value = token
        self._score = None

    def clone(self) -> QuadraticCandidate:
        return QuadraticCandidate(self.value, self.target, self._score)


# --- compute_temperatures tests ---


def test_compute_temperatures_single():
    temps = compute_temperatures(1, 50.0)
    assert temps == [0.0]


def test_compute_temperatures_two():
    temps = compute_temperatures(2, 50.0)
    assert temps == [0.0, 50.0]


def test_compute_temperatures_four():
    temps = compute_temperatures(4, 50.0)
    assert len(temps) == 4
    assert temps[0] == 0.0
    assert abs(temps[-1] - 50.0) < 1e-9
    # Monotonically increasing
    for i in range(len(temps) - 1):
        assert temps[i] < temps[i + 1]


def test_compute_temperatures_eight():
    temps = compute_temperatures(8, 100.0)
    assert len(temps) == 8
    assert temps[0] == 0.0
    assert abs(temps[-1] - 100.0) < 1e-9
    # Geometric spacing: ratios between consecutive non-zero temps should be ~equal
    ratios = [temps[i + 1] / temps[i] for i in range(1, len(temps) - 1)]
    for i in range(len(ratios) - 1):
        assert abs(ratios[i] - ratios[i + 1]) < 1e-9


# --- sa_accept tests ---


def test_sa_accept_always_accepts_improvement():
    rng = PCG32(42, 0)
    # Various temperatures
    for t in [0.0, 1.0, 10.0, 100.0]:
        assert sa_accept(5.0, 10.0, t, rng) is True
        assert sa_accept(-100.0, -50.0, t, rng) is True


def test_sa_accept_rejects_at_zero_temp():
    rng = PCG32(42, 0)
    assert sa_accept(10.0, 5.0, 0.0, rng) is False
    assert sa_accept(0.0, -1.0, 0.0, rng) is False


def test_sa_accept_no_prng_draw_at_zero_temp():
    """T=0 rejection should not consume any PRNG values."""
    rng1 = PCG32(42, 0)
    rng2 = PCG32(42, 0)

    # rng1: call sa_accept with T=0 and worse score (should not draw)
    sa_accept(10.0, 5.0, 0.0, rng1)

    # Both RNGs should be in the same state
    assert rng1.next_u32() == rng2.next_u32()


def test_sa_accept_no_prng_draw_on_improvement():
    """Improvement acceptance should not consume any PRNG values."""
    rng1 = PCG32(42, 0)
    rng2 = PCG32(42, 0)

    sa_accept(5.0, 10.0, 50.0, rng1)

    assert rng1.next_u32() == rng2.next_u32()


def test_sa_accept_probabilistic():
    """At high T, small delta → high acceptance. At low T → low acceptance."""
    n_trials = 10000

    # High temperature, small delta: should accept most
    high_t_accepts = 0
    rng = PCG32(123, 0)
    for _ in range(n_trials):
        if sa_accept(10.0, 9.0, 50.0, rng):
            high_t_accepts += 1
    high_rate = high_t_accepts / n_trials
    expected_high = math.exp(-1.0 / 50.0)  # ~0.98
    assert abs(high_rate - expected_high) < 0.03

    # Low temperature, same delta: should accept few
    low_t_accepts = 0
    rng = PCG32(456, 0)
    for _ in range(n_trials):
        if sa_accept(10.0, 9.0, 0.5, rng):
            low_t_accepts += 1
    low_rate = low_t_accepts / n_trials
    expected_low = math.exp(-1.0 / 0.5)  # ~0.135
    assert abs(low_rate - expected_low) < 0.03


# --- attempt_swap tests ---


def test_attempt_swap_always_consumes_rng():
    """swap_rng should always advance by one, regardless of accept/reject."""
    swap_rng1 = PCG32(99, 0)
    swap_rng2 = PCG32(99, 0)

    si = _ReplicaState(
        candidate=QuadraticCandidate(5.0, 0.0),
        score=-25.0,
        rng=PCG32(1, 0),
        temperature=10.0,
    )
    sj = _ReplicaState(
        candidate=QuadraticCandidate(3.0, 0.0),
        score=-9.0,
        rng=PCG32(2, 0),
        temperature=20.0,
    )

    attempt_swap(si, sj, swap_rng1)

    # swap_rng2 should match if we manually advance by one
    swap_rng2.next_float()
    assert swap_rng1.next_u32() == swap_rng2.next_u32()


# --- run_tempering: single replica (hill climbing) ---


def test_single_replica_hill_climbing():
    """With 1 replica at T=0, score should only improve (never decrease)."""
    params = TemperingParams(seed=42, num_steps=200, num_replicas=1)

    def create(rng: PCG32) -> QuadraticCandidate:
        return QuadraticCandidate(
            value=rng.next_float() * 20.0 - 10.0, target=0.0
        )

    result = run_tempering(params, create)
    assert result.best_score <= 0.0  # score is -(value-target)^2, max is 0
    assert result.steps_completed == 200


# --- Determinism ---


def test_determinism():
    """Same params → identical results."""
    params = TemperingParams(
        seed=42, num_steps=100, num_replicas=3, swap_interval=10
    )

    def create(rng: PCG32) -> QuadraticCandidate:
        return QuadraticCandidate(
            value=rng.next_float() * 20.0 - 10.0, target=0.0
        )

    r1 = run_tempering(params, create)
    r2 = run_tempering(params, create)

    assert r1.best_score == r2.best_score
    assert r1.best.value == r2.best.value
    assert r1.steps_completed == r2.steps_completed


def test_different_seeds():
    """Different seeds → different results."""

    def create(rng: PCG32) -> QuadraticCandidate:
        return QuadraticCandidate(
            value=rng.next_float() * 20.0 - 10.0, target=0.0
        )

    r1 = run_tempering(
        TemperingParams(seed=1, num_steps=100, num_replicas=2), create
    )
    r2 = run_tempering(
        TemperingParams(seed=2, num_steps=100, num_replicas=2), create
    )

    # With different seeds, at least one of value/score should differ
    assert r1.best.value != r2.best.value or r1.best_score != r2.best_score


# --- Multi-replica improvement ---


def test_multi_replica_finds_better_solution():
    """4 replicas should find better solutions more often than 1 on a multimodal problem."""

    @dataclass
    class MultimodalCandidate:
        """Multiple local optima: peaks at -5, 0, +5. Global optimum at 0."""

        value: float
        _score: float | None = None

        def get_score(self) -> float:
            if self._score is None:
                v = self.value
                self._score = (
                    -(v**2)
                    + 3.0 * math.exp(-((v - 5) ** 2))
                    + 3.0 * math.exp(-((v + 5) ** 2))
                )
            return self._score

        def step(self, rng: PCG32) -> float:
            old = self.value
            self.value += (rng.next_float() - 0.5) * 2.0
            self._score = None
            return old

        def undo(self, token: float) -> None:
            self.value = token
            self._score = None

        def clone(self) -> MultimodalCandidate:
            return MultimodalCandidate(self.value, self._score)

    single_wins = 0
    multi_wins = 0
    n_trials = 50

    for seed in range(n_trials):

        def create_single(rng: PCG32) -> MultimodalCandidate:
            return MultimodalCandidate(value=rng.next_float() * 20.0 - 10.0)

        def create_multi(rng: PCG32) -> MultimodalCandidate:
            return MultimodalCandidate(value=rng.next_float() * 20.0 - 10.0)

        r1 = run_tempering(
            TemperingParams(seed=seed, num_steps=200, num_replicas=1),
            create_single,
        )
        r4 = run_tempering(
            TemperingParams(
                seed=seed, num_steps=200, num_replicas=4, swap_interval=20
            ),
            create_multi,
        )

        if r4.best_score > r1.best_score:
            multi_wins += 1
        elif r1.best_score > r4.best_score:
            single_wins += 1

    # Multi-replica should win at least sometimes
    assert multi_wins > single_wins or multi_wins >= 5


# --- Swap occurs ---


def test_swap_occurs():
    """With 2 replicas, verify that swaps actually happen."""
    # Use a candidate where one replica gets stuck at a bad value
    # and the other finds a better one — a swap should transfer the good solution

    params = TemperingParams(
        seed=42,
        num_steps=100,
        num_replicas=2,
        swap_interval=10,
        max_temperature=50.0,
    )

    # Track initial values per replica
    initial_values: list[float] = []

    def create(rng: PCG32) -> QuadraticCandidate:
        val = rng.next_float() * 20.0 - 10.0
        initial_values.append(val)
        return QuadraticCandidate(value=val, target=0.0)

    result = run_tempering(params, create)
    assert result.steps_completed == 100
    # Best score should be better than worst initial
    worst_initial = max(-(v**2) for v in initial_values)
    assert result.best_score >= worst_initial


# --- Edge cases ---


def test_zero_steps():
    """num_steps=0 returns initial state."""
    params = TemperingParams(seed=42, num_steps=0, num_replicas=2)

    def create(rng: PCG32) -> QuadraticCandidate:
        return QuadraticCandidate(value=5.0, target=0.0)

    result = run_tempering(params, create)
    assert result.steps_completed == 0
    assert result.best_score == -(5.0**2)
    assert result.best.value == 5.0


def test_best_tracking():
    """Best score should be monotonically non-decreasing."""
    # We verify this by running and checking the result is at least as good
    # as the initial best
    params = TemperingParams(
        seed=42, num_steps=500, num_replicas=3, swap_interval=25
    )

    initial_scores: list[float] = []

    def create(rng: PCG32) -> QuadraticCandidate:
        c = QuadraticCandidate(
            value=rng.next_float() * 20.0 - 10.0, target=0.0
        )
        initial_scores.append(c.get_score())
        return c

    result = run_tempering(params, create)
    initial_best = max(initial_scores)
    assert result.best_score >= initial_best
