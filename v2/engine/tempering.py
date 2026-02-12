"""Parallel tempering (replica exchange) with simulated annealing.

Called by ``generate.py`` to run the multi-replica optimization loop. This
module provides the SA math (accept/reject via ``sa_accept``, temperature
schedules via ``compute_temperatures``) while ``generate.py`` owns the
replica array, scoring, and swap orchestration.

The existing terrain engine uses pure hill-climbing: try a random mutation,
keep it if the score improves, revert it otherwise. This gets stuck in local
optima — a layout that's pretty good but can't reach a better one because
every single-step path goes through a worse intermediate state.

Simulated annealing (SA) fixes this by sometimes accepting worse moves,
controlled by a "temperature" parameter. At high temperature, almost any
move is accepted (random exploration). At low temperature, only improvements
are accepted (hill climbing). Cooling the temperature over time lets SA
explore broadly first, then refine.

The problem with plain SA is choosing the temperature schedule. Cool too
fast and you get stuck. Cool too slow and you waste time.

Parallel tempering sidesteps this by running multiple replicas simultaneously
at different fixed temperatures — from T=0 (pure hill climbing) up to a hot
chain that explores freely. Periodically, adjacent replicas attempt to swap
their candidate solutions using the Metropolis criterion. This lets good
solutions discovered by hot exploratory chains migrate down to the cold
chain for refinement, while the cold chain's refined solutions can migrate
up for further exploration from a different basin.

Key design decisions:
- Abstract candidate interface (TemperingCandidate protocol) so we can test
  the SA logic with toy problems before integrating with the real engine.
- Undo tokens instead of clone-before-step: avoids cloning every step.
  Cloning only happens when tracking a new best (rare) or during swaps.
- T=0 cold chain produces zero PRNG draws on rejection, so single-replica
  mode is backwards-compatible with the existing hill-climber's PRNG
  consumption pattern.
- Each replica gets its own PRNG stream (same seed, different sequence
  number). Swaps exchange candidates and scores but not PRNGs or
  temperatures, preserving determinism.
- A dedicated swap PRNG (sequence = num_replicas) ensures swap decisions
  are deterministic and independent of replica stepping.

Subject to the Rust-parity constraint — ``engine_rs/src/generate.rs``
reimplements this logic inline rather than as a separate module, but the
SA math and swap criteria must match exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Generic, Protocol, TypeVar

from .prng import PCG32

T = TypeVar("T", bound="TemperingCandidate")


class TemperingCandidate(Protocol):
    def get_score(self) -> float:
        """Current score (higher is better). Should be cached."""
        ...

    def step(self, rng: PCG32, t_factor: float = 1.0) -> Any:
        """One mutation step in place. Returns an undo token.
        t_factor controls exploration: 0.0 = minimal, 1.0 = full.
        May handle internal validation (e.g., revert invalid placements
        before returning). Score updated only if state actually changed."""
        ...

    def undo(self, token: Any) -> None:
        """Revert the last step using the undo token."""
        ...

    def clone(self) -> TemperingCandidate:
        """Independent deep copy."""
        ...


@dataclass
class TemperingParams:
    seed: int
    num_steps: int
    num_replicas: int = 1
    swap_interval: int = 20
    max_temperature: float = 50.0


@dataclass
class TemperingResult(Generic[T]):
    best: T
    best_score: float
    steps_completed: int


@dataclass
class _ReplicaState:
    candidate: Any
    score: float
    rng: PCG32
    temperature: float


def compute_temperatures(
    num_replicas: int,
    max_temperature: float,
    temp_ladder_min_ratio: float = 0.01,
) -> list[float]:
    """Compute temperature ladder for parallel tempering.

    K=1: [0.0]
    K=2: [0.0, max_temperature]
    K>2: [0.0] + geometric from max_temperature*temp_ladder_min_ratio to max_temperature
    """
    if num_replicas <= 0:
        return []
    if num_replicas == 1:
        return [0.0]
    if num_replicas == 2:
        return [0.0, max_temperature]

    temps = [0.0]
    t_min = max_temperature * temp_ladder_min_ratio
    for i in range(1, num_replicas):
        frac = (i - 1) / (num_replicas - 2)
        t = t_min * (max_temperature / t_min) ** frac
        temps.append(t)
    return temps


def sa_accept(
    current_score: float,
    new_score: float,
    temperature: float,
    rng: PCG32,
) -> bool:
    """Simulated annealing acceptance criterion.

    Always accepts improvements (no PRNG draw).
    T=0: rejects worse (no PRNG draw).
    T>0: accepts worse with P = exp((new - current) / T), consuming one rng.next_float().
    """
    if new_score >= current_score:
        return True
    if temperature <= 0.0:
        return False
    delta = new_score - current_score
    p = math.exp(delta / temperature)
    return rng.next_float() < p


def attempt_swap(
    state_i: _ReplicaState,
    state_j: _ReplicaState,
    swap_rng: PCG32,
) -> bool:
    """Attempt replica exchange between states i and j.

    Always consumes one swap_rng.next_float() for determinism.
    On accept: swaps candidates and scores (temperatures and PRNGs stay fixed).
    """
    r = swap_rng.next_float()

    ti = state_i.temperature
    tj = state_j.temperature
    si = state_i.score
    sj = state_j.score

    if ti <= 0.0:
        accept = sj >= si
    elif tj <= 0.0:
        accept = si >= sj
    else:
        delta = (1.0 / ti - 1.0 / tj) * (sj - si)
        if delta >= 0.0:
            accept = True
        else:
            accept = r < math.exp(delta)

    if accept:
        state_i.candidate, state_j.candidate = (
            state_j.candidate,
            state_i.candidate,
        )
        state_i.score, state_j.score = state_j.score, state_i.score

    return accept


def run_tempering(
    params: TemperingParams,
    create_candidate: Callable[[PCG32], T],
) -> TemperingResult[T]:
    """Run parallel tempering optimization.

    Args:
        params: Tempering parameters (seed, steps, replicas, etc.)
        create_candidate: Factory that creates a candidate from a PRNG.
            Called once per replica with a uniquely-seeded PRNG.

    Returns:
        TemperingResult with the best candidate found across all replicas.
    """
    temperatures = compute_temperatures(
        params.num_replicas, params.max_temperature
    )

    replicas: list[_ReplicaState] = []
    for i in range(params.num_replicas):
        rng = PCG32(params.seed, seq=i)
        candidate = create_candidate(rng)
        score = candidate.get_score()
        replicas.append(
            _ReplicaState(
                candidate=candidate,
                score=score,
                rng=rng,
                temperature=temperatures[i],
            )
        )

    swap_rng = PCG32(params.seed, seq=params.num_replicas)

    best_candidate = replicas[0].candidate.clone()
    best_score = replicas[0].score
    for r in replicas[1:]:
        if r.score > best_score:
            best_score = r.score
            best_candidate = r.candidate.clone()

    if params.num_steps <= 0:
        return TemperingResult(
            best=best_candidate,
            best_score=best_score,
            steps_completed=0,
        )

    steps_done = 0
    remaining = params.num_steps

    while remaining > 0:
        batch_size = min(params.swap_interval, remaining)

        for replica in replicas:
            for _ in range(batch_size):
                old_score = replica.candidate.get_score()
                t_factor = (
                    replica.temperature / params.max_temperature
                    if params.max_temperature > 0
                    else 0.0
                )
                token = replica.candidate.step(replica.rng, t_factor)
                new_score = replica.candidate.get_score()

                if old_score != new_score:
                    if not sa_accept(
                        old_score, new_score, replica.temperature, replica.rng
                    ):
                        replica.candidate.undo(token)
                    else:
                        replica.score = new_score

                        if new_score > best_score:
                            best_score = new_score
                            best_candidate = replica.candidate.clone()

        steps_done += batch_size
        remaining -= batch_size

        if remaining > 0 and params.num_replicas > 1:
            for i in range(params.num_replicas - 1):
                attempt_swap(replicas[i], replicas[i + 1], swap_rng)

    return TemperingResult(
        best=best_candidate,
        best_score=best_score,
        steps_completed=steps_done,
    )
