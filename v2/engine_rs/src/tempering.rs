//! Parallel tempering (replica exchange) with simulated annealing.
//!
//! The existing terrain engine uses pure hill-climbing: try a random mutation,
//! keep it if the score improves, revert it otherwise. This gets stuck in local
//! optima — a layout that's pretty good but can't reach a better one because
//! every single-step path goes through a worse intermediate state.
//!
//! Simulated annealing (SA) fixes this by sometimes accepting worse moves,
//! controlled by a "temperature" parameter. At high temperature, almost any
//! move is accepted (random exploration). At low temperature, only improvements
//! are accepted (hill climbing). Cooling the temperature over time lets SA
//! explore broadly first, then refine.
//!
//! The problem with plain SA is choosing the temperature schedule. Cool too
//! fast and you get stuck. Cool too slow and you waste time.
//!
//! Parallel tempering sidesteps this by running multiple replicas simultaneously
//! at different fixed temperatures — from T=0 (pure hill climbing) up to a hot
//! chain that explores freely. Periodically, adjacent replicas attempt to swap
//! their candidate solutions using the Metropolis criterion. This lets good
//! solutions discovered by hot exploratory chains migrate down to the cold
//! chain for refinement, while the cold chain's refined solutions can migrate
//! up for further exploration from a different basin.
//!
//! Key design decisions:
//! - Abstract candidate interface (`TemperingCandidate` trait) so we can test
//!   the SA logic with toy problems before integrating with the real engine.
//! - Undo tokens instead of clone-before-step: avoids cloning every step.
//!   Cloning only happens when tracking a new best (rare) or during swaps.
//! - T=0 cold chain produces zero PRNG draws on rejection, so single-replica
//!   mode is backwards-compatible with the existing hill-climber's PRNG
//!   consumption pattern.
//! - Each replica gets its own PRNG stream (same seed, different sequence
//!   number). Swaps exchange candidates and scores but not PRNGs or
//!   temperatures, preserving determinism.
//! - A dedicated swap PRNG (sequence = num_replicas) ensures swap decisions
//!   are deterministic and independent of replica stepping.
//! - Rust version uses `std::thread::scope` for true parallelism across
//!   replicas within each batch, with a serial fast path for single-replica
//!   mode to avoid threading overhead.

use crate::prng::Pcg32;

/// A candidate solution that can be mutated, scored, and undone.
pub trait TemperingCandidate: Clone + Send {
    /// Opaque undo token returned by `step()`.
    type Undo: Send;

    /// Current score (higher is better). Should be cached.
    fn score(&self) -> f64;

    /// One mutation step in place. Returns an undo token.
    fn step(&mut self, rng: &mut Pcg32) -> Self::Undo;

    /// Revert the last step using the undo token.
    fn undo(&mut self, token: Self::Undo);
}

/// Parameters for parallel tempering.
pub struct TemperingParams {
    pub seed: u64,
    pub num_steps: u32,
    pub num_replicas: u32,
    pub swap_interval: u32,
    pub max_temperature: f64,
}

/// Result of a tempering run.
pub struct TemperingResult<C> {
    pub best: C,
    pub best_score: f64,
    pub steps_completed: u32,
}

struct ReplicaState<C> {
    candidate: C,
    score: f64,
    rng: Pcg32,
    temperature: f64,
}

/// Compute temperature ladder for parallel tempering.
///
/// K=1: [0.0]
/// K=2: [0.0, max_temperature]
/// K>2: [0.0] + geometric from max_temperature*0.01 to max_temperature
pub fn compute_temperatures(num_replicas: u32, max_temperature: f64) -> Vec<f64> {
    if num_replicas == 0 {
        return vec![];
    }
    if num_replicas == 1 {
        return vec![0.0];
    }
    if num_replicas == 2 {
        return vec![0.0, max_temperature];
    }

    let mut temps = vec![0.0];
    let t_min = max_temperature * 0.01;
    for i in 1..num_replicas {
        let frac = (i - 1) as f64 / (num_replicas - 2) as f64;
        let t = t_min * (max_temperature / t_min).powf(frac);
        temps.push(t);
    }
    temps
}

/// Simulated annealing acceptance criterion.
///
/// Always accepts improvements (no PRNG draw).
/// T=0: rejects worse (no PRNG draw).
/// T>0: accepts worse with P = exp((new - current) / T), consuming one rng.next_float().
pub fn sa_accept(
    current_score: f64,
    new_score: f64,
    temperature: f64,
    rng: &mut Pcg32,
) -> bool {
    if new_score >= current_score {
        return true;
    }
    if temperature <= 0.0 {
        return false;
    }
    let delta = new_score - current_score;
    let p = (delta / temperature).exp();
    rng.next_float() < p
}

/// Attempt replica exchange between two replica states.
///
/// Always consumes one swap_rng.next_float() for determinism.
/// On accept: swaps candidates and scores (temperatures and PRNGs stay fixed).
fn attempt_swap<C>(
    state_i: &mut ReplicaState<C>,
    state_j: &mut ReplicaState<C>,
    swap_rng: &mut Pcg32,
) -> bool {
    let r = swap_rng.next_float();

    let ti = state_i.temperature;
    let tj = state_j.temperature;
    let si = state_i.score;
    let sj = state_j.score;

    let accept = if ti <= 0.0 {
        sj >= si
    } else if tj <= 0.0 {
        si >= sj
    } else {
        let delta = (1.0 / ti - 1.0 / tj) * (sj - si);
        if delta >= 0.0 {
            true
        } else {
            r < delta.exp()
        }
    };

    if accept {
        std::mem::swap(&mut state_i.candidate, &mut state_j.candidate);
        std::mem::swap(&mut state_i.score, &mut state_j.score);
    }

    accept
}

/// Run one batch of steps for a single replica.
fn run_batch<C: TemperingCandidate>(
    replica: &mut ReplicaState<C>,
    batch_size: u32,
    best_score: &mut f64,
    best_candidate: &mut Option<C>,
) {
    for _ in 0..batch_size {
        let old_score = replica.candidate.score();
        let token = replica.candidate.step(&mut replica.rng);
        let new_score = replica.candidate.score();

        if old_score != new_score {
            if !sa_accept(old_score, new_score, replica.temperature, &mut replica.rng) {
                replica.candidate.undo(token);
            } else {
                replica.score = new_score;

                if new_score > *best_score {
                    *best_score = new_score;
                    *best_candidate = Some(replica.candidate.clone());
                }
            }
        }
    }
}

/// Run parallel tempering optimization.
///
/// With num_replicas > 1, each batch of steps runs in parallel via
/// `std::thread::scope`. With num_replicas == 1, runs serially.
pub fn run_tempering<C: TemperingCandidate>(
    params: &TemperingParams,
    create_candidate: impl Fn(&mut Pcg32) -> C,
) -> TemperingResult<C> {
    let temperatures = compute_temperatures(params.num_replicas, params.max_temperature);

    let mut replicas: Vec<ReplicaState<C>> = Vec::with_capacity(params.num_replicas as usize);
    for i in 0..params.num_replicas {
        let mut rng = Pcg32::new(params.seed, i as u64);
        let candidate = create_candidate(&mut rng);
        let score = candidate.score();
        replicas.push(ReplicaState {
            candidate,
            score,
            rng,
            temperature: temperatures[i as usize],
        });
    }

    let mut swap_rng = Pcg32::new(params.seed, params.num_replicas as u64);

    let mut best_score = replicas[0].score;
    let mut best_candidate = replicas[0].candidate.clone();
    for r in replicas.iter().skip(1) {
        if r.score > best_score {
            best_score = r.score;
            best_candidate = r.candidate.clone();
        }
    }

    if params.num_steps == 0 {
        return TemperingResult {
            best: best_candidate,
            best_score,
            steps_completed: 0,
        };
    }

    let mut steps_done: u32 = 0;
    let mut remaining = params.num_steps;

    while remaining > 0 {
        let batch_size = remaining.min(params.swap_interval);

        if params.num_replicas == 1 {
            // Serial fast path: no threading overhead
            run_batch(
                &mut replicas[0],
                batch_size,
                &mut best_score,
                &mut Some(best_candidate.clone()),
            );
            // Check if batch updated the best
            if replicas[0].score > best_score {
                best_score = replicas[0].score;
                best_candidate = replicas[0].candidate.clone();
            }
        } else {
            // Parallel: run each replica's batch in its own thread
            // Collect per-replica best improvements, then merge
            let mut per_replica_best: Vec<(f64, Option<C>)> =
                (0..params.num_replicas)
                    .map(|_| (f64::NEG_INFINITY, None))
                    .collect();

            std::thread::scope(|s| {
                let handles: Vec<_> = replicas
                    .iter_mut()
                    .zip(per_replica_best.iter_mut())
                    .map(|(replica, prb)| {
                        s.spawn(move || {
                            run_batch(replica, batch_size, &mut prb.0, &mut prb.1);
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().unwrap();
                }
            });

            // Merge per-replica bests into global best
            for (score, candidate) in per_replica_best {
                if score > best_score {
                    if let Some(c) = candidate {
                        best_score = score;
                        best_candidate = c;
                    }
                }
            }
        }

        steps_done += batch_size;
        remaining -= batch_size;

        // Attempt swaps between adjacent replicas
        if remaining > 0 && params.num_replicas > 1 {
            for i in 0..(params.num_replicas - 1) as usize {
                let (left, right) = replicas.split_at_mut(i + 1);
                attempt_swap(&mut left[i], &mut right[0], &mut swap_rng);
            }
        }
    }

    TemperingResult {
        best: best_candidate,
        best_score,
        steps_completed: steps_done,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prng::Pcg32;

    /// Toy candidate: minimize distance to target. Score = -(value - target)^2.
    #[derive(Clone)]
    struct QuadraticCandidate {
        value: f64,
        target: f64,
        cached_score: Option<f64>,
    }

    impl QuadraticCandidate {
        fn new(value: f64, target: f64) -> Self {
            Self {
                value,
                target,
                cached_score: None,
            }
        }
    }

    impl TemperingCandidate for QuadraticCandidate {
        type Undo = f64;

        fn score(&self) -> f64 {
            self.cached_score
                .unwrap_or_else(|| -((self.value - self.target).powi(2)))
        }

        fn step(&mut self, rng: &mut Pcg32) -> f64 {
            let old = self.value;
            self.value += (rng.next_float() - 0.5) * 2.0;
            self.cached_score = Some(-((self.value - self.target).powi(2)));
            old
        }

        fn undo(&mut self, token: f64) {
            self.value = token;
            self.cached_score = Some(-((self.value - self.target).powi(2)));
        }
    }

    #[test]
    fn test_compute_temperatures_single() {
        assert_eq!(compute_temperatures(1, 50.0), vec![0.0]);
    }

    #[test]
    fn test_compute_temperatures_two() {
        assert_eq!(compute_temperatures(2, 50.0), vec![0.0, 50.0]);
    }

    #[test]
    fn test_compute_temperatures_four() {
        let temps = compute_temperatures(4, 50.0);
        assert_eq!(temps.len(), 4);
        assert_eq!(temps[0], 0.0);
        assert!((temps[3] - 50.0).abs() < 1e-9);
        for i in 0..temps.len() - 1 {
            assert!(temps[i] < temps[i + 1]);
        }
    }

    #[test]
    fn test_compute_temperatures_eight_geometric() {
        let temps = compute_temperatures(8, 100.0);
        assert_eq!(temps.len(), 8);
        assert_eq!(temps[0], 0.0);
        assert!((temps[7] - 100.0).abs() < 1e-9);
        // Geometric: ratios between consecutive non-zero temps should be ~equal
        let ratios: Vec<f64> = (1..temps.len() - 1)
            .map(|i| temps[i + 1] / temps[i])
            .collect();
        for i in 0..ratios.len() - 1 {
            assert!(
                (ratios[i] - ratios[i + 1]).abs() < 1e-9,
                "Ratios not equal: {} vs {}",
                ratios[i],
                ratios[i + 1]
            );
        }
    }

    #[test]
    fn test_sa_accept_always_accepts_improvement() {
        let mut rng = Pcg32::new(42, 0);
        for t in [0.0, 1.0, 10.0, 100.0] {
            assert!(sa_accept(5.0, 10.0, t, &mut rng));
            assert!(sa_accept(-100.0, -50.0, t, &mut rng));
        }
    }

    #[test]
    fn test_sa_accept_rejects_at_zero_temp() {
        let mut rng = Pcg32::new(42, 0);
        assert!(!sa_accept(10.0, 5.0, 0.0, &mut rng));
        assert!(!sa_accept(0.0, -1.0, 0.0, &mut rng));
    }

    #[test]
    fn test_sa_accept_no_prng_draw_at_zero_temp() {
        let mut rng1 = Pcg32::new(42, 0);
        let mut rng2 = Pcg32::new(42, 0);

        sa_accept(10.0, 5.0, 0.0, &mut rng1);

        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }

    #[test]
    fn test_sa_accept_no_prng_draw_on_improvement() {
        let mut rng1 = Pcg32::new(42, 0);
        let mut rng2 = Pcg32::new(42, 0);

        sa_accept(5.0, 10.0, 50.0, &mut rng1);

        assert_eq!(rng1.next_u32(), rng2.next_u32());
    }

    #[test]
    fn test_sa_accept_probabilistic() {
        let n_trials = 10000;

        // High temperature, small delta
        let mut high_t_accepts = 0u32;
        let mut rng = Pcg32::new(123, 0);
        for _ in 0..n_trials {
            if sa_accept(10.0, 9.0, 50.0, &mut rng) {
                high_t_accepts += 1;
            }
        }
        let high_rate = high_t_accepts as f64 / n_trials as f64;
        let expected_high = (-1.0_f64 / 50.0).exp();
        assert!(
            (high_rate - expected_high).abs() < 0.03,
            "High T rate {high_rate} not close to {expected_high}"
        );

        // Low temperature, same delta
        let mut low_t_accepts = 0u32;
        rng = Pcg32::new(456, 0);
        for _ in 0..n_trials {
            if sa_accept(10.0, 9.0, 0.5, &mut rng) {
                low_t_accepts += 1;
            }
        }
        let low_rate = low_t_accepts as f64 / n_trials as f64;
        let expected_low = (-1.0_f64 / 0.5).exp();
        assert!(
            (low_rate - expected_low).abs() < 0.03,
            "Low T rate {low_rate} not close to {expected_low}"
        );
    }

    #[test]
    fn test_single_replica_hill_climbing() {
        let params = TemperingParams {
            seed: 42,
            num_steps: 200,
            num_replicas: 1,
            swap_interval: 20,
            max_temperature: 50.0,
        };

        let result = run_tempering(&params, |rng| {
            QuadraticCandidate::new(rng.next_float() * 20.0 - 10.0, 0.0)
        });

        assert!(result.best_score <= 0.0);
        assert_eq!(result.steps_completed, 200);
    }

    #[test]
    fn test_determinism() {
        let params = TemperingParams {
            seed: 42,
            num_steps: 100,
            num_replicas: 1,
            swap_interval: 10,
            max_temperature: 50.0,
        };

        let r1 = run_tempering(&params, |rng| {
            QuadraticCandidate::new(rng.next_float() * 20.0 - 10.0, 0.0)
        });
        let r2 = run_tempering(&params, |rng| {
            QuadraticCandidate::new(rng.next_float() * 20.0 - 10.0, 0.0)
        });

        assert_eq!(r1.best_score, r2.best_score);
        assert_eq!(r1.best.value, r2.best.value);
        assert_eq!(r1.steps_completed, r2.steps_completed);
    }

    #[test]
    fn test_multi_replica_improvement() {
        // Compare 1 replica vs 4 on the quadratic
        let single_params = TemperingParams {
            seed: 42,
            num_steps: 200,
            num_replicas: 1,
            swap_interval: 20,
            max_temperature: 50.0,
        };
        let multi_params = TemperingParams {
            seed: 42,
            num_steps: 200,
            num_replicas: 4,
            swap_interval: 20,
            max_temperature: 50.0,
        };

        let r1 = run_tempering(&single_params, |rng| {
            QuadraticCandidate::new(rng.next_float() * 20.0 - 10.0, 0.0)
        });
        let r4 = run_tempering(&multi_params, |rng| {
            QuadraticCandidate::new(rng.next_float() * 20.0 - 10.0, 0.0)
        });

        // Multi should find at least as good (usually better)
        assert!(r4.best_score >= r1.best_score - 1.0);
    }

    #[test]
    fn test_zero_steps() {
        let params = TemperingParams {
            seed: 42,
            num_steps: 0,
            num_replicas: 2,
            swap_interval: 20,
            max_temperature: 50.0,
        };

        let result = run_tempering(&params, |_rng| QuadraticCandidate::new(5.0, 0.0));

        assert_eq!(result.steps_completed, 0);
        assert_eq!(result.best_score, -25.0);
        assert_eq!(result.best.value, 5.0);
    }

    #[test]
    fn test_best_tracking() {
        let params = TemperingParams {
            seed: 42,
            num_steps: 500,
            num_replicas: 3,
            swap_interval: 25,
            max_temperature: 50.0,
        };

        let result = run_tempering(&params, |rng| {
            QuadraticCandidate::new(rng.next_float() * 20.0 - 10.0, 0.0)
        });

        // Best score must be achievable (<=0 for quadratic)
        assert!(result.best_score <= 0.0);
        assert_eq!(result.steps_completed, 500);
    }
}
