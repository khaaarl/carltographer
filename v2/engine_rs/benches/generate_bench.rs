//! Criterion benchmarks for the Carltographer terrain generation engine.
//!
//! 35 cases from a pairwise (all-pairs) covering array over 10 parameters:
//!   Table size (3) × Symmetry (2) × Mission (7) × Terrain (3) × Steps (4)
//!   × Feature gap (2) × Edge gap (2) × All-feature gap (2) × All-edge gap (2)
//!   × Replicas (4: 1, 2, 4, 8)
//!
//! Constraints:
//!   - mission=none → symmetry forced off
//!   - steps >= 50 OR table >= 60×44 → replicas != 8
//!   - steps >= 100 OR table >= 90×44 → replicas != 4
//!
//! JSON fixtures generated via:
//!   `python engine_rs/benches/generate_fixtures.py` (from v2/)
//!
//! Run with: `cargo bench` from `v2/engine_rs/`

use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use engine_rs::generate::generate;
use engine_rs::types::EngineParams;

macro_rules! bench_case {
    ($fn_name:ident, $bench_name:expr, $file:expr) => {
        fn $fn_name(c: &mut Criterion) {
            let params: EngineParams = serde_json::from_str(include_str!($file)).unwrap();
            c.bench_function($bench_name, |b| b.iter(|| generate(&params)));
        }
    };
}

// -- 01-04: no mission (symmetry forced off) --
bench_case!(
    bench_01,
    "90x44_crates_none_nosym_10_g0000_r1",
    "fixtures/90x44_crates_none_nosym_10_g0000_r1.json"
);
bench_case!(
    bench_02,
    "44x30_wtc_none_nosym_20_g1010_r8",
    "fixtures/44x30_wtc_none_nosym_20_g1010_r8.json"
);
bench_case!(
    bench_03,
    "60x44_crates_none_nosym_50_g1101_r4",
    "fixtures/60x44_crates_none_nosym_50_g1101_r4.json"
);
bench_case!(
    bench_04,
    "44x30_wtc_none_nosym_100_g0111_r2",
    "fixtures/44x30_wtc_none_nosym_100_g0111_r2.json"
);

// -- 05-08: Hammer and Anvil --
bench_case!(
    bench_05,
    "44x30_wtc_HnA_nosym_10_g1001_r8",
    "fixtures/44x30_wtc_HnA_nosym_10_g1001_r8.json"
);
bench_case!(
    bench_06,
    "90x44_crates_HnA_sym_20_g0100_r1",
    "fixtures/90x44_crates_HnA_sym_20_g0100_r1.json"
);
bench_case!(
    bench_07,
    "60x44_wtc_HnA_nosym_50_g0110_r4",
    "fixtures/60x44_wtc_HnA_nosym_50_g0110_r4.json"
);
bench_case!(
    bench_08,
    "60x44_crates_HnA_sym_100_g1011_r2",
    "fixtures/60x44_crates_HnA_sym_100_g1011_r2.json"
);

// -- 09-12: Dawn of War --
bench_case!(
    bench_09,
    "44x30_wtc_DoW_sym_10_g0110_r8",
    "fixtures/44x30_wtc_DoW_sym_10_g0110_r8.json"
);
bench_case!(
    bench_10,
    "60x44_crates_DoW_nosym_20_g1001_r4",
    "fixtures/60x44_crates_DoW_nosym_20_g1001_r4.json"
);
bench_case!(
    bench_11,
    "90x44_crates_DoW_sym_50_g0011_r2",
    "fixtures/90x44_crates_DoW_sym_50_g0011_r2.json"
);
bench_case!(
    bench_12,
    "90x44_wtc_DoW_nosym_100_g1100_r1",
    "fixtures/90x44_wtc_DoW_nosym_100_g1100_r1.json"
);

// -- 13-16: Tipping Point --
bench_case!(
    bench_13,
    "60x44_wtc_TipPt_sym_10_g1010_r2",
    "fixtures/60x44_wtc_TipPt_sym_10_g1010_r2.json"
);
bench_case!(
    bench_14,
    "44x30_crates_TipPt_nosym_20_g0101_r8",
    "fixtures/44x30_crates_TipPt_nosym_20_g0101_r8.json"
);
bench_case!(
    bench_15,
    "44x30_crates_TipPt_sym_50_g1001_r4",
    "fixtures/44x30_crates_TipPt_sym_50_g1001_r4.json"
);
bench_case!(
    bench_16,
    "90x44_wtc_TipPt_nosym_100_g0110_r1",
    "fixtures/90x44_wtc_TipPt_nosym_100_g0110_r1.json"
);

// -- 17-20: Sweeping Engagement --
bench_case!(
    bench_17,
    "60x44_crates_SwpEng_nosym_10_g0101_r4",
    "fixtures/60x44_crates_SwpEng_nosym_10_g0101_r4.json"
);
bench_case!(
    bench_18,
    "44x30_wtc_SwpEng_sym_20_g1010_r8",
    "fixtures/44x30_wtc_SwpEng_sym_20_g1010_r8.json"
);
bench_case!(
    bench_19,
    "90x44_crates_SwpEng_nosym_50_g1101_r2",
    "fixtures/90x44_crates_SwpEng_nosym_50_g1101_r2.json"
);
bench_case!(
    bench_20,
    "60x44_wtc_SwpEng_sym_100_g0010_r1",
    "fixtures/60x44_wtc_SwpEng_sym_100_g0010_r1.json"
);

// -- 21-24: Crucible of Battle --
bench_case!(
    bench_21,
    "44x30_crates_Crucible_sym_10_g1000_r8",
    "fixtures/44x30_crates_Crucible_sym_10_g1000_r8.json"
);
bench_case!(
    bench_22,
    "60x44_wtc_Crucible_nosym_20_g0111_r2",
    "fixtures/60x44_wtc_Crucible_nosym_20_g0111_r2.json"
);
bench_case!(
    bench_23,
    "60x44_wtc_Crucible_sym_50_g0110_r4",
    "fixtures/60x44_wtc_Crucible_sym_50_g0110_r4.json"
);
bench_case!(
    bench_24,
    "90x44_crates_Crucible_nosym_100_g1001_r1",
    "fixtures/90x44_crates_Crucible_nosym_100_g1001_r1.json"
);

// -- 25-28: Search and Destroy --
bench_case!(
    bench_25,
    "60x44_wtc_SnD_nosym_10_g0111_r4",
    "fixtures/60x44_wtc_SnD_nosym_10_g0111_r4.json"
);
bench_case!(
    bench_26,
    "44x30_crates_SnD_sym_20_g1000_r8",
    "fixtures/44x30_crates_SnD_sym_20_g1000_r8.json"
);
bench_case!(
    bench_27,
    "44x30_wtc_SnD_nosym_50_g0010_r1",
    "fixtures/44x30_wtc_SnD_nosym_50_g0010_r1.json"
);
bench_case!(
    bench_28,
    "90x44_crates_SnD_sym_100_g1101_r2",
    "fixtures/90x44_crates_SnD_sym_100_g1101_r2.json"
);

// -- 29-35: WTC + polygon shapes --
bench_case!(
    bench_29,
    "60x44_wtcPoly_none_nosym_20_g0101_r4",
    "fixtures/60x44_wtcPoly_none_nosym_20_g0101_r4.json"
);
bench_case!(
    bench_30,
    "44x30_wtcPoly_HnA_sym_10_g1010_r8",
    "fixtures/44x30_wtcPoly_HnA_sym_10_g1010_r8.json"
);
bench_case!(
    bench_31,
    "90x44_wtcPoly_DoW_nosym_50_g0011_r2",
    "fixtures/90x44_wtcPoly_DoW_nosym_50_g0011_r2.json"
);
bench_case!(
    bench_32,
    "60x44_wtcPoly_TipPt_sym_100_g1100_r1",
    "fixtures/60x44_wtcPoly_TipPt_sym_100_g1100_r1.json"
);
bench_case!(
    bench_33,
    "44x30_wtcPoly_SwpEng_nosym_10_g0110_r8",
    "fixtures/44x30_wtcPoly_SwpEng_nosym_10_g0110_r8.json"
);
bench_case!(
    bench_34,
    "90x44_wtcPoly_Crucible_sym_20_g1001_r1",
    "fixtures/90x44_wtcPoly_Crucible_sym_20_g1001_r1.json"
);
bench_case!(
    bench_35,
    "44x30_wtcPoly_SnD_sym_50_g0000_r2",
    "fixtures/44x30_wtcPoly_SnD_sym_50_g0000_r2.json"
);

fn config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = config();
    targets =
        bench_01, bench_02, bench_03, bench_04,
        bench_05, bench_06, bench_07, bench_08,
        bench_09, bench_10, bench_11, bench_12,
        bench_13, bench_14, bench_15, bench_16,
        bench_17, bench_18, bench_19, bench_20,
        bench_21, bench_22, bench_23, bench_24,
        bench_25, bench_26, bench_27, bench_28,
        bench_29, bench_30, bench_31, bench_32,
        bench_33, bench_34, bench_35,
}
criterion_main!(benches);
