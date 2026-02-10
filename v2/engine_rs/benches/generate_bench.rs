//! Criterion benchmarks for the Carltographer terrain generation engine.
//!
//! 28 cases from a pairwise (all-pairs) covering array over 9 parameters:
//!   Table size (3) × Symmetry (2) × Mission (7) × Terrain (2) × Steps (4)
//!   × Feature gap (2) × Edge gap (2) × All-feature gap (2) × All-edge gap (2)
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
    "44x30_crates_none_nosym_10_g0000",
    "fixtures/44x30_crates_none_nosym_10_g0000.json"
);
bench_case!(
    bench_02,
    "60x44_wtc_none_nosym_20_g1010",
    "fixtures/60x44_wtc_none_nosym_20_g1010.json"
);
bench_case!(
    bench_03,
    "90x44_crates_none_nosym_50_g1101",
    "fixtures/90x44_crates_none_nosym_50_g1101.json"
);
bench_case!(
    bench_04,
    "44x30_wtc_none_nosym_100_g0111",
    "fixtures/44x30_wtc_none_nosym_100_g0111.json"
);

// -- 05-08: Hammer and Anvil --
bench_case!(
    bench_05,
    "60x44_crates_HnA_nosym_10_g1001",
    "fixtures/60x44_crates_HnA_nosym_10_g1001.json"
);
bench_case!(
    bench_06,
    "90x44_wtc_HnA_sym_20_g0100",
    "fixtures/90x44_wtc_HnA_sym_20_g0100.json"
);
bench_case!(
    bench_07,
    "44x30_wtc_HnA_nosym_50_g0110",
    "fixtures/44x30_wtc_HnA_nosym_50_g0110.json"
);
bench_case!(
    bench_08,
    "60x44_crates_HnA_sym_100_g1011",
    "fixtures/60x44_crates_HnA_sym_100_g1011.json"
);

// -- 09-12: Dawn of War --
bench_case!(
    bench_09,
    "90x44_wtc_DoW_nosym_10_g0110",
    "fixtures/90x44_wtc_DoW_nosym_10_g0110.json"
);
bench_case!(
    bench_10,
    "44x30_crates_DoW_sym_20_g1001",
    "fixtures/44x30_crates_DoW_sym_20_g1001.json"
);
bench_case!(
    bench_11,
    "60x44_crates_DoW_nosym_50_g0011",
    "fixtures/60x44_crates_DoW_nosym_50_g0011.json"
);
bench_case!(
    bench_12,
    "90x44_wtc_DoW_sym_100_g1100",
    "fixtures/90x44_wtc_DoW_sym_100_g1100.json"
);

// -- 13-16: Tipping Point --
bench_case!(
    bench_13,
    "44x30_wtc_TipPt_sym_10_g1010",
    "fixtures/44x30_wtc_TipPt_sym_10_g1010.json"
);
bench_case!(
    bench_14,
    "60x44_crates_TipPt_nosym_20_g0101",
    "fixtures/60x44_crates_TipPt_nosym_20_g0101.json"
);
bench_case!(
    bench_15,
    "90x44_crates_TipPt_sym_50_g1001",
    "fixtures/90x44_crates_TipPt_sym_50_g1001.json"
);
bench_case!(
    bench_16,
    "44x30_wtc_TipPt_nosym_100_g0110",
    "fixtures/44x30_wtc_TipPt_nosym_100_g0110.json"
);

// -- 17-20: Sweeping Engagement --
bench_case!(
    bench_17,
    "60x44_wtc_SwpEng_sym_10_g0101",
    "fixtures/60x44_wtc_SwpEng_sym_10_g0101.json"
);
bench_case!(
    bench_18,
    "90x44_crates_SwpEng_nosym_20_g1010",
    "fixtures/90x44_crates_SwpEng_nosym_20_g1010.json"
);
bench_case!(
    bench_19,
    "44x30_crates_SwpEng_sym_50_g1101",
    "fixtures/44x30_crates_SwpEng_sym_50_g1101.json"
);
bench_case!(
    bench_20,
    "60x44_wtc_SwpEng_nosym_100_g0010",
    "fixtures/60x44_wtc_SwpEng_nosym_100_g0010.json"
);

// -- 21-24: Crucible of Battle --
bench_case!(
    bench_21,
    "90x44_crates_Crucible_sym_10_g1000",
    "fixtures/90x44_crates_Crucible_sym_10_g1000.json"
);
bench_case!(
    bench_22,
    "44x30_wtc_Crucible_nosym_20_g0111",
    "fixtures/44x30_wtc_Crucible_nosym_20_g0111.json"
);
bench_case!(
    bench_23,
    "60x44_wtc_Crucible_sym_50_g0110",
    "fixtures/60x44_wtc_Crucible_sym_50_g0110.json"
);
bench_case!(
    bench_24,
    "90x44_crates_Crucible_nosym_100_g1001",
    "fixtures/90x44_crates_Crucible_nosym_100_g1001.json"
);

// -- 25-28: Search and Destroy --
bench_case!(
    bench_25,
    "44x30_crates_SnD_nosym_10_g0111",
    "fixtures/44x30_crates_SnD_nosym_10_g0111.json"
);
bench_case!(
    bench_26,
    "60x44_wtc_SnD_sym_20_g1000",
    "fixtures/60x44_wtc_SnD_sym_20_g1000.json"
);
bench_case!(
    bench_27,
    "90x44_wtc_SnD_nosym_50_g0010",
    "fixtures/90x44_wtc_SnD_nosym_50_g0010.json"
);
bench_case!(
    bench_28,
    "44x30_crates_SnD_sym_100_g1101",
    "fixtures/44x30_crates_SnD_sym_100_g1101.json"
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
}
criterion_main!(benches);
