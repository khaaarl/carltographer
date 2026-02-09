//! Criterion benchmarks for the Carltographer terrain generation engine.
//!
//! Run with: `cargo bench` from `v2/engine_rs/`
//!
//! JSON fixtures generated via:
//!   `python scripts/profile_engine.py dump-json --scenario <name>`

use criterion::{criterion_group, criterion_main, Criterion};
use engine_rs::generate::generate;
use engine_rs::types::EngineParams;

// -- JSON fixtures --
// Each corresponds to a test scenario from engine_cmp/compare.py.

/// 100 steps, no visibility — pure generation workload.
const BASIC_100_JSON: &str = r#"{
  "seed": 42,
  "table_width_inches": 60.0,
  "table_depth_inches": 44.0,
  "catalog": {
    "objects": [
      {
        "item": {
          "id": "crate_5x2.5",
          "shapes": [
            {
              "shape_type": "rectangular_prism",
              "width_inches": 5.0,
              "depth_inches": 2.5,
              "height_inches": 2.0
            }
          ],
          "name": "Crate"
        }
      }
    ],
    "features": [
      {
        "item": {
          "id": "crate",
          "feature_type": "obstacle",
          "components": [
            {
              "object_id": "crate_5x2.5"
            }
          ]
        }
      }
    ],
    "name": "Test Catalog"
  },
  "num_steps": 100,
  "skip_visibility": true
}"#;

/// 100 steps with gap enforcement + feature count preferences, no visibility.
const ALL_FEATURES_JSON: &str = r#"{
  "seed": 42,
  "table_width_inches": 60.0,
  "table_depth_inches": 44.0,
  "catalog": {
    "objects": [
      {
        "item": {
          "id": "crate_5x2.5",
          "shapes": [
            {
              "shape_type": "rectangular_prism",
              "width_inches": 5.0,
              "depth_inches": 2.5,
              "height_inches": 2.0
            }
          ],
          "name": "Crate"
        }
      }
    ],
    "features": [
      {
        "item": {
          "id": "crate",
          "feature_type": "obstacle",
          "components": [
            {
              "object_id": "crate_5x2.5"
            }
          ]
        }
      }
    ],
    "name": "Test Catalog"
  },
  "num_steps": 100,
  "feature_count_preferences": [
    {
      "feature_type": "obstacle",
      "min": 3,
      "max": 10
    }
  ],
  "min_feature_gap_inches": 2.0,
  "min_edge_gap_inches": 1.0,
  "skip_visibility": true
}"#;

/// 50 steps with visibility computation enabled.
/// Uses 5" tall crates (above 4" blocking height threshold) so terrain
/// actually generates blocking segments for the intersection loop.
const VISIBILITY_50_JSON: &str = r#"{
  "seed": 42,
  "table_width_inches": 60.0,
  "table_depth_inches": 44.0,
  "catalog": {
    "objects": [
      {
        "item": {
          "id": "crate_5x2.5",
          "shapes": [
            {
              "shape_type": "rectangular_prism",
              "width_inches": 5.0,
              "depth_inches": 2.5,
              "height_inches": 5.0
            }
          ],
          "name": "Crate"
        }
      }
    ],
    "features": [
      {
        "item": {
          "id": "crate",
          "feature_type": "obstacle",
          "components": [
            {
              "object_id": "crate_5x2.5"
            }
          ]
        }
      }
    ],
    "name": "Test Catalog"
  },
  "num_steps": 50
}"#;

/// 50 steps with mission (Hammer and Anvil) + visibility + DZ scoring.
/// Uses 5" tall crates (above 4" blocking height threshold) so terrain
/// actually generates blocking segments for the intersection loop.
const MISSION_HNA_JSON: &str = r#"{
  "seed": 42,
  "table_width_inches": 60.0,
  "table_depth_inches": 44.0,
  "catalog": {
    "objects": [
      {
        "item": {
          "id": "crate_5x2.5",
          "shapes": [
            {
              "shape_type": "rectangular_prism",
              "width_inches": 5.0,
              "depth_inches": 2.5,
              "height_inches": 5.0
            }
          ],
          "name": "Crate"
        }
      }
    ],
    "features": [
      {
        "item": {
          "id": "crate",
          "feature_type": "obstacle",
          "components": [
            {
              "object_id": "crate_5x2.5"
            }
          ]
        }
      }
    ],
    "name": "Test Catalog"
  },
  "num_steps": 50,
  "mission": {
    "name": "Hammer and Anvil",
    "objectives": [
      { "id": "1", "position": { "x_inches": 0.0, "z_inches": 0.0 }, "range_inches": 3.0 },
      { "id": "2", "position": { "x_inches": 0.0, "z_inches": -16.0 }, "range_inches": 3.0 },
      { "id": "3", "position": { "x_inches": 0.0, "z_inches": 16.0 }, "range_inches": 3.0 },
      { "id": "4", "position": { "x_inches": -20.0, "z_inches": 0.0 }, "range_inches": 3.0 },
      { "id": "5", "position": { "x_inches": 20.0, "z_inches": 0.0 }, "range_inches": 3.0 }
    ],
    "deployment_zones": [
      {
        "id": "green",
        "polygons": [[
          { "x_inches": -30.0, "z_inches": -22.0 },
          { "x_inches": -12.0, "z_inches": -22.0 },
          { "x_inches": -12.0, "z_inches": 22.0 },
          { "x_inches": -30.0, "z_inches": 22.0 }
        ]]
      },
      {
        "id": "red",
        "polygons": [[
          { "x_inches": 12.0, "z_inches": -22.0 },
          { "x_inches": 30.0, "z_inches": -22.0 },
          { "x_inches": 30.0, "z_inches": 22.0 },
          { "x_inches": 12.0, "z_inches": 22.0 }
        ]]
      }
    ],
    "rotationally_symmetric": true
  }
}"#;

fn bench_basic_100(c: &mut Criterion) {
    let params: EngineParams = serde_json::from_str(BASIC_100_JSON).unwrap();
    c.bench_function("generate_basic_100_steps", |b| {
        b.iter(|| generate(&params));
    });
}

fn bench_all_features(c: &mut Criterion) {
    let params: EngineParams = serde_json::from_str(ALL_FEATURES_JSON).unwrap();
    c.bench_function("generate_all_features", |b| {
        b.iter(|| generate(&params));
    });
}

fn bench_with_visibility(c: &mut Criterion) {
    let params: EngineParams = serde_json::from_str(VISIBILITY_50_JSON).unwrap();
    c.bench_function("generate_with_visibility_50", |b| {
        b.iter(|| generate(&params));
    });
}

fn bench_with_visibility_100(c: &mut Criterion) {
    // Same as VISIBILITY_50 but with 100 steps — more terrain accumulates,
    // making later visibility computations progressively heavier.
    let json = VISIBILITY_50_JSON.replace("\"num_steps\": 50", "\"num_steps\": 100");
    let params: EngineParams = serde_json::from_str(&json).unwrap();
    c.bench_function("generate_with_visibility_100", |b| {
        b.iter(|| generate(&params));
    });
}

fn bench_with_mission(c: &mut Criterion) {
    let params: EngineParams = serde_json::from_str(MISSION_HNA_JSON).unwrap();
    c.bench_function("generate_with_mission_hna", |b| {
        b.iter(|| generate(&params));
    });
}

criterion_group!(
    benches,
    bench_basic_100,
    bench_all_features,
    bench_with_visibility,
    bench_with_visibility_100,
    bench_with_mission
);
criterion_main!(benches);
