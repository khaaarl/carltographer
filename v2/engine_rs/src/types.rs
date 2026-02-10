//! Data types matching the carltographer JSON schema.
//!
//! Every struct here derives Serialize + Deserialize so it can
//! round-trip through the JSON interchange format.

use serde::{Deserialize, Serialize};

// -- Geometry / transforms -----------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Transform {
    #[serde(default)]
    pub x_inches: f64,
    #[serde(default)]
    pub y_inches: f64,
    #[serde(default)]
    pub z_inches: f64,
    #[serde(default)]
    pub rotation_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricShape {
    pub shape_type: String,
    pub width_inches: f64,
    pub depth_inches: f64,
    pub height_inches: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<Transform>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opacity_height_inches: Option<f64>,
}

impl GeometricShape {
    /// Return the height used for LOS blocking.
    ///
    /// When `opacity_height_inches` is set, it overrides physical height
    /// (e.g. a tall ruin with windows might only block at 3").
    /// When `None`, physical height is used (backward compatible).
    pub fn effective_opacity_height(&self) -> f64 {
        self.opacity_height_inches.unwrap_or(self.height_inches)
    }
}

// -- Objects / features --------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainObject {
    pub id: String,
    pub shapes: Vec<GeometricShape>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureComponent {
    pub object_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Transform>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainFeature {
    pub id: String,
    pub feature_type: String,
    pub components: Vec<FeatureComponent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

fn is_false(v: &bool) -> bool {
    !v
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacedFeature {
    pub feature: TerrainFeature,
    pub transform: Transform,
    #[serde(default, skip_serializing_if = "is_false")]
    pub locked: bool,
}

// -- Mission / Deployment ------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x_inches: f64,
    pub z_inches: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveMarker {
    pub id: String,
    pub position: Point2D,
    #[serde(default = "default_range_inches")]
    pub range_inches: f64,
}

fn default_range_inches() -> f64 {
    3.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentZone {
    pub id: String,
    #[serde(default)]
    pub polygons: Vec<Vec<Point2D>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mission {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub objectives: Vec<ObjectiveMarker>,
    #[serde(default)]
    pub deployment_zones: Vec<DeploymentZone>,
    #[serde(default)]
    pub rotationally_symmetric: bool,
}

// -- Layout --------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainLayout {
    pub table_width_inches: f64,
    pub table_depth_inches: f64,
    #[serde(default)]
    pub placed_features: Vec<PlacedFeature>,
    #[serde(default)]
    pub rotationally_symmetric: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub terrain_objects: Vec<TerrainObject>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visibility: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mission: Option<Mission>,
}

// -- Catalog -------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogObject {
    pub item: TerrainObject,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantity: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFeature {
    pub item: TerrainFeature,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantity: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TerrainCatalog {
    #[serde(default)]
    pub objects: Vec<CatalogObject>,
    #[serde(default)]
    pub features: Vec<CatalogFeature>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

// -- Engine I/O ----------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureCountPreference {
    pub feature_type: String,
    #[serde(default)]
    pub min: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<u32>,
}

fn default_weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringTargets {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overall_visibility_target: Option<f64>,
    #[serde(default = "default_weight")]
    pub overall_visibility_weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dz_visibility_target: Option<f64>,
    #[serde(default = "default_weight")]
    pub dz_visibility_weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dz_hidden_target: Option<f64>,
    #[serde(default = "default_weight")]
    pub dz_hidden_weight: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective_hidability_target: Option<f64>,
    #[serde(default = "default_weight")]
    pub objective_hidability_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineParams {
    pub seed: u64,
    pub table_width_inches: f64,
    pub table_depth_inches: f64,
    pub catalog: TerrainCatalog,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_steps: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub initial_layout: Option<TerrainLayout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_count_preferences: Option<Vec<FeatureCountPreference>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_feature_gap_inches: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_edge_gap_inches: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_all_feature_gap_inches: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_all_edge_gap_inches: Option<f64>,
    #[serde(default = "default_rotation_granularity")]
    pub rotation_granularity_deg: f64,
    #[serde(default)]
    pub rotationally_symmetric: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mission: Option<Mission>,
    #[serde(default)]
    pub skip_visibility: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scoring_targets: Option<ScoringTargets>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_replicas: Option<u32>,
    #[serde(default = "default_swap_interval")]
    pub swap_interval: u32,
    #[serde(default = "default_max_temperature")]
    pub max_temperature: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tuning: Option<TuningParams>,
    #[serde(default = "default_standard_blocking_height")]
    pub standard_blocking_height_inches: f64,
    #[serde(default = "default_infantry_blocking_height")]
    pub infantry_blocking_height_inches: Option<f64>,
}

fn default_rotation_granularity() -> f64 {
    15.0
}

fn default_swap_interval() -> u32 {
    20
}

fn default_max_temperature() -> f64 {
    50.0
}

fn default_standard_blocking_height() -> f64 {
    4.0
}
fn default_infantry_blocking_height() -> Option<f64> {
    Some(2.2)
}

fn default_max_retries() -> u32 {
    100
}
fn default_retry_decay() -> f64 {
    0.95
}
fn default_min_move_range() -> f64 {
    2.0
}
fn default_max_extra_mutations() -> u32 {
    3
}
fn default_tile_size() -> f64 {
    2.0
}
fn default_delete_weight_last() -> f64 {
    0.25
}
fn default_rotate_on_move_prob() -> f64 {
    0.5
}
fn default_shortage_boost() -> f64 {
    2.0
}
fn default_excess_boost() -> f64 {
    2.0
}
fn default_penalty_factor() -> f64 {
    0.1
}
fn default_phase2_base() -> f64 {
    10.0
}
fn default_temp_ladder_min_ratio() -> f64 {
    0.01
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningParams {
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_retry_decay")]
    pub retry_decay: f64,
    #[serde(default = "default_min_move_range")]
    pub min_move_range: f64,
    #[serde(default = "default_max_extra_mutations")]
    pub max_extra_mutations: u32,
    #[serde(default = "default_tile_size")]
    pub tile_size: f64,
    #[serde(default = "default_delete_weight_last")]
    pub delete_weight_last: f64,
    #[serde(default = "default_rotate_on_move_prob")]
    pub rotate_on_move_prob: f64,
    #[serde(default = "default_shortage_boost")]
    pub shortage_boost: f64,
    #[serde(default = "default_excess_boost")]
    pub excess_boost: f64,
    #[serde(default = "default_penalty_factor")]
    pub penalty_factor: f64,
    #[serde(default = "default_phase2_base")]
    pub phase2_base: f64,
    #[serde(default = "default_temp_ladder_min_ratio")]
    pub temp_ladder_min_ratio: f64,
}

impl Default for TuningParams {
    fn default() -> Self {
        Self {
            max_retries: 100,
            retry_decay: 0.95,
            min_move_range: 2.0,
            max_extra_mutations: 3,
            tile_size: 2.0,
            delete_weight_last: 0.25,
            rotate_on_move_prob: 0.5,
            shortage_boost: 2.0,
            excess_boost: 2.0,
            penalty_factor: 0.1,
            phase2_base: 10.0,
            temp_ladder_min_ratio: 0.01,
        }
    }
}

impl EngineParams {
    pub fn steps(&self) -> u32 {
        self.num_steps.unwrap_or(100)
    }

    pub fn tuning(&self) -> TuningParams {
        self.tuning.clone().unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineResult {
    pub layout: TerrainLayout,
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub steps_completed: u32,
}

// -- Tests ---------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_round_trip() {
        let json = r#"{
            "seed": 42,
            "table_width_inches": 60.0,
            "table_depth_inches": 44.0,
            "catalog": {
                "objects": [{
                    "item": {
                        "id": "crate_5x2.5",
                        "shapes": [{
                            "shape_type": "rectangular_prism",
                            "width_inches": 5.0,
                            "depth_inches": 2.5,
                            "height_inches": 2.0
                        }]
                    }
                }],
                "features": [{
                    "item": {
                        "id": "crate",
                        "feature_type": "obstacle",
                        "components": [{"object_id": "crate_5x2.5"}]
                    }
                }]
            },
            "num_steps": 100
        }"#;

        let params: EngineParams = serde_json::from_str(json).expect("deserialize");
        assert_eq!(params.seed, 42);
        assert_eq!(params.steps(), 100);
        assert_eq!(params.catalog.objects.len(), 1);
        assert_eq!(params.catalog.features.len(), 1);

        // Re-serialize and verify it's valid JSON
        let out = serde_json::to_string(&params).expect("serialize");
        let _: EngineParams = serde_json::from_str(&out).expect("re-deserialize");
    }

    #[test]
    fn result_serializes() {
        let result = EngineResult {
            layout: TerrainLayout {
                table_width_inches: 60.0,
                table_depth_inches: 44.0,
                placed_features: vec![],
                rotationally_symmetric: false,
                terrain_objects: vec![],
                visibility: None,
                mission: None,
            },
            score: 0.0,
            steps_completed: 50,
        };
        let json = serde_json::to_string(&result).expect("serialize");
        assert!(json.contains("\"table_width_inches\":60.0"));
        assert!(json.contains("\"steps_completed\":50"));
    }

    #[test]
    fn empty_catalog_deserializes() {
        let json = r#"{
            "seed": 1,
            "table_width_inches": 60.0,
            "table_depth_inches": 44.0,
            "catalog": {}
        }"#;
        let params: EngineParams = serde_json::from_str(json).expect("deserialize");
        assert_eq!(params.steps(), 100);
        assert!(params.catalog.objects.is_empty());
        assert!(params.catalog.features.is_empty());
    }
}
