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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacedFeature {
    pub feature: TerrainFeature,
    pub transform: Transform,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visibility: Option<serde_json::Value>,
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
    #[serde(default)]
    pub rotationally_symmetric: bool,
}

impl EngineParams {
    pub fn steps(&self) -> u32 {
        self.num_steps.unwrap_or(100)
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

        let params: EngineParams =
            serde_json::from_str(json).expect("deserialize");
        assert_eq!(params.seed, 42);
        assert_eq!(params.steps(), 100);
        assert_eq!(params.catalog.objects.len(), 1);
        assert_eq!(params.catalog.features.len(), 1);

        // Re-serialize and verify it's valid JSON
        let out = serde_json::to_string(&params).expect("serialize");
        let _: EngineParams =
            serde_json::from_str(&out).expect("re-deserialize");
    }

    #[test]
    fn result_serializes() {
        let result = EngineResult {
            layout: TerrainLayout {
                table_width_inches: 60.0,
                table_depth_inches: 44.0,
                placed_features: vec![],
                rotationally_symmetric: false,
                visibility: None,
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
        let params: EngineParams =
            serde_json::from_str(json).expect("deserialize");
        assert_eq!(params.steps(), 100);
        assert!(params.catalog.objects.is_empty());
        assert!(params.catalog.features.is_empty());
    }
}
