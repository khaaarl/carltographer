//! Carltographer terrain generation engine — Rust implementation.
//!
//! Exposes a single Python-callable function `generate_json` that
//! accepts a JSON string (engine_params) and returns a JSON string
//! (engine_result).

use pyo3::prelude::*;

pub mod collision;
pub mod generate;
mod mutation;
pub mod prng;
pub mod tempering;
pub mod types;
pub mod visibility;

/// Run the terrain generation engine.
///
/// Takes a JSON string matching the `engine_params` schema and
/// returns a JSON string matching the `engine_result` schema.
#[pyfunction]
fn generate_json(params_json: &str) -> PyResult<String> {
    let params: types::EngineParams = serde_json::from_str(params_json)
        .map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid engine_params JSON: {e}"
            ))
        })?;

    let result = generate::generate(&params);

    serde_json::to_string(&result).map_err(|e| {
        PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Failed to serialize engine_result: {e}"
        ))
    })
}

/// Carltographer Rust engine, importable from Python.
#[pymodule]
fn engine_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_json, m)?)?;
    Ok(())
}
