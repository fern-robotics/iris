use hf_hub::api::sync::ApiRepo;
use iris_core::utils::cuda_is_available;
use iris_core::{Device, Result};

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else {
        println!("Running on CPU; build with the cuda feature to use the GPU.");
        Ok(Device::Cpu)
    }
}

/// Loads all safetensors shards named by a Hugging Face index file.
pub fn hub_load_safetensors(repo: &ApiRepo, json_file: &str) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(iris_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(iris_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => iris_core::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => iris_core::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    safetensors_files
        .iter()
        .map(|file| repo.get(file).map_err(iris_core::Error::wrap))
        .collect()
}
