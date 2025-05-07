use half::bf16;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;

pub fn load_weights(data: &[u8]) -> HashMap<String, Vec<f32>> {
    let tensor_data = SafeTensors::deserialize(data).unwrap();
    let mut map = HashMap::new();
    for (weight_name, tensor) in tensor_data.tensors() {
        let f32_weights: Vec<f32> = tensor
            .data()
            .chunks(2)
            .map(|chunk| bf16::from_le_bytes(chunk.try_into().unwrap()))
            .map(|x| x.to_f32())
            .collect();
        map.insert(weight_name, f32_weights);
    }
    map
}
