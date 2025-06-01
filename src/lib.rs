use half::bf16;
use safetensors::SafeTensors;
use std::collections::HashMap;
pub mod compute_handle;
pub mod gpu;

#[derive(Debug)]
pub struct BgTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub fn load_weights(data: &[u8]) -> HashMap<String, BgTensor> {
    let tensor_data = SafeTensors::deserialize(data).unwrap();
    let mut map = HashMap::new();
    for (weight_name, tensor) in tensor_data.tensors() {
        let f32_weights: Vec<f32> = tensor
            .data()
            .chunks(2)
            .map(|chunk| {
                let float = bf16::from_le_bytes(chunk.try_into().unwrap());
                float.to_f32()
            })
            .collect();
        let shape = tensor.shape().to_vec();
        let tensor = BgTensor {
            data: f32_weights,
            shape,
        };
        map.insert(weight_name, tensor);
    }
    map
}
