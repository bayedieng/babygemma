use babygemma;
use std::fs;

fn main() {
    let model_weights_data = fs::read("model/model.safetensors").unwrap();
    let weights_map = babygemma::load_weights(&model_weights_data);
    weights_map.keys().for_each(|key| println!("{key}"));
}
