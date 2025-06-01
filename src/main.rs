use babygemma;
use babygemma::BgTensor;
use babygemma::compute_handle::Linear;
use babygemma::gpu::GpuContext;
use memmap2::Mmap;
use std::fs::File;

fn linear(input: &Vec<Vec<f32>>, kernel: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let n_in_features = input[0].len();
    let n_tokens = input.len();
    let n_out_features = kernel.len();
    let mut output_buffer = vec![vec![0.; n_out_features]; n_tokens];

    for j in 0..n_out_features {
        for i in 0..n_in_features {
            for k in 0..n_tokens {
                output_buffer[k][j] += input[k][i] * kernel[j][i];
            }
        }
    }
    output_buffer
}

fn main() {
    // let model_weights_file = File::open("model/model.safetensors").unwrap();
    // let model_weights_data = unsafe { Mmap::map(&model_weights_file).unwrap() };
    // let weights_map = babygemma::load_weights(&model_weights_data);
    // let linear = Linear::new(2, 3);
    // let ctx = GpuContext::new();
    // let input = vec![-0.0279, 0.9937];
    // let kernel = vec![0.0019, 0.9587, -0.3810, 0.6508, -0.5161, 1.2580];
    // let out = linear.forward(&ctx, &input, &kernel);
    // println!("{}", out.len());
    // println!("{out:?}")
    let input = vec![1.0959, -0.5172, 1.4105, 1.3620, -0.0860, 1.6156];

    let kernel = vec![
        -0.2377, -0.6112, 0.0929, -0.6267, 0.7036, -0.4276, -0.3583, 1.4472, -0.1927, 0.8928,
        -0.4741, -2.1613,
    ];
    let input_tensor = BgTensor {
        data: input,
        shape: vec![2, 3],
    };
    let kernel_tensor = BgTensor {
        data: kernel,
        shape: vec![4, 3],
    };
    let ctx = GpuContext::new();
    let linear = Linear::new(input_tensor, kernel_tensor);
    let out_tensor = linear.forward(&ctx);
    println!("{:?}", out_tensor)
}
