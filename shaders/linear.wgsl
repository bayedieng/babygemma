@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let glob_i = global_id.x;
    let input_len = arrayLength(&input);
    let kernel_len = arrayLength(&kernel);
    let output_len = arrayLength(&output);
    if (glob_i >= output_len) {
        return;
    }
    
    for (var i: u32; i <= input_len; i++) {
        output[glob_i] += kernel[glob_i + i] * input[i];
    }
    
    
}

