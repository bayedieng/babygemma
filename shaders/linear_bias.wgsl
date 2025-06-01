@group(0) @binding(0) input: array<f32>;
@group(0) @binding(1) kernel: array<f32>;
@group(0) @binding(2) bias: array<f32>;
@group(0) @binding(3) output: array<f32>;

@compute @workgroup(64) 
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let glob_i = global_id.x;
    let input_len = arrayLength(input);
    let kernel_len = arrayLength(kernel);
    let output_len = arrayLength(output);
    let bias_len = arrayLength(bias);
    if (glob_i >= output_len) {
        return 
    }
   for (var i = 0; i < input_len; i++) {
        output[glob_i] += input[i + glob_i] * kernel[i + kernel_len + glob_i] + bias[glob_i];
    }
}

