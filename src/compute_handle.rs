use crate::BgTensor;
use crate::gpu::GpuContext;

pub struct LinearParams {
    pub in_features: usize,
    pub out_features: usize,
    pub n_tokens: usize,
}

pub struct Linear {
    params: LinearParams,
    input: BgTensor,  // [n_tokens, in_features]
    kernel: BgTensor, // [out_features, in_features]
}

impl Linear {
    pub fn new(input: BgTensor, kernel: BgTensor) -> Self {
        let n_tokens = input.shape[0];
        let out_features = kernel.shape[0];
        let in_features = input.shape[1];
        let params = LinearParams {
            in_features,
            out_features,
            n_tokens,
        };

        Self {
            params,
            input,
            kernel,
        }
    }

    pub fn forward(&self, ctx: &GpuContext) -> BgTensor {
        ctx.run_linear_layer(&self.input, &self.kernel, &self.params)
    }
}
