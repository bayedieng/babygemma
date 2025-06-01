use crate::{BgTensor, compute_handle::LinearParams};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()));

        let downlevel_capabillities = adapter.as_ref().unwrap().get_downlevel_capabilities();
        if !downlevel_capabillities
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS)
        {
            panic!("Adapter does not support compute shaders")
        }

        let (device, queue) = pollster::block_on(adapter.unwrap().request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("Failed to create device");

        Self { device, queue }
    }

    pub fn run_linear_layer(
        &self,
        input: &BgTensor,
        kernel: &BgTensor,
        params: &LinearParams,
    ) -> BgTensor {
        // [in_features, out_features]
        let output = vec![0.; params.in_features * params.out_features];

        let input_buffer_slice = unsafe {
            std::slice::from_raw_parts(
                input.data.as_ptr() as *const u8,
                input.data.len() * size_of::<f32>(),
            )
        };
        let kernel_buffer_slice = unsafe {
            std::slice::from_raw_parts(
                kernel.data.as_ptr() as *const u8,
                kernel.data.len() * size_of::<f32>(),
            )
        };
        let output_buffer_slice = unsafe {
            std::slice::from_raw_parts(
                output.as_ptr() as *const u8,
                output.len() * size_of::<f32>(),
            )
        };
        let input_wgpu_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Input Buffer"),
                contents: input_buffer_slice,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let kernel_wgpu_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Kernel Buffer"),
                    contents: kernel_buffer_slice,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_wgpu_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Output Buffer"),
                    contents: output_buffer_slice,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                });

        let shader_src = format!(
            "
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read> kernel: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;

        @compute @workgroup_size(64) 
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
            let glob_i = global_id.x;
            let n_in_features = u32({});
            let n_out_features = u32({});
            let n_tokens: u32 = u32({});

            if (glob_i >= n_out_features) {{
                return;
            }}
            for (var i: u32; i <= n_in_features; i++) {{
                output[n_tokens * n_out_features + glob_i] += input[n_tokens * n_in_features + i] * kernel[glob_i * n_in_features + i];
            }}
            }}
              
        ",
            params.in_features, params.out_features, params.n_tokens
        );
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.as_str().into()),
            });

        // supposedly needed to read the buffer from CPU. we'll see.
        let read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Read Buffer"),
            size: output_wgpu_buffer.size(),
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });

        let input_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let kernel_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let output_bind_group_layout_entry = wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: Some(NonZeroU64::new(4).unwrap()),
            },
            count: None,
        };

        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        input_bind_group_layout_entry,
                        kernel_bind_group_layout_entry,
                        output_bind_group_layout_entry,
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_wgpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel_wgpu_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_wgpu_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(64, 1, 1);
        drop(compute_pass);

        encoder.copy_buffer_to_buffer(
            &output_wgpu_buffer,
            0,
            &read_buffer,
            0,
            output_wgpu_buffer.size(),
        );

        let command_buffer = encoder.finish();
        self.queue.submit([command_buffer]);

        let buffer_slice = read_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

        self.device.poll(wgpu::Maintain::Wait);

        let data: Vec<f32> = buffer_slice
            .get_mapped_range()
            .chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let ret_tensor = BgTensor {
            data,
            shape: vec![params.n_tokens, params.out_features],
        };

        ret_tensor
    }
}
