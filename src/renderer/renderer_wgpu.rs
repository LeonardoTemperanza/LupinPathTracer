
mod wgsl_preprocessor;
use wgsl_preprocessor::*;

use crate::base::*;
//use crate::platform::*;

use winit::window::Window;

use egui::{ClippedPrimitive, TexturesDelta};

use crate::renderer::*;

include!("renderer_wgpu/shaders.rs");

pub struct Renderer<'a>
{
    instance: wgpu::Instance,
    surface:  wgpu::Surface<'a>,
    adapter:  wgpu::Adapter,
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    swapchain_format: wgpu::TextureFormat,
    present_mode: wgpu::PresentMode,
    width: i32,
    height: i32,

    next_frame: Option<wgpu::SurfaceTexture>,

    // EGUI
    egui_render_state: EGUIRenderState,

    // Common shader pipelines
    pathtracer: wgpu::ComputePipeline,
    show_texture: wgpu::RenderPipeline
}

pub struct EGUIRenderState
{
    renderer: egui_wgpu::Renderer,
}

pub struct GPUTimer
{
    query_set: wgpu::QuerySet,
    num_added_timestamps: u32,
    max_timestamps: u32
}

impl<'a> RendererImpl<'a> for Renderer<'a>
{
    fn new(window: &'a Window, init_width: i32, init_height: i32)->Self
    {
        use wgpu::*;

        let instance_desc = InstanceDescriptor
        {
            // Windows considerations:
            // The Vulkan backend seems to be
            // the best when debugging with RenderDoc,
            // but its resizing behavior is the worst
            // by far compared to opengl and dx12. It also
            // seems to have a bit of input lag for some reason.
            // (will have to double check on that.)
            // dx12 resizes fine with few artifacts.
            // The best one seems to be OpenGL (it doesn't
            // have any issues) but it's not really debuggable.
            #[cfg(target_os = "windows")]
            backends: wgpu::Backends::VULKAN,

            #[cfg(not(target_os = "windows"))]
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,

            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        };
        let instance: Instance = Instance::new(instance_desc);

        let maybe_surface = instance.create_surface(window);
        let mut surface: Surface = maybe_surface.expect("Failed to create WGPU surface");

        let adapter_options = RequestAdapterOptions
        {
            power_preference: PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        };
        let maybe_adapter = wait_for(instance.request_adapter(&adapter_options));
        let adapter: Adapter = maybe_adapter.expect("Failed to get adapter");

        let device_desc = DeviceDescriptor
        {
            label: None,
            required_features: Features::TIMESTAMP_QUERY,
            required_limits: Limits::default(),
        };
        let maybe_device_queue = wait_for(adapter.request_device(&device_desc, None));
        let (device, queue): (Device, Queue) = maybe_device_queue.expect("Failed to get device");

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let default_present_mode = PresentMode::Immediate;
        configure_surface(&mut surface, &device, swapchain_format, default_present_mode, init_width, init_height);

        // Compile shaders
        let mut preprocessor_params = PreprocessorParams::default();
        preprocessor_params.all_shader_names = &SHADER_NAMES;
        preprocessor_params.all_shader_contents = &SHADER_CONTENTS;
        let pathtracer_module = compile_shader(&device,
                                               Some("PathtracerShader"),
                                               "pathtracer.wgsl",
                                               preprocessor_params);

        let show_texture_module = compile_shader(&device,
                                                 Some("ShowTextureShader"),
                                                 "show_texture.wgsl",
                                                 preprocessor_params);

        // Create shader pipeline
        let pathtracer_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor
        {
            label: None,
            entries:
            &[
                BindGroupLayoutEntry
                {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::StorageTexture
                    {
                        access: StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2
                    },
                    count: None
                },
                BindGroupLayoutEntry
                {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer
                    {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None
                },
                BindGroupLayoutEntry
                {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer
                    {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None
                },
                BindGroupLayoutEntry
                {
                    binding: 3,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer
                    {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None
                },
                BindGroupLayoutEntry
                {
                    binding: 4,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer
                    {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None
                },
                BindGroupLayoutEntry
                {
                    binding: 5,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer
                    {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None
                },
            ]
        });

        let pathtracer_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor
        {
            label: None,
            bind_group_layouts: &[&pathtracer_bind_group_layout],
            push_constant_ranges: &[]
        });

        let pathtracer = device.create_compute_pipeline(&ComputePipelineDescriptor
        {
            label: Some("PathtracerPipeline"),
            layout: Some(&pathtracer_layout),
            module: &pathtracer_module,
            entry_point: "cs_main"
        });

        let show_texture_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor
        {
            label: None,
            entries:
            &[
                BindGroupLayoutEntry
                {
                    binding: 0,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Texture
                    {
                        sample_type: TextureSampleType::Float { filterable: true },
                        multisampled: false,
                        view_dimension: TextureViewDimension::D2
                    },
                    count: None
                },
                BindGroupLayoutEntry
                {
                    binding: 1,
                    visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None
                }
            ]
        });

        let show_texture_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor
        {
            label: None,
            bind_group_layouts: &[&show_texture_bind_group_layout],
            push_constant_ranges: &[]
        });

        let show_texture = device.create_render_pipeline(&RenderPipelineDescriptor
        {
            label: Some("ShowTexturePipeline"),
            layout: Some(&show_texture_layout),
            vertex: VertexState
            {
                module: &show_texture_module,
                entry_point: "vs_main",
                buffers: &[]
            },
            fragment: Some(FragmentState
            {
                module: &show_texture_module,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState
                {
                    format: swapchain_format,
                    blend: Some(BlendState::REPLACE),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            primitive: PrimitiveState
            {
                topology: PrimitiveTopology::TriangleList,
                ..Default::default()
            }
        });

        // Init egui info
        let renderer = egui_wgpu::Renderer::new(&device, swapchain_format, None, 1);
        let egui_render_state = EGUIRenderState { renderer };

        return Renderer
        {
            instance,
            surface,
            adapter,
            device,
            queue,
            swapchain_format,
            present_mode: default_present_mode,
            width: init_width,
            height: init_height,

            next_frame: None,

            // Egui
            egui_render_state,

            // Common shader pipelines
            pathtracer,
            show_texture,
        };
    }

    fn resize(&mut self, width: i32, height: i32)
    {
        self.width = width;
        self.height = height;

        use wgpu::*;
        if self.next_frame.is_some()
        {
            let next_frame = self.next_frame.take().unwrap();
            drop(next_frame);
        }

        configure_surface(&mut self.surface, &self.device, self.swapchain_format, self.present_mode, width, height);
    }
    
    fn draw_scene(&mut self, scene: &Scene, render_to: &Texture, camera_transform: Mat4)
    {
        use wgpu::*;
        let camera_transform_uniform = self.upload_uniform(to_u8_slice(&[camera_transform]));
        let surface = &self.surface;
        let device  = &self.device;
        let queue   = &self.queue;
        let (width, height) = (render_to.width(), render_to.height());

        let encoder_desc = CommandEncoderDescriptor
        {
            label: None,
        };
        let mut encoder = device.create_command_encoder(&encoder_desc);

        // Compute pass to generate image
        {
            let view = render_to.create_view(&wgpu::TextureViewDescriptor::default());

            let bind_group = device.create_bind_group(&BindGroupDescriptor
            {
                label: None,
                layout: &self.pathtracer.get_bind_group_layout(0),
                entries:
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&view) },
                    BindGroupEntry { binding: 1, resource: buffer_resource(&scene.verts_pos) },
                    BindGroupEntry { binding: 2, resource: buffer_resource(&scene.indices) },
                    BindGroupEntry { binding: 3, resource: buffer_resource(&scene.bvh_nodes) },
                    BindGroupEntry { binding: 4, resource: buffer_resource(&scene.verts) },
                    BindGroupEntry { binding: 5, resource: buffer_resource(&camera_transform_uniform) }
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor
            {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pathtracer);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            const WORKGROUP_SIZE_X: u32 = 8;
            const WORKGROUP_SIZE_Y: u32 = 8;
            let num_workers_x = (width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let num_workers_y = (height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn show_texture(&mut self, texture: &Texture)
    {
        use wgpu::*;        
        let surface = &self.surface;
        let device  = &self.device;
        let queue   = &self.queue;

        if self.next_frame.is_none() { return; }
        let next_frame = self.next_frame.as_ref().unwrap();
        let frame_view = next_frame.texture.create_view(&TextureViewDescriptor::default());

        let encoder_desc = CommandEncoderDescriptor
        {
            label: None,
        };
        let mut encoder = device.create_command_encoder(&encoder_desc);

        {
            let texture_view = texture.create_view(&Default::default());
            let sampler = device.create_sampler(&Default::default());

            let bind_group = device.create_bind_group(&BindGroupDescriptor
            {
                label: Some("Bind group"),
                layout: &self.show_texture.get_bind_group_layout(0),
                entries:
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&texture_view) },
                    BindGroupEntry { binding: 1, resource: BindingResource::Sampler(&sampler) },
                ],
            });

            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor
            {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment
                {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations
                    {
                        load: LoadOp::Clear(Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None
            });
            
            render_pass.set_pipeline(&self.show_texture);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));
    }

    fn draw_egui(&mut self,
                     tris: Vec<ClippedPrimitive>,
                     textures_delta: &TexturesDelta,
                     width: i32, height: i32, scale: f32)
    {
        let egui: &mut EGUIRenderState = &mut self.egui_render_state;

        use wgpu::*;

        if self.next_frame.is_none() { return; }
        let next_frame = self.next_frame.as_ref().unwrap();
        let frame_view = next_frame.texture.create_view(&TextureViewDescriptor::default());

        let win_width = width.max(1) as u32;
        let win_height = height.max(1) as u32;

        let encoder_desc = wgpu::CommandEncoderDescriptor { label: Some("EGUI Encoder") };
        let mut encoder = self.device.create_command_encoder(&encoder_desc);
        for (id, image_delta) in &textures_delta.set
        {
            egui.renderer.update_texture(&self.device, &self.queue, *id, &image_delta);
        }

        let screen_descriptor = egui_wgpu::ScreenDescriptor
        {
            size_in_pixels: [win_width, win_height],
            pixels_per_point: scale
        };
        egui.renderer.update_buffers(&self.device, &self.queue, &mut encoder, &tris, &screen_descriptor);

        let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor
        {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment
            {
                view: &frame_view,
                resolve_target: None,
                ops: wgpu::Operations
                {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            label: Some("EGUI Main Render Pass"),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        egui.renderer.render(&mut egui_pass, &tris, &screen_descriptor);
        drop(egui_pass);

        for x in &textures_delta.free
        {
            egui.renderer.free_texture(x)
        }

        self.queue.submit(Some(encoder.finish()));
    }

    fn begin_frame(&mut self)
    {
        use wgpu::*;
        if self.next_frame.is_some()
        {
            let next_frame = self.next_frame.take().unwrap();
        }

        self.next_frame = try_get_next_frame(&self.surface);
    }

    fn end_frame(&mut self)
    {
        use wgpu::*;
        if self.next_frame.is_some()
        {
            let next_frame = self.next_frame.take().unwrap();
            next_frame.present();
        }
    }

    fn upload_buffer(&mut self, buffer: &[u8])->wgpu::Buffer
    {
        use wgpu::*;
        let wgpu_buffer = self.device.create_buffer(&BufferDescriptor
        {
            label: None,
            size: buffer.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&wgpu_buffer, 0, buffer);

        return wgpu_buffer;
    }

    fn upload_uniform(&mut self, buffer: &[u8])->wgpu::Buffer
    {
        use wgpu::*;
        let wgpu_buffer = self.device.create_buffer(&BufferDescriptor
        {
            label: None,
            size: buffer.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&wgpu_buffer, 0, buffer);

        return wgpu_buffer;
    }

    fn create_empty_buffer(&mut self)->wgpu::Buffer
    {
        use wgpu::*;
        let buffer = self.device.create_buffer(&BufferDescriptor
        {
            label: None,
            size: 0,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        return buffer;
    }

    // Lets the user read a buffer from the GPU to the CPU. This will
    // cause latency so it should be used very sparingly if at all
    fn read_buffer(&mut self, buffer: wgpu::Buffer, output: &mut[u8])
    {
        assert!(buffer.size() == output.len() as u64);

        let gpu_read_buffer = self.device.create_buffer(&wgpu::BufferDescriptor
        {
            label: Some("TimerQueryBuffer"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let encoder_desc = wgpu::CommandEncoderDescriptor { label: None };
        let mut encoder = self.device.create_command_encoder(&encoder_desc);
        encoder.copy_buffer_to_buffer(&buffer, 0, &gpu_read_buffer, 0, buffer.size());

        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        // Wait for read 
        let buffer_slice = gpu_read_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read, |result|
        {
            assert!(result.is_ok());
        });
        self.device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        output[..].clone_from_slice(&data);
    }

    fn create_texture(&mut self, width: u32, height: u32)->wgpu::Texture
    {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let view_formats: Vec<wgpu::TextureFormat> = vec![format];
        let wgpu_desc = wgpu::TextureDescriptor
        {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            format: format,
            mip_level_count: 1,
            sample_count: 1,
            size:  wgpu::Extent3d { width: width, height: height, depth_or_array_layers: 1 },
            usage: wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::STORAGE_BINDING |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[format]
        };

        return self.device.create_texture(&wgpu_desc);
    }

    fn get_texture_size(texture: &Texture)->(i32, i32)
    {
        return (texture.width() as i32, texture.height() as i32);
    }

    fn resize_texture(&mut self, texture: &mut Texture, width: i32, height: i32)
    {
        let width = width.max(0) as u32;
        let height = height.max(0) as u32;

        let wgpu_desc = wgpu::TextureDescriptor
        {
            label: None,
            dimension: texture.dimension(),
            format: texture.format(),
            mip_level_count: texture.mip_level_count(),
            sample_count: texture.sample_count(),
            size:  wgpu::Extent3d { width: width, height: height, depth_or_array_layers: 1 },
            usage: texture.usage(),
            view_formats: &[texture.format()]
        };

        *texture = self.device.create_texture(&wgpu_desc);
    }

    fn texture_to_egui_texture(&mut self, texture: &Texture, filter_near: bool)->egui::TextureId
    {
        let filter_mode = if filter_near { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        return self.egui_render_state.renderer.register_native_texture(&self.device, &view, filter_mode);
    }

    fn update_egui_texture(&mut self, texture: &Texture, texture_id: egui::TextureId, filter_near: bool)
    {
        let filter_mode = if filter_near { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
        let egui_renderer = &mut self.egui_render_state.renderer;
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        egui_renderer.update_egui_texture_from_wgpu_texture(&self.device, &view, filter_mode, texture_id)
    }

    fn create_gpu_timer(&mut self, num_timestamps: u32)->GPUTimer
    {
        let query_set = self.device.create_query_set(&wgpu::QuerySetDescriptor
        {
            label: Some("TimerQuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: num_timestamps,
        });

        return GPUTimer { query_set, num_added_timestamps: 0, max_timestamps: num_timestamps }
    }

    fn add_timestamp(&mut self, timer: &mut GPUTimer)
    {
        assert!(timer.num_added_timestamps < timer.max_timestamps);

        let encoder_desc = wgpu::CommandEncoderDescriptor { label: Some("TimerEncoder") };
        let mut encoder = self.device.create_command_encoder(&encoder_desc);
        encoder.write_timestamp(&timer.query_set, timer.num_added_timestamps);

        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        timer.num_added_timestamps += 1;
    }

    fn get_gpu_times(&mut self, timer: &GPUTimer, times: &mut [f32])
    {
        assert!(times.len() == timer.max_timestamps as usize - 1);

        // Create a buffer to resolve query results
        let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor
        {
            label: Some("TimerQueryBuffer"),
            size: 8 * timer.max_timestamps as u64,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC |
                   wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let encoder_desc = wgpu::CommandEncoderDescriptor { label: Some("TimerEncoder") };
        let mut encoder = self.device.create_command_encoder(&encoder_desc);

        // Resolve the query set to the buffer
        encoder.resolve_query_set(&timer.query_set, 0..timer.max_timestamps as u32, &query_buffer, 0);
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        // Read the result back on the CPU
        let mut read_data: Vec<u8> = vec![0; query_buffer.size() as usize];
        self.read_buffer(query_buffer, &mut read_data);

        let timestamps: &[u64] = to_u64_slice(&read_data);

        for i in 0..(timer.max_timestamps - 1) as usize
        {
            // This elapsed time is in nanoseconds
            let elapsed = timestamps[i+1] - timestamps[i];
            // Convert to milliseconds
            times[i] = elapsed as f32 / 1_000_000.0;
        }
    }

    fn set_vsync(&mut self, flag: bool)
    {
        use wgpu::*;

        if flag
        {
            self.present_mode = PresentMode::Fifo;
        }
        else
        {
            self.present_mode = PresentMode::Immediate;
        }

        configure_surface(&mut self.surface, &self.device, self.swapchain_format, self.present_mode, self.width, self.height);
    }

    fn log_backend(&self)
    {
        print!("Using WGPU, with backend: ");
        let backend = self.adapter.get_info().backend;
        match backend
        {
            wgpu::Backend::Empty  => { println!("Empty"); }
            wgpu::Backend::Vulkan => { println!("Vulkan"); }
            wgpu::Backend::Metal  => { println!("Metal"); }
            wgpu::Backend::Dx12   => { println!("D3D12"); }
            wgpu::Backend::Gl     => { println!("OpenGL"); }
            wgpu::Backend::BrowserWebGpu => { println!("WebGPU"); }
        }
    }
} 

////////
// WGPU specific utils
fn buffer_resource(buffer: &wgpu::Buffer)->wgpu::BindingResource
{
    use wgpu::*;
    return BindingResource::Buffer(BufferBinding
    {
        buffer: buffer,
        offset: 0,
        size: None
    })
}

fn try_get_next_frame(surface: &wgpu::Surface)->Option<wgpu::SurfaceTexture>
{
    use wgpu::*;
    return match surface.get_current_texture()
    {
        Ok(next_frame) => Some(next_frame),
        Err(SurfaceError::Outdated) => { println!("Outdated!"); None },  // This happens on some platforms when minimized
        Err(e) =>
        {
            eprintln!("Dropped frame with error: {}", e);
            None
        },
    }
}

fn configure_surface(surface: &mut wgpu::Surface,
                     device: &wgpu::Device,
                     format: wgpu::TextureFormat,
                     present_mode: wgpu::PresentMode,
                     width: i32, height: i32)
{
    use wgpu::*;

    // Don't panic when trying to resize to 0 (e.g. when minimizing)
    let des_width: u32  = width.max(1) as u32;
    let des_height: u32 = height.max(1) as u32;
    let surface_config = wgpu::SurfaceConfiguration
    {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: format,
        width: des_width,
        height: des_height,
        present_mode: present_mode,
        desired_maximum_frame_latency: 0,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![format],
    };

    // TODO: This should be better. We should check if this was successful
    // and if not, revert to the default configuration for the surface
    // (surface.get_default_config)
    surface.configure(&device, &surface_config);
}

fn compile_shader(device: &wgpu::Device,
                  label: Option<&str>,
                  shader_name: &str,
                  preprocessor_params: PreprocessorParams)->wgpu::ShaderModule
{
    use wgpu::*;

    let default_shader = "@compute @workgroup_size(1, 1, 1) fn cs_main() {}
                          @vertex fn vs_main()->@builtin(position) vec4f { return vec4f(0.0f); }
                          @fragment fn fs_main() {}";

    let (preprocessed_src, src_map) = preprocess_shader(shader_name, default_shader, preprocessor_params);

    let desc = ShaderModuleDescriptor
    {
        label: label,
        source: ShaderSource::Wgsl(preprocessed_src.into()),
    };

    // Catch compilation errors
    device.push_error_scope(ErrorFilter::Validation);

    // Omit runtime checks on shaders on release builds
    #[cfg(debug_assertions)]
    let module = device.create_shader_module(desc);
    #[cfg(not(debug_assertions))]
    let module = unsafe { device.create_shader_module_unchecked(desc) };

    let error_future = device.pop_error_scope();
    // Shaders are compiled on the CPU synchronously so we're not really waiting for anything
    let error = wait_for(error_future);

    if error.is_some()
    {
        let error = error.unwrap();
        
        // TODO: Translate shader error
        println!("Shader Compilation Error: {}", error);

        // Compile a default shader instead
        let module = device.create_shader_module(ShaderModuleDescriptor
        {
            label: Some("DefaultShader"),
            source: ShaderSource::Wgsl(default_shader.into())
        });

        return module;
    }

    return module;
}
