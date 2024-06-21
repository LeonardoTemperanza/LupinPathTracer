
use crate::base::*;
//use crate::platform::*;

use winit::window::Window;

use egui::{ClippedPrimitive, TexturesDelta};

use crate::renderer::*;

pub struct Renderer<'a>
{
    instance: wgpu::Instance,
    surface:  wgpu::Surface<'a>,
    adapter:  wgpu::Adapter,
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    swapchain_format: wgpu::TextureFormat,

    next_frame: Option<wgpu::SurfaceTexture>,

    // EGUI
    egui_render_state: EGUIRenderState,

    // Common shader pipelines
    //pathtrace_pipelines: [wgpu::ComputePipeline; PathtraceMode::Count as usize],
    pathtracer: wgpu::ComputePipeline,
    pathtracer_layout: wgpu::BindGroupLayout,

    // Profiling
    timer_query_set: wgpu::QuerySet
}

pub struct EGUIRenderState
{
    renderer: egui_wgpu::Renderer,
}

pub struct Texture
{
    desc:   TextureDesc,
    handle: wgpu::Texture,
    view:   wgpu::TextureView,
}

pub struct TextureDesc
{
    label: Option<String>,
    size: wgpu::Extent3d,
    mip_level_count: u32,
    sample_count: u32,
    dimension: wgpu::TextureDimension,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
}

impl<'a> RendererImpl<'a> for Renderer<'a>
{
    ////////
    // Initialization

    fn new(window: &'a Window, init_width: i32, init_height: i32)->Self
    {
        use wgpu::*;

        let instance_desc = InstanceDescriptor
        {
            // Windows considerations:
            // The Vulkan backend seems to be
            // the best when debugging with RenderDoc,
            // but its resizing behavior is the worst
            // by far compared to opengl and dx12.
            // dx12 resizes fine with few artifacts but
            // takes a while to start up (though that might
            // be an implementation issue). The best one seems
            // to be OpenGL (it doesn't have any issues)
            // but it's not really debuggable.
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

        configure_surface(&mut surface, &device, swapchain_format, init_width, init_height);

        // Get first frame to render to
        let next_frame = try_get_next_frame(&surface);

        // Compile all shader variations
        let pathtracer_module = device.create_shader_module(ShaderModuleDescriptor
        {
            label: Some("PathtracerShader"),
            source: ShaderSource::Wgsl(include_str!("../shaders/pathtracer.wgsl").into()),
        });

        // Create shader pipeline
        let pathtracer_bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
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
            entry_point: "main"
        });

        // Init egui info
        let renderer = egui_wgpu::Renderer::new(&device, swapchain_format, None, 1);
        let egui_render_state = EGUIRenderState { renderer };

        // Profiling
        let timer_query_set = device.create_query_set(&wgpu::QuerySetDescriptor
        {
            label: Some("TimerQuerySet"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,  // Two queries are used, one for start time and one for end time
        });

        return Renderer
        {
            instance,
            surface,
            adapter,
            device,
            queue,
            swapchain_format,

            next_frame,

            // Egui
            egui_render_state,

            // Common shader pipelines
            pathtracer,
            pathtracer_layout: pathtracer_bind_group_layout,

            // Profiling
            timer_query_set
        };
    }

    ////////
    // Utils

    fn resize(&mut self, width: i32, height: i32)
    {
        use wgpu::*;
        if self.next_frame.is_some()
        {
            let next_frame = self.next_frame.take().unwrap();
            drop(next_frame);
        }

        configure_surface(&mut self.surface, &self.device, self.swapchain_format, width, height);
    }

    fn create_texture(&mut self, width: u32, height: u32)->Texture
    {
        let format = wgpu::TextureFormat::Rgba8Unorm;

        let view_formats: Vec<wgpu::TextureFormat> = vec![format];
        let texture_desc = TextureDesc
        {
            label: None,
            dimension: wgpu::TextureDimension::D2,
            format: format,
            mip_level_count: 1,
            sample_count: 1,
            size: wgpu::Extent3d { width: width, height: height, depth_or_array_layers: 1 },
            usage: wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::STORAGE_BINDING |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
        };

        let wgpu_desc = wgpu::TextureDescriptor
        {
            label: texture_desc.label.as_deref(),
            dimension: texture_desc.dimension,
            format: texture_desc.format,
            mip_level_count: texture_desc.mip_level_count,
            sample_count: texture_desc.sample_count,
            size:  texture_desc.size,
            usage: texture_desc.usage,
            view_formats: &[texture_desc.format]
        };

        let texture = self.device.create_texture(&wgpu_desc);

        let view = texture.create_view(&wgpu::TextureViewDescriptor
        {
            format: Some(format),
            ..Default::default()
        });

        return Texture { desc: texture_desc, handle: texture, view };
    }

    fn resize_texture(&mut self, texture: &mut Texture, width: i32, height: i32)
    {
        texture.desc.size.width  = width.max(1) as u32;
        texture.desc.size.height = height.max(1) as u32;
        texture.handle.destroy();

        let texture_desc = &texture.desc;
        let wgpu_desc = wgpu::TextureDescriptor
        {
            label: texture_desc.label.as_deref(),
            dimension: texture_desc.dimension,
            format: texture_desc.format,
            mip_level_count: texture_desc.mip_level_count,
            sample_count: texture_desc.sample_count,
            size:  texture_desc.size,
            usage: texture_desc.usage,
            view_formats: &[texture_desc.format]
        };

        texture.handle = self.device.create_texture(&wgpu_desc);

        texture.view = texture.handle.create_view(&wgpu::TextureViewDescriptor
        {
            format: Some(texture.desc.format),
            ..Default::default()
        });
    }

    fn texture_to_egui_texture(&mut self, texture: &Texture, filter_near: bool)->egui::TextureId
    {
        let filter_mode = if filter_near { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
        return self.egui_render_state.renderer.register_native_texture(&self.device, &texture.view, filter_mode);
    }

    fn update_egui_texture(&mut self, texture: &Texture, texture_id: egui::TextureId, filter_near: bool)
    {
        let filter_mode = if filter_near { wgpu::FilterMode::Nearest } else { wgpu::FilterMode::Linear };
        let egui_renderer = &mut self.egui_render_state.renderer;
        egui_renderer.update_egui_texture_from_wgpu_texture(&self.device, &texture.view, filter_mode, texture_id)
    }

    ////////
    // Rendering

    // Will later take a scene as input
    fn draw_scene(&mut self, scene: &Scene, render_to: &Texture)
    {
        use wgpu::*;
        let surface = &self.surface;
        let device  = &self.device;
        let queue   = &self.queue;
        let (width, height) = (render_to.desc.size.width, render_to.desc.size.height);

        let encoder_desc = CommandEncoderDescriptor
        {
            label: None,
        };
        let mut encoder = device.create_command_encoder(&encoder_desc);

        // Compute pass to generate image
        {
            let bind_group = device.create_bind_group(&BindGroupDescriptor
            {
                label: None,
                layout: &self.pathtracer_layout,
                entries:
                &[
                    BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&render_to.view) },
                    BindGroupEntry { binding: 1, resource: buffer_resource(&scene.verts_pos) },
                    BindGroupEntry { binding: 2, resource: buffer_resource(&scene.indices) },
                    BindGroupEntry { binding: 3, resource: buffer_resource(&scene.bvh_nodes) },
                    BindGroupEntry { binding: 4, resource: buffer_resource(&scene.verts) },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor
            {
                label: None,
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&self.pathtracer);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            const WORKGROUP_SIZE_X: u32 = 16;
            const WORKGROUP_SIZE_Y: u32 = 16;
            let num_workers_x = (width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let num_workers_y = (height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
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

    ////////
    // CPU <-> GPU transfers

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

    ////////
    // Miscellaneous

    fn draw_test_triangle(&self)
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

        let shader_source = "
        @vertex
        fn vert_main(@builtin(vertex_index) VertexIndex : u32)->@builtin(position) vec4f {
          var pos = array<vec2f, 3>
          (
            vec2(-1.0, 1),
            vec2(-1.0, 0.0),
            vec2(0.0, 1.0)
          );

          return vec4f(pos[VertexIndex], 0.0, 1.0);
        }

        @fragment
        fn frag_main(@builtin )->@location(0) vec4f
        {
          return vec4(1.0, 0.0, 0.0, 1.0);
        }
        ";

        let module = device.create_shader_module(ShaderModuleDescriptor
        {
            label: None,
            source: ShaderSource::Wgsl(shader_source.into())
        });

        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor
        {
            label: None,
            layout: None,
            vertex: VertexState
            {
                module: &module,
                entry_point: "vert_main",
                buffers: &[]
            },
            fragment: Some(FragmentState
            {
                module: &module,
                entry_point: "frag_main",
                targets: &[Some(ColorTargetState
                {
                    format: self.swapchain_format,
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

        {
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
            
            render_pass.set_pipeline(&pipeline);
            render_pass.draw(0..3, 0..1);
        }

        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));
    }

    fn gpu_timer_begin(&self)
    {
        let encoder_desc = wgpu::CommandEncoderDescriptor { label: Some("TimerEncoder") };
        let mut encoder = self.device.create_command_encoder(&encoder_desc);
        encoder.write_timestamp(&self.timer_query_set, 0);
        
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));
    }

    // Returns GPU time spent between the begin and end calls,
    // in milliseconds. This will make the CPU wait for
    // all the calls to be finished on the GPU side, so it can only
    // really be used for very simple profiling, and definitely not
    // in release builds
    fn gpu_timer_end(&mut self)->f32
    {
        let encoder_desc = wgpu::CommandEncoderDescriptor { label: Some("TimerEncoder") };
        let mut encoder = self.device.create_command_encoder(&encoder_desc);
        encoder.write_timestamp(&self.timer_query_set, 1);

        // Create a buffer to resolve query results
        let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor
        {
            label: Some("TimerQueryBuffer"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC |
                   wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Resolve the query set to the buffer
        encoder.resolve_query_set(&self.timer_query_set, 0..2, &query_buffer, 0);
        let command_buffer = encoder.finish();
        self.queue.submit(Some(command_buffer));

        // Read the result back on the CPU
        let mut read_data = Vec::<u8>::new();
        read_data.resize(query_buffer.size() as usize, 0);
        self.read_buffer(query_buffer, &mut read_data);

        let timestamps: &[u64] = to_u64_slice(&read_data);

        // This elapsed time is in nanoseconds
        let elapsed_time = timestamps[1] - timestamps[0];

        // Convert to milliseconds
        return elapsed_time as f32 / 1_000_000.0;
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

impl TextureImpl for Texture
{
    fn get_size(&self)->(u32, u32)
    {
        return (self.desc.size.width, self.desc.size.height);
    }
}

pub fn get_texture_size(texture: &Texture)->(u32, u32)
{
    return (texture.desc.size.width, texture.desc.size.height);
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
        present_mode: PresentMode::Fifo,
        desired_maximum_frame_latency: 0,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![format],
    };

    // TODO: This panics if the surface config isn't supported.
    // How to select one that is closest to this one or default back to something else?
    surface.configure(&device, &surface_config);
}