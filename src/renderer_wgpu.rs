
use crate::base::*;

use winit::window::Window;

use egui::{ClippedPrimitive, TexturesDelta};

pub struct Renderer<'a>
{
    instance: wgpu::Instance,
    surface:  wgpu::Surface<'a>,
    adapter:  wgpu::Adapter,
    device:   wgpu::Device,
    queue:    wgpu::Queue,
    swapchain_format: wgpu::TextureFormat,

    frame_view:    Option<wgpu::TextureView>,
    frame_texture: Option<wgpu::SurfaceTexture>,
}

pub struct EGUIRenderState
{
    renderer: egui_wgpu::Renderer,
}

// Info stored on the CPU to upload to GPU
pub struct SceneParams
{
    // Vector or images, vector of models
    // etc. etc.
}

// Handles for GPU resources used for rendering
pub struct Scene
{
    compute_program: wgpu::RenderPipeline,
    display_program: wgpu::RenderPipeline,
}

pub fn configure_surface(surface: &mut wgpu::Surface,
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

impl<'a> Renderer<'a>
{
    ////////
    // Initialization
    pub fn new(window: &'a Window)->Self
    {
        use wgpu::*;

        let instance_desc = InstanceDescriptor::default();
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
            required_features: Features::empty(),
            required_limits: Limits::default(),
        };
        let maybe_device_queue = wait_for(adapter.request_device(&device_desc, None));
        let (device, queue): (Device, Queue) = maybe_device_queue.expect("Failed to get device");

        let pipeline_layout_desc = PipelineLayoutDescriptor
        {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        };
        let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_desc);

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let win_size = window.inner_size();
        configure_surface(&mut surface, &device, swapchain_format, win_size.width as i32, win_size.height as i32);

        return Renderer
        {
            instance,
            surface,
            adapter,
            device,
            queue,
            swapchain_format,

            frame_view: None,
            frame_texture: None,
        };
    }

    pub fn init_egui(&self)->EGUIRenderState
    {
        let renderer = egui_wgpu::Renderer::new(&self.device, self.swapchain_format, None, 1);
        return EGUIRenderState { renderer };
    }

    ////////
    // Utils
    pub fn resize(&mut self, width: i32, height: i32)
    {
        configure_surface(&mut self.surface, &self.device, self.swapchain_format, width, height);
    }

    pub fn prepare_frame(&mut self)
    {
        use wgpu::*;

        self.frame_texture = None;
        self.frame_view = None;

        let frame = match self.surface.get_current_texture()
        {
            Ok(frame) => frame,
            Err(SurfaceError::Outdated) => { return; },  // This happens on some platforms when minimized
            Err(e) =>
            {
                eprintln!("Dropped frame with error: {}", e);
                return;
            },
        };

        let view = frame.texture.create_view(&TextureViewDescriptor::default());
        self.frame_texture = Some(frame);
        self.frame_view    = Some(view);
    }

    ////////
    // Rendering
    // Will later take a scene
    pub fn draw_scene(&mut self) //main_shader: ShaderHandle)
    {
        use wgpu::*;
        let surface = &self.surface;
        let device  = &self.device;
        let queue   = &self.queue;

        // Compute pass to generate image


        // Render pass to display the generated image on the screen

        let frame_view = self.frame_view.as_ref().unwrap();
        let encoder_desc = CommandEncoderDescriptor
        {
            label: None,
        };
        let mut encoder = device.create_command_encoder(&encoder_desc);

        {
            let color_attachment = RenderPassColorAttachment
            {
                view: frame_view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::GREEN),
                    store: StoreOp::Store,
                },
            };

            let render_pass_desc = RenderPassDescriptor
            {
                label: None,
                color_attachments: &[Some(color_attachment)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            };
            let render_pass = encoder.begin_render_pass(&render_pass_desc);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    pub fn draw_egui(&mut self, egui_renderer: &mut EGUIRenderState,
                     tris: Vec<ClippedPrimitive>,
                     textures_delta: &TexturesDelta,
                     width: i32, height: i32, scale: f32)
    {
        let frame_view = self.frame_view.as_ref().unwrap();
        let egui: &mut EGUIRenderState = egui_renderer;

        let win_width = width.max(0) as u32;
        let win_height = height.max(0) as u32;

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

        let mut egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: frame_view,
                resolve_target: None,
                ops: wgpu::Operations {
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

    pub fn swap_buffers(&mut self)
    {
        let texture = self.frame_texture.take();
        texture.unwrap().present();
    }

    ////////
    // Upload to GPU

    // Uploads the entire scene to GPU
    pub fn upload_scene(&mut self)
    {
        // WebGPU does not support texture arrays yet, so that's why
        // that is not supported by this renderer. Texture atlasing
        // is required to dynamically index into textures
    }
}
