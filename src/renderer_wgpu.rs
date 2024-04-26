
extern crate wgpu;
extern crate winit;
use winit::window::Window;

extern crate egui_wgpu_backend;

use crate::base::*;

pub struct Renderer<'a>
{
    instance:  wgpu::Instance,
    surface:   wgpu::Surface<'a>,
    adapter:   wgpu::Adapter,
    device:    wgpu::Device,
    queue:     wgpu::Queue
}

type TextureHandle = u32;
type BufferHandle  = u32;
type ShaderHandle  = u32;

pub fn init<'a>(window: &'a Window)->Renderer<'a>
{
    use wgpu::*;

    let instance_desc = InstanceDescriptor::default();
    let instance: Instance = Instance::new(instance_desc);

    let maybe_surface = instance.create_surface(window);
    let surface: Surface = maybe_surface.expect("Failed to create WGPU surface");

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

/*
    let pipeline_desc = wgpu::RenderPipelineDescriptor
    {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: VertexState::default(),
        fragment: None,
        primitive: PrimitiveState::default(),
        depth_stencil: None,
        multisample: MultisampleState::default(),
        multiview: None,
    };
    let pipeline = device.create_render_pipeline(&pipeline_desc);
*/

    return Renderer
    {
        instance,
        surface,
        adapter,
        device,
        queue
    };
}

pub fn init_egui(renderer: &Renderer)
{
    use egui_wgpu_backend::{RenderPass, ScreenDescriptor};
    let mut egui_rpass = RenderPass::new(&renderer.device, surface_format);
}

pub fn resize(r: &mut Renderer, width: u32, height: u32)
{
    let win_width = width.max(1);
    let win_height = height.max(1);
    let maybe_surface_config = r.surface.get_default_config(&r.adapter, win_width, win_height);
    let surface_config = maybe_surface_config.expect("Failed to get surface default configuration");
    r.surface.configure(&r.device, &surface_config);
}

pub fn upload_texture(r: &mut Renderer)->TextureHandle
{
    return 0;
}

pub fn upload_model()->BufferHandle
{
    return 0;
}

pub fn compile_shader()->ShaderHandle
{
    return 0;
}

pub fn draw_scene(renderer: &mut Renderer)
{
    use wgpu::*;
    let surface  = &renderer.surface;
    let device   = &renderer.device;
    let queue    = &renderer.queue;

    let frame = surface.get_current_texture().expect("Failed to acquire next swapchain texture");
    let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

    let encoder_desc = CommandEncoderDescriptor
    {
        label: None,
    };
    let mut encoder = device.create_command_encoder(&encoder_desc);

    {
        let color_attachment = RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(wgpu::Color::GREEN),
                store: StoreOp::Store,
            },
        };

        let render_pass_desc = &wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let render_pass = encoder.begin_render_pass(&render_pass_desc);
        //render_pass.set_pipeline(&pipeline);
        //render_pass.draw(1..0, 1..0);
    }

    queue.submit(Some(encoder.finish()));
    frame.present();
}
