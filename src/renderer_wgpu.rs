
use winit::window::Window;

use crate::base::*;

pub type TextureHandle = u32;
pub type BufferHandle  = u32;
pub type ShaderHandle  = u32;
pub type ProgramHandle = u32;
pub type ComputeProgramHandle = u32;

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

    // Resources to be accessed with the handles
    shaders:          Vec<wgpu::ShaderModule>,
    programs:         Vec<wgpu::RenderPipeline>,
    compute_programs: Vec<wgpu::ComputePipeline>
}

pub struct EGUIRenderState
{
    pub render_pass: egui_wgpu_backend::RenderPass,
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
    // Initialization and drawing
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

            shaders:  vec![],
            programs: vec![],
            compute_programs: vec![]
        };
    }

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

    pub fn draw_scene(&mut self)
    {
        use wgpu::*;
        let surface  = &self.surface;
        let device   = &self.device;
        let queue    = &self.queue;

        let frame_view_ref = self.frame_view.as_ref().unwrap();
        let encoder_desc = CommandEncoderDescriptor
        {
            label: None,
        };
        let mut encoder = device.create_command_encoder(&encoder_desc);

        {
            let color_attachment = RenderPassColorAttachment {
                view: frame_view_ref,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(wgpu::Color::GREEN),
                    store: StoreOp::Store,
                },
            };

            let render_pass_desc = RenderPassDescriptor {
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

    pub fn init_egui(&self)->EGUIRenderState
    {
        use egui_wgpu_backend::{RenderPass, ScreenDescriptor};

        let render_pass = RenderPass::new(&self.device, self.swapchain_format, 1);
        return EGUIRenderState
        {
            render_pass
        };
    }

    pub fn draw_egui(&self,
                     egui_state: &mut EGUIRenderState,
                     textures_delta: &egui::TexturesDelta,
                     paint_jobs: Vec<egui::ClippedPrimitive>,
                     win_width: i32,
                     win_height: i32,
                     scale_factor: f32)
    {
        // Don't panic with 0 size (e.g. when minimized)
        let win_width_u32: u32  = win_width.max(1) as u32;
        let win_height_u32: u32 = win_height.max(1) as u32;

        use wgpu::*;
        use egui_wgpu_backend::ScreenDescriptor;

        let frame_view = self.frame_view.as_ref().unwrap();

        let encoder_desc = wgpu::CommandEncoderDescriptor::default();
        let mut encoder = self.device.create_command_encoder(&encoder_desc);

        // Upload all resources for the GPU.
        let screen_descriptor = ScreenDescriptor {
            physical_width: win_width_u32,
            physical_height: win_height_u32,
            scale_factor: scale_factor,
        };
        
        let res = egui_state.render_pass.add_textures(&self.device, &self.queue, textures_delta);
        res.expect("Failed to add textures in EGUI render pass");

        egui_state.render_pass.update_buffers(&self.device, &self.queue, &paint_jobs, &screen_descriptor);

        let res = egui_state.render_pass.execute(&mut encoder,
                                                 frame_view,
                                                 &paint_jobs,
                                                 &screen_descriptor,
                                                 None);

        res.expect("Failed to execute EGUI render pass");
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    pub fn swap_buffers(&mut self)
    {
        let texture = self.frame_texture.take();
        texture.unwrap().present();
    }

    ////////
    // Upload to GPU

    // WebGPU does not support texture arrays yet. So texture atlasing
    // is required to dynamically index into textures

    pub fn upload_texture(&mut self)->TextureHandle
    {
        return 0;
    }

    pub fn upload_model(&mut self)->BufferHandle
    {
        use wgpu::*;

        return 0;
    }

    pub fn compile_shader(&mut self, source: &str)->ShaderHandle
    {
        use wgpu::*;
        let shader_desc = ShaderModuleDescriptor
        {
            label: None,
            source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(source)),
        };
        let shader: ShaderModule = self.device.create_shader_module(shader_desc);

        self.shaders.push(shader);
        return (self.shaders.len() - 1) as u32;
    }

    pub fn create_program(&mut self, compute_shader: ShaderHandle)->ComputeProgramHandle
    {
        use wgpu::*;

        let bind_group_layout_entry = BindGroupLayoutEntry
        {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer
            {
                ty: BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None
            },
            count: None
        };

        // Signature of our compute main function
        let bind_group_layout_desc = BindGroupLayoutDescriptor
        {
            label: None,
            entries: &[bind_group_layout_entry]
        };
        let bind_group_layout = self.device.create_bind_group_layout(&bind_group_layout_desc);

        let pipeline_layout_desc = wgpu::PipelineLayoutDescriptor
        {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        };
        let pipeline_layout = self.device.create_pipeline_layout(&pipeline_layout_desc);

        let program_desc = ComputePipelineDescriptor
        {
            label: None,
            layout: Some(&pipeline_layout),
            module: &self.shaders[compute_shader as usize],
            entry_point: "main"
        };
        let program: ComputePipeline = self.device.create_compute_pipeline(&program_desc);

        self.compute_programs.push(program);
        return (self.compute_programs.len() - 1) as u32;
    }

    // Do some compute stuff
    pub fn test(compute_program: ComputeProgramHandle)
    {

    }
}
