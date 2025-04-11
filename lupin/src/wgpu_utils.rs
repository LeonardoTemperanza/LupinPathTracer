
use crate::base::*;

// Windowing libraries tipically implement the Into<wgpu::SurfaceTarget> trait,
// if not you can easily implement it yourself
pub fn init_default_wgpu_context<'a>(device_desc: wgpu::DeviceDescriptor,
                             window: impl Into<wgpu::SurfaceTarget<'a>>,
                             window_width: i32, window_height: i32)->(wgpu::Device, wgpu::Queue, wgpu::Surface<'a>, wgpu::Adapter)
{
    let instance_desc = wgpu::InstanceDescriptor {
        #[cfg(target_os = "windows")]
        backends: wgpu::Backends::VULKAN,

        #[cfg(not(target_os = "windows"))]
        #[cfg(not(target_arch = "wasm32"))]
        backends: wgpu::Backends::PRIMARY,

        #[cfg(target_arch = "wasm32")]
        backends: wgpu::Backends::GL,
        ..Default::default()
    };
    let instance: wgpu::Instance = wgpu::Instance::new(instance_desc);

    let maybe_surface = instance.create_surface(window);
    let mut surface: wgpu::Surface = maybe_surface.expect("Failed to create WGPU surface");

    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    };
    let maybe_adapter = wait_for(instance.request_adapter(&adapter_options));
    let adapter: wgpu::Adapter = maybe_adapter.expect("Failed to get adapter");

    let maybe_device_queue = wait_for(adapter.request_device(&device_desc, None));
    let (device, queue) = maybe_device_queue.expect("Failed to get device");

    let swapchain_capabilities = surface.get_capabilities(&adapter);
    let swapchain_format = swapchain_capabilities.formats[0];

    let default_present_mode = wgpu::PresentMode::Immediate;
    configure_surface(&mut surface, &device, swapchain_format, default_present_mode, window_width, window_height);

    return (device, queue, surface, adapter);
}

pub fn upload_storage_buffer(device: &wgpu::Device, queue: &wgpu::Queue, buf: &[u8])->wgpu::Buffer
{
    let wgpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: buf.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    queue.write_buffer(&wgpu_buffer, 0, unsafe { to_u8_slice(&buf) });
    return wgpu_buffer
}

pub fn create_empty_storage_buffer(device: &wgpu::Device, _queue: &wgpu::Queue)->wgpu::Buffer
{
    let wgpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 0,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    return wgpu_buffer
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

// set_vsync function

// Take a look at this and check which one should even be implemented


/*
fn upload_buffer(&mut self, buffer: &[u8])->wgpu::Buffer;
fn upload_uniform(&mut self, buffer: &[u8])->Buffer;
fn create_empty_buffer(&mut self)->Buffer;
// Lets the user read a buffer from the GPU to the CPU. This will
// cause latency so it should be used very sparingly if at all
fn read_buffer(&mut self, buffer: Buffer, output: &mut[u8]);
fn read_texture(&mut self, texture: Texture, output: &mut[u8]);

// Textures
fn create_texture(&mut self, width: u32, height: u32)->Texture;
fn create_egui_output_texture(&mut self, width: u32, height: u32)->Texture;
fn get_texture_size(texture: &Texture)->(i32, i32);
fn resize_texture(&mut self, texture: &mut Texture, width: i32, height: i32);
fn texture_to_egui_texture(&mut self, texture: &Texture, filter_near: bool)->egui::TextureId;
fn update_egui_texture(&mut self, texture: &Texture, texture_id: egui::TextureId, filter_near: bool);

// GPU Timer
fn create_gpu_timer(&mut self, num_timestamps: u32)->GPUTimer;
fn add_timestamp(&mut self, timer: &mut GPUTimer);
// Returns an array of values, each of which represents the time
// spent between two timestamps added, in milliseconds. This will
// make the CPU wait for all the calls to be finished on the GPU side,
// so it should be used sparingly, perhaps at the end of a benchmark
// or for profiling
fn get_gpu_times(&mut self, timer: &GPUTimer, times: &mut [f32]);

// Miscellaneous
fn set_vsync(&mut self, flag: bool);  // Off by default
fn log_backend(&self);  // Logs the currently used renderer. In the case of WGPU, logs the used backend
*/
