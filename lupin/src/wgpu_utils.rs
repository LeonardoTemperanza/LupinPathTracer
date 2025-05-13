
use crate::base::*;

// Windowing libraries tipically implement the Into<wgpu::SurfaceTarget> trait,
// if not you can easily implement it yourself
pub fn init_default_wgpu_context<'a>(device_desc: wgpu::DeviceDescriptor,
                             surface_config: &wgpu::SurfaceConfiguration,
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
    let instance: wgpu::Instance = wgpu::Instance::new(&instance_desc);

    let maybe_surface = instance.create_surface(window);
    let surface: wgpu::Surface = maybe_surface.expect("Failed to create WGPU surface");

    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    };
    let maybe_adapter = wait_for(instance.request_adapter(&adapter_options));
    let adapter: wgpu::Adapter = maybe_adapter.expect("Failed to get adapter");

    let maybe_device_queue = wait_for(adapter.request_device(&device_desc, None));
    let (device, queue) = maybe_device_queue.expect("Failed to get device");

    surface.configure(&device, &surface_config);

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

    queue.write_buffer(&wgpu_buffer, 0, to_u8_slice(&buf));
    return wgpu_buffer
}

pub fn create_empty_storage_buffer(device: &wgpu::Device)->wgpu::Buffer
{
    let wgpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 1,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    return wgpu_buffer
}
