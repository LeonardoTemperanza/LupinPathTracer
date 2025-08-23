
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
    let instance = wgpu::Instance::new(&instance_desc);

    let surface = instance.create_surface(window).expect("Failed to create WGPU surface");

    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    };
    let adapter = wait_for(instance.request_adapter(&adapter_options)).expect("Failed to get adapter");

    let (device, queue) = wait_for(adapter.request_device(&device_desc, None)).expect("Failed to get device");

    surface.configure(&device, &surface_config);

    return (device, queue, surface, adapter);
}

pub fn init_default_wgpu_context_no_window(device_desc: wgpu::DeviceDescriptor)->(wgpu::Device, wgpu::Queue, wgpu::Adapter)
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
    let instance = wgpu::Instance::new(&instance_desc);

    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    };
    let adapter = wait_for(instance.request_adapter(&adapter_options)).expect("Failed to get adapter");

    let (device, queue) = wait_for(adapter.request_device(&device_desc, None)).expect("Failed to get device");

    return (device, queue, adapter);
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

pub fn upload_storage_buffer_with_name(device: &wgpu::Device, queue: &wgpu::Queue, buf: &[u8], label: &str)->wgpu::Buffer
{
    let wgpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
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
        size: 0,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    return wgpu_buffer
}

pub fn create_storage_buffer_with_size(device: &wgpu::Device, size: usize) -> wgpu::Buffer
{
    let wgpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    return wgpu_buffer
}

pub fn create_storage_buffer_with_size_and_name(device: &wgpu::Device, size: usize, name: &str) -> wgpu::Buffer
{
    let wgpu_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(name),
        size: size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    return wgpu_buffer
}

pub fn create_white_texture(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::Texture
{
    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[]
    });

    let white_texel: [u8; 4] = [ 255, 255, 255, 255 ];
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        &white_texel,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * size.width),
            rows_per_image: Some(size.height)
        },
        size
    );

    return texture;
}

pub fn create_linear_sampler(device: &wgpu::Device) -> wgpu::Sampler
{
    return device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
}
