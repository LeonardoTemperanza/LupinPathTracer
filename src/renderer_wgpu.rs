
extern crate wgpu;
extern crate glfw;

use crate::base::*;

pub struct Renderer
{
    // Missing lifetime specifier... hehehe... might want to learn the main
    // part of the language now.
    surface: wgpu::Surface,
    device: wgpu::Device,
    adapter: wgpu::Adapter,
    instance: wgpu::Instance,
}

type TextureHandle = u32;
type BufferHandle  = u32;
type ShaderHandle  = u32;

pub fn init(window: &glfw::PWindow)->Renderer
{
    let instance_desc = wgpu::InstanceDescriptor::default();
    let instance: wgpu::Instance = wgpu::Instance::new(instance_desc);

    let surface: wgpu::Surface = instance.create_surface(&window).expect("Failed to create WGPU surface");

    let adapter_options = wgpu::RequestAdapterOptions
    {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: Some(&surface),
        force_fallback_adapter: false,
    };
    let adapter: wgpu::Adapter = wait_for(instance.request_adapter(&adapter_options)).expect("Failed to get adapter");

    let device_desc = wgpu::DeviceDescriptor
    {
        label: None,
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
    };
    let (device, queue): (wgpu::Device, wgpu::Queue) = wait_for(adapter.request_device(&device_desc, None)).expect("Failed to get device");

    let surface_config = surface.get_default_config(&adapter, 2000, 1000).expect("Failed to get surface default configuration");
    surface.configure(&device, &surface_config);
    
    return Renderer {};
}

pub fn cleanup(renderer: &mut Renderer)
{

}

pub fn upload_texture(r: &mut Renderer)->TextureHandle
{
    return 0;
}

pub fn upload_model()->BufferHandle
{
    return 0;
}

pub fn draw_scene()
{
    // Compile raytracer compute shader

}
