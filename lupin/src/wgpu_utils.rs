
use crate::base::*;

pub fn init_wgpu_context(device_desc: wgpu::DeviceDescriptor, window: &Window)->wgpu::Device
{
    let instance_desc = InstanceDescriptor
    {
        // Backend considerations on windows:
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

    let device_desc = DeviceDescriptor
    {
        label: None,
        required_features: Features::TIMESTAMP_QUERY |  // For profiling
                           Features::TEXTURE_BINDING_ARRAY |  // For arrays of textures (pathtracer)
                           Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,  // (pathtracer)
        required_limits: Limits::default(),
    };
    let maybe_device_queue = wait_for(adapter.request_device(&device_desc, None));
    let (device, queue): (Device, Queue) = maybe_device_queue.expect("Failed to get device");
}

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
