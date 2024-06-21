
use crate::base::*;

use winit::window::Window;

use egui::{ClippedPrimitive, TexturesDelta};

//#[cfg()]
mod renderer_wgpu;

// This defines the high level structure for the
// various renderers (which may use different graphics APIs)
// You may even implement one with CUDA.

// Contains the GPU resource
// handles for the scene info
pub struct Scene
{
    pub verts_pos: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub bvh_nodes: wgpu::Buffer,
    pub verts: wgpu::Buffer

    /*
    // Texture atlases
    atlas_1_channel: u32,
    atlas_3_channels: u32,
    atlas_hdr_3_channels: u32,*/
}

// This doesn't include positions, as that
// is stored in a separate buffer for locality
#[repr(C)]
#[derive(Default)]
pub struct Vertex
{
    pub normal: Vec3,
    pub padding0: f32,
    pub tex_coords: Vec2,
    pub padding1: f32,
    pub padding2: f32,
}

// NOTE: The odd ordering of the fields
// ensures that the struct is 32 bytes wide,
// given that vec3f has 16-byte padding (on the GPU)
#[repr(C)]
#[derive(Default)]
pub struct BvhNode
{
    pub aabb_min: Vec3,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    pub tri_begin_or_first_child: u32,
    pub aabb_max: Vec3,
    pub tri_count: u32
}

pub trait RendererImpl<'a>
{
    // Initialization
    fn new(window: &'a Window, init_width: i32, init_height: i32)->Self;
    fn resize(&mut self, width: i32, height: i32);

    // Utils
    fn create_texture(&mut self, width: u32, height: u32)->Texture;
    fn resize_texture(&mut self, texture: &mut Texture, width: i32, height: i32);
    fn texture_to_egui_texture(&mut self, texture: &Texture, filter_near: bool)->egui::TextureId;
    fn update_egui_texture(&mut self, texture: &Texture, texture_id: egui::TextureId, filter_near: bool);

    // Rendering
    fn draw_scene(&mut self, scene: &Scene, render_to: &Texture);
    fn draw_egui(&mut self,
                 tris: Vec<ClippedPrimitive>,
                 textures_delta: &TexturesDelta,
                 width: i32, height: i32, scale: f32);
    fn begin_frame(&mut self);
    fn end_frame(&mut self);

    // CPU <-> GPU transfers
    fn upload_buffer(&mut self, buffer: &[u8])->Buffer;
    fn create_empty_buffer(&mut self)->Buffer;
    // Lets the user read a buffer from the GPU to the CPU. This will
    // cause latency so it should be used very sparingly if at all
    fn read_buffer(&mut self, buffer: Buffer, output: &mut[u8]);

    // Miscellaneous
    fn draw_test_triangle(&self);
    fn gpu_timer_begin(&self);
    // Returns GPU time spent between the begin and end calls,
    // in milliseconds. This will make the CPU wait for
    // all the calls to be finished on the GPU side, so it can only
    // really be used for very simple profiling, and definitely not
    // in release builds
    fn gpu_timer_end(&mut self)->f32;
    fn log_backend(&self);  // Logs the currently used renderer. In the case of WGPU, logs the used backend
}

pub trait TextureImpl
{
    fn get_size(&self)->(u32, u32);
}


//#[cfg()]
pub type Texture  = renderer_wgpu::Texture;
pub type Renderer<'a> = renderer_wgpu::Renderer<'a>;
pub type Buffer   = wgpu::Buffer;