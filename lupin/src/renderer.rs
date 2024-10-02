
use crate::base::*;

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
#[derive(Default, Clone, Copy)]
pub struct BvhNode
{
    pub aabb_min: Vec3,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    pub tri_begin_or_first_child: u32,
    pub aabb_max: Vec3,
    pub tri_count: u32
}

// Constants
pub const BVH_MAX_DEPTH: i32 = 25;

// This will need to be used when creating the device
fn get_device_spec()->wgpu::DeviceDescriptor<'static>
{
    // We currently need these two features:
    // 1) Arrays of texture bindings, to store textures of arbitrary sizes (do we actually want to do this?)
    // 2) Texture sampling and buffer non uniform indexing, to access textures (do we actually want to do this?)
    return wgpu::DeviceDescriptor
    {
        label: None,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY |
                           wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        required_limits: wgpu::Limits::default(),
    };
}

// Rendering
fn draw_scene(scene: &Scene, render_to: &wgpu::Texture, camera_transform: Mat4) {}

// Here we want:
// 1 stuff for drawing the scene. This should allow separating this draw call into multiple draw calls,
// so that the gpu can continue doing other work. Also, if the rendering takes really long, it's good
// user interface to let the user cancel the operation. Also, if the rendering takes too long i think vulkan
// will kill the process. NOTE: multi-queues are not implemented in wgpu as of today,
// but when they are, scene rendering can be moved to a separate queue and then we can synchronize as necessary.
// 2 stuff for executing shaders in general. Use the fancy debugging system.
// 3 constructing the bvh should be here, as it is an integral part of the renderer
