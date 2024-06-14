
// There should be a wgsl custom preprocessor which implements:
// #assert(expression), should write to a debug texture (only in debug)

// Scene representation
//@group(0) @binding(0) var models: storage

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> output_size: vec2u;

// Like alpha and roughness
//@group(1) @binding(0) var atlas_1_channel: texture_2d<r8unorm, read>;
// Like base color
//@group(1) @binding(2) var atlas_3_channels: texture_2d<rgb8unorm, read>;
// Like environment maps
//@group(1) @binding(4) var atlas_hdr_3_channels: texture_2d<rgbf32, read>;

const f32_max: f32 = 0x1.fffffep+127;

struct Material
{
    color_scale: vec3f,
    alpha_scale: f32,
    roughness_scale: f32,
    emission_scale: f32,

    // Textures (base coordinates for atlas lookup)
    // An out of bounds index will yield the neutral value
    // for the corresponding texture type
    color_texture: vec2u,      // 3 channels
    alpha_texture: vec2u,      // 1 channel
    roughness_texture: vec2u,  // 1 channel
    emission_texture: vec2u    // hdr 3 channels
}

struct Ray
{
    ori: vec3f,
    dir: vec3f,
    inv_dir: vec3f  // Precomputed inverse of the ray direction, for performance
}

struct BvhNode
{
    aabb_min:  vec3f,
    aabb_max:  vec3f,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    tri_begin_or_first_child: u32,
    tri_count: u32

    // Second child is at index first_child + 1
}

struct TlasNode
{
    aabb_min: vec3f,
    aabb_max: vec3f,
    // If is_leaf is true, this is mesh_instance,
    // otherwise this is first_child
    mesh_instance_or_first_child: u32,
    is_leaf:   u32,  // 32-bit bool

    // Second child is at index first_child + 1
}

struct MeshInstance
{
    // Transform from world space to model space
    inverse_transform: mat4x3f,
    material: Material,
    bvh_root: u32
}

struct Scene
{
    env_map: vec2u,
}

// From: https://tavianator.com/2011/ray_box.html
// Returns f32_max if there was no hit
fn ray_aabb_dst(ray: Ray, aabb_min: vec3f, aabb_max: vec3f)->f32
{
    let t_min: vec3f = (aabb_min - ray.ori) * ray.inv_dir;
    let t_max: vec3f = (aabb_max - ray.ori) * ray.inv_dir;
    let t1: vec3f = min(t_min, t_max);
    let t2: vec3f = max(t_min, t_max);
    let dst_far: f32  = min(min(t2.x, t2.y), t2.z);
    let dst_near: f32 = max(max(t1.x, t1.y), t1.z);

    let did_hit: bool = dst_far >= dst_near && dst_far > 0.0f;
    return select(f32_max, dst_near, did_hit);
}

@compute
//#compute_workgroup_1d
//#compute_workgroup_2d
//#compute_workgroup_3d
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    let coords = vec2u(global_id.xy);
    if coords.x < output_size.x && coords.y < output_size.y
    {
        textureStore(output_texture, coords, vec4f(f32(coords.x) / f32(output_size.x), f32(coords.y) / f32(output_size.y), 1.0f, 1.0f));
    }
}
