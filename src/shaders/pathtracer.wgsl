
// There should be a wgsl custom preprocessor which implements:
// #assert(expression), should write to a debug texture (only in debug)

// NOTE: Early returns are heavily discouraged here because it
// will lead to thread divergence, and that in turn will cause
// any proceeding __syncthreads() call to invoke UB, as some threads
// will never reach that point because they early returned. In general
// execution between multiple threads should be as predictable as possible

// Scene representation
//@group(0) @binding(0) var models: storage

@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> verts_pos: array<vec3f>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> bvh_nodes: array<BvhNode>;

//@group(1) @binding(0) var atlas_1_channel: texture_2d<r8unorm, read>;
// Like base color
//@group(1) @binding(2) var atlas_3_channels: texture_2d<rgb8unorm, read>;
// Like environment maps
//@group(1) @binding(4) var atlas_hdr_3_channels: texture_2d<rgbf32, read>;

// Should add a per_frame here that i can use to send camera transform

const f32_max: f32 = 0x1.fffffep+127;

// This doesn't include positions, as that
// is stored in a separate buffer for locality
struct Vertex
{
    normal: vec3f,
    tex_coords: vec3f
}

struct PerFrame
{
    camera_transform: mat4x3f
}

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

    // We also need some texture wrapping info, i guess.
    // Stuff like: repeat, clamp, etc.
}

struct Ray
{
    ori: vec3f,
    dir: vec3f,
    inv_dir: vec3f  // Precomputed inverse of the ray direction, for performance
}

// NOTE: The odd ordering of the fields
// ensures that the struct is 32 bytes wide,
// given that vec3f has 16-byte padding
struct BvhNode
{
    aabb_min:  vec3f,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    tri_begin_or_first_child: u32,
    aabb_max:  vec3f,
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
// For misses, t = f32_max
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

// From: https://www.shadertoy.com/view/MlGcDz
// Triangle intersection. Returns { t, u, v }
// For misses, t = f32_max
fn ray_tri_dst(ray: Ray, v0: vec3f, v1: vec3f, v2: vec3f)->vec3f
{
    let v1v0 = v1 - v0;
    let v2v0 = v2 - v0;
    let rov0 = ray.ori - v0;

    // Cramer's rule for solving p(t) = ro+t·rd = p(u,v) = vo + u·(v1-v0) + v·(v2-v1).
    // The four determinants above have lots of terms in common. Knowing the changing
    // the order of the columns/rows doesn't change the volume/determinant, and that
    // the volume is dot(cross(a,b,c)), we can precompute some common terms and reduce
    // it all to:
    let n = cross(v1v0, v2v0);
    let q = cross(rov0, ray.dir);
    let d = 1.0 / dot(ray.dir, n);
    let u = d * dot(-q, v2v0);
    let v = d * dot(q, v1v0);
    var t = d * dot(-n, rov0);

    if min(u, v) < 0.0 || (u+v) > 1.0 { t = f32_max; }
    return vec3f(t, u, v);
}

struct HitInfo
{
    dst: f32,
    normal: vec3f,
    tex_coords: vec3f
}
fn ray_scene_intersection(ray: Ray)->HitInfo
{
    var stack: array<u32, 40>;
    var stack_idx: i32 = 1;
    stack[0] = 0u;

    var min_dst = f32_max;
    var tri_idx: u32 = 0;
    while stack_idx > 0
    {
        stack_idx--;
        let node = bvh_nodes[stack[stack_idx]];

        if node.tri_count > 0u  // Leaf node
        {
            let tri_begin = node.tri_begin_or_first_child;
            let tri_count = node.tri_count;
            for(var i: u32 = tri_begin; i < tri_begin + tri_count; i++)
            {
                let v0: vec3f = verts_pos[indices[i*3 + 0]];
                let v1: vec3f = verts_pos[indices[i*3 + 1]];
                let v2: vec3f = verts_pos[indices[i*3 + 2]];
                let dst: f32 = ray_tri_dst(ray, v0, v1, v2).x;
                if dst < min_dst
                {
                    min_dst = dst;
                    tri_idx = i;
                }
            }
        }
        else  // Non-leaf node
        {
            let left_child  = node.tri_begin_or_first_child;
            let right_child = left_child + 1;
            let left_child_node  = bvh_nodes[left_child];
            let right_child_node = bvh_nodes[right_child];

            let left_dst  = ray_aabb_dst(ray, left_child_node.aabb_min,  left_child_node.aabb_max);
            let right_dst = ray_aabb_dst(ray, right_child_node.aabb_min, right_child_node.aabb_max);

            // Push children onto the stack
            // The closest child should be looked at
            // first. This order is chosen so that it's more
            // likely that the second child will never need
            // to be queried
            if left_dst <= right_dst
            {
                if right_dst < min_dst
                {
                    stack[stack_idx] = right_child;
                    stack_idx++;
                }
                
                if left_dst < min_dst
                {
                    stack[stack_idx] = left_child;
                    stack_idx++;
                }
            }
            else
            {
                if left_dst < min_dst
                {
                    stack[stack_idx] = left_child;
                    stack_idx++;
                }

                if right_dst < min_dst
                {
                    stack[stack_idx] = right_child;
                    stack_idx++;
                }
            }
        }
    }

    var hit_info: HitInfo = HitInfo(min_dst, vec3f(0.0f), vec3f(0.0f));
    if min_dst != f32_max
    {
        let v0: vec3f = verts_pos[indices[tri_idx*3 + 0]];
        let v1: vec3f = verts_pos[indices[tri_idx*3 + 1]];
        let v2: vec3f = verts_pos[indices[tri_idx*3 + 2]];
        hit_info.normal = normalize(cross(v1 - v0, v2 - v0));
    }

    return hit_info;
}

@compute
//#compute_workgroup_1d
//#compute_workgroup_2d
//#compute_workgroup_3d
@workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>)
{
    var frag_coord = vec2f(global_id.xy) + 0.5f;
    var output_dim = textureDimensions(output_texture).xy;
    var resolution = vec2f(output_dim.xy);

    var uv = frag_coord / resolution;
    var coord = 2.0f * uv - 1.0f;
    coord.y *= -resolution.y / resolution.x;
    
    var camera_look_at = normalize(vec3(coord, 1.0f));

    var camera_ray = Ray(vec3f(-0.3f, 1.0f, -2.0f), camera_look_at, 1.0f / camera_look_at);

    var hit = ray_scene_intersection(camera_ray);

    var color = vec4f(select(vec3f(0.0f), hit.normal, hit.dst != f32_max), 1.0f);

    if global_id.x < output_dim.x && global_id.y < output_dim.y
    {
        var coord_color = vec4f(uv, 1.0f, 1.0f);
        textureStore(output_texture, global_id.xy, color);
    }
}
