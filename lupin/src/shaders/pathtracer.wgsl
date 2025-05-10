
// Group 0: Scene Description
// Mesh
@group(0) @binding(0) var<storage, read> verts_pos_array: binding_array<VertsPos>;
@group(0) @binding(1) var<storage, read> verts_array: binding_array<Verts>;
@group(0) @binding(2) var<storage, read> indices_array: binding_array<Indices>;
@group(0) @binding(3) var<storage, read> bvh_nodes_array: binding_array<BvhNodes>;
// Instances
@group(0) @binding(4) var<storage, read> tlas_nodes: array<Vertex>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
// Textures
//@group(0) @binding(6) var textures: binding_array<texture_2d<f32>>;
//@group(0) @binding(7) var samplers: binding_array<sampler>;

// Group 1: Pathtrace settings
@group(1) @binding(0) var<uniform> camera_transform: mat4x4f;

// Group 2: Render target
@group(2) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;

// We need these wrappers for some reason...
struct VertsPos { data: array<vec3f> }
struct Verts    { data: array<Vertex> }
struct Indices  { data: array<u32> }
struct BvhNodes { data: array<BvhNode> }

// Constants
const f32_max: f32 = 0x1.fffffep+127;

// This doesn't include positions; those are
// stored in a separate buffer for locality.
struct Vertex
{
    normal: vec3f,
    // 4 bytes padding
    tex_coords: vec2f
    // 8 bytes padding
}

struct Instance
{
    pos: vec3f,
    mesh_idx: u32,
    texture_idx: u32,
    sampler_idx: u32,
    // 8 bytes padding
}

struct Ray
{
    ori: vec3f,
    dir: vec3f,
    inv_dir: vec3f  // Precomputed inverse of the ray direction, for performance.
}

fn transform_point(p: vec3f, transform: mat4x4f)->vec3f
{
    let p_vec4 = vec4f(p, 1.0f);
    let transformed = transform * p_vec4;
    return (transformed / transformed.w).xyz;
}

fn transform_dir(dir: vec3f, transform: mat4x4f)->vec3f
{
    let dir_vec4 = vec4f(dir, 0.0f);
    return (transform * dir_vec4).xyz;
}

// NOTE: The odd ordering of the fields
// ensures that the struct is 32 bytes wide,
// given that vec3f has 16-byte alignment.
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

// From: https://tavianator.com/2011/ray_box.html
// For misses, t = f32_max
fn ray_aabb_dst(ray: Ray, aabb_min: vec3f, aabb_max: vec3f)->f32
{
    let t_min: vec3f = (aabb_min - 0.001 - ray.ori) * ray.inv_dir;
    let t_max: vec3f = (aabb_max + 0.001 - ray.ori) * ray.inv_dir;
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
    tex_coords: vec2f
}

const MAX_BVH_DEPTH: u32 = 25;
const STACK_SIZE: u32 = (MAX_BVH_DEPTH + 1) * 8 * 8;  // shared memory
var<workgroup> stack: array<u32, STACK_SIZE>;         // shared memory

fn ray_scene_intersection(local_id: vec3u, ray: Ray, test_idx: u32, test_pos: vec3f)->HitInfo
{
    // Transform the ray with the inverse of the instance transform.
    let ray_trans = Ray(ray.ori - test_pos, ray.dir, ray.inv_dir);

    // Comment/Uncomment to test the performance of shared memory
    // vs local array (registers or global memory)
    // Shared memory is faster (on a GTX 1070).
    //let offset: u32 = 0u;                                                // local
    let offset = (local_id.y * 8 + local_id.x) * (MAX_BVH_DEPTH + 1);  // shared memory

    //var stack: array<u32, 26>;  // local
    var stack_idx: u32 = 2;
    stack[0 + offset] = 0u;
    stack[1 + offset] = 0u;

    var num_boxes_hit: i32 = 0;

    // t, u, v
    var min_hit = vec3f(f32_max, 0.0f, 0.0f);
    var tri_idx: u32 = 0;
    while stack_idx > 1
    {
        stack_idx--;
        let node = bvh_nodes_array[test_idx].data[stack[stack_idx + offset]];

        if node.tri_count > 0u  // Leaf node
        {
            let tri_begin = node.tri_begin_or_first_child;
            let tri_count = node.tri_count;
            for(var i: u32 = tri_begin; i < tri_begin + tri_count; i++)
            {
                let v0: vec3f = verts_pos_array[test_idx].data[indices_array[test_idx].data[i*3 + 0]];
                let v1: vec3f = verts_pos_array[test_idx].data[indices_array[test_idx].data[i*3 + 1]];
                let v2: vec3f = verts_pos_array[test_idx].data[indices_array[test_idx].data[i*3 + 2]];
                let hit: vec3f = ray_tri_dst(ray_trans, v0, v1, v2);
                if hit.x < min_hit.x
                {
                    min_hit = hit;
                    tri_idx = i;
                }
            }
        }
        else  // Non-leaf node
        {
            let left_child  = node.tri_begin_or_first_child;
            let right_child = left_child + 1;
            let left_child_node  = bvh_nodes_array[test_idx].data[left_child];
            let right_child_node = bvh_nodes_array[test_idx].data[right_child];

            let left_dst  = ray_aabb_dst(ray_trans, left_child_node.aabb_min,  left_child_node.aabb_max);
            let right_dst = ray_aabb_dst(ray_trans, right_child_node.aabb_min, right_child_node.aabb_max);

            // Push children onto the stack
            // The closest child should be looked at
            // first. This order is chosen so that it's more
            // likely that the second child will never need
            // to be visited in depth.

            let visit_left_first: bool = left_dst <= right_dst;
            let push_left:  bool = left_dst < min_hit.x;
            let push_right: bool = right_dst < min_hit.x;

            if visit_left_first
            {
                if push_right
                {
                    stack[stack_idx + offset] = right_child;
                    stack_idx++;
                }

                if push_left
                {
                    stack[stack_idx + offset] = left_child;
                    stack_idx++;
                }
            }
            else
            {
                if push_left
                {
                    stack[stack_idx + offset] = left_child;
                    stack_idx++;
                }

                if push_right
                {
                    stack[stack_idx + offset] = right_child;
                    stack_idx++;
                }
            }
        }
    }

    var hit_info: HitInfo = HitInfo(min_hit.x, vec3f(0.0f), vec2f(0.0f));
    if hit_info.dst != f32_max
    {
        let vert0: Vertex = verts_array[test_idx].data[indices_array[test_idx].data[tri_idx*3 + 0]];
        let vert1: Vertex = verts_array[test_idx].data[indices_array[test_idx].data[tri_idx*3 + 1]];
        let vert2: Vertex = verts_array[test_idx].data[indices_array[test_idx].data[tri_idx*3 + 2]];
        let u = min_hit.y;
        let v = min_hit.z;
        let w = 1.0 - u - v;

        hit_info.normal = vert0.normal*w + vert1.normal*u + vert2.normal*v;
        hit_info.tex_coords = vert0.tex_coords*w + vert1.tex_coords*u + vert2.tex_coords*v;
    }

    return hit_info;
}

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id: vec3<u32>)
{
    var frag_coord = vec2f(global_id.xy) + 0.5f;
    var output_dim = textureDimensions(output_texture).xy;
    var resolution = vec2f(output_dim.xy);

    var uv = frag_coord / resolution;
    var coord = 2.0f * uv - 1.0f;
    coord.y *= -resolution.y / resolution.x;

    var camera_look_at = normalize(vec3(coord, 1.0f));

    var camera_ray = Ray(vec3f(0.0f, 0.0f, 0.0f), camera_look_at, 1.0f / camera_look_at);
    camera_ray.ori = transform_point(camera_ray.ori, camera_transform);
    camera_ray.dir = transform_dir(camera_ray.dir, camera_transform);
    camera_ray.inv_dir = 1.0f / camera_ray.dir;

    var closest_hit = HitInfo(f32_max, vec3f(0.0), vec2f(0.0));
    for (var i = 0; i < 4; i++)
    {
        var hit = ray_scene_intersection(local_id, camera_ray, instances[i].mesh_idx, instances[i].pos);
        if hit.dst < closest_hit.dst {
            closest_hit = hit;
        }
    }

    var color = vec4f(select(vec3f(0.0f), closest_hit.normal * 0.5f + 0.5f, closest_hit.dst != f32_max), 1.0f);
    //var color = vec4f(select(vec3f(0.0f), vec3f(closest_hit.tex_coords, 1.0f), closest_hit.dst != f32_max), 1.0f);
    color = max(color, vec4f(0.0));  // Clamp to 0

    if global_id.x < output_dim.x && global_id.y < output_dim.y {
        textureStore(output_texture, global_id.xy, color);
    }
}
