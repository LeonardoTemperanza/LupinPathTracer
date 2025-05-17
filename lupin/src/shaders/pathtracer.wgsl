
// Group 0: Scene Description
// Mesh
@group(0) @binding(0)  var<storage, read> verts_pos_array: binding_array<VertsPos>;
@group(0) @binding(1)  var<storage, read> verts_array: binding_array<Verts>;
@group(0) @binding(2)  var<storage, read> indices_array: binding_array<Indices>;
@group(0) @binding(3)  var<storage, read> bvh_nodes_array: binding_array<BvhNodes>;
// Instances
@group(0) @binding(4)  var<storage, read> tlas_nodes: array<TlasNode>;
@group(0) @binding(5)  var<storage, read> instances: array<Instance>;
@group(0) @binding(6)  var<storage, read> materials: array<Material>;
// Textures
@group(0) @binding(7)  var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(8)  var samplers: binding_array<sampler>;
@group(0) @binding(9)  var env_map:  texture_2d<f32>;
@group(0) @binding(10) var env_map_sampler: sampler;

// Group 1: Pathtrace settings
@group(1) @binding(0) var<uniform> camera_transform: mat4x4f;
@group(1) @binding(1) var<uniform> accum_counter: u32;  // If this is 0, nothing is taken from the previous frame
@group(1) @binding(2) var prev_frame: texture_2d<f32>;

// Group 2: Render target
@group(2) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;

// We need these wrappers, or we get a
// compilation error, for some reason...
struct VertsPos { data: array<vec3f>   }
struct Verts    { data: array<Vertex>  }
struct Indices  { data: array<u32>     }
struct BvhNodes { data: array<BvhNode> }

//////////////////////////////////////////////
// Scene description
//////////////////////////////////////////////

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
    inv_transform: mat4x4f,
    mesh_idx: u32,
    mat_idx: u32,
    // 8 bytes padding
}

// Wgsl doesn't have enums...
const MAT_TYPE_MATTE: u32       = 0;
const MAT_TYPE_GLOSSY: u32      = 1;
const MAT_TYPE_REFLECTIVE: u32  = 2;
const MAT_TYPE_TRANSPARENT: u32 = 3;
const MAT_TYPE_REFRACTIVE: u32  = 4;
const MAT_TYPE_SUBSURFACE: u32  = 5;
const MAT_TYPE_VOLUMETRIC: u32  = 6;
const MAT_TYPE_GLTFPBR: u32     = 7;

struct Material
{
    color: vec4f,
    emission: vec4f,
    scattering: vec4f,
    mat_type: u32,
    roughness: f32,
    metallic: f32,
    ior: f32,
    sc_anisotropy: f32,
    tr_depth: f32,

    color_tex_idx:      u32,
    emission_tex_idx:   u32,
    roughness_tex_idx:  u32,
    scattering_tex_idx: u32,
    normal_tex_idx:     u32,
    // 4 bytes padding
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
    tri_count: u32,

    // Second child is at index first_child + 1
}

struct TlasNode
{
    aabb_min: vec3f,
    left_right: u32,  // 2x16 bits. If it's 0, this node is a leaf
    aabb_max: vec3f,
    instance_idx: u32,
}

//////////////////////////////////////////////
// Entrypoints
//////////////////////////////////////////////

@compute
@workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id: vec3u)
{
    let output_dim = textureDimensions(output_texture).xy;
    init_rng(global_id.y * global_id.x + global_id.x, output_dim.y * output_dim.x + output_dim.x);

    const NUM_SAMPLES: u32 = 1;
    var color = vec3f(0.0f);
    for(var sample: u32 = 0; sample < NUM_SAMPLES; sample++)
    {
        let camera_ray = compute_camera_ray(global_id, output_dim);
        color += pathtrace(local_id, camera_ray);
    }
    color /= f32(NUM_SAMPLES);
    color = max(color, vec3f(0.0f));

    // Progressive rendering.
    if accum_counter != 0
    {
        let frag_coord = vec2f(global_id.xy) + 0.5f;
        let resolution = vec2f(output_dim);
        let uv = frag_coord / resolution;

        let weight = 1.0f / f32(accum_counter);
        let prev_color = textureSampleLevel(prev_frame, samplers[0], uv, 0.0f).rgb;
        color = prev_color * (1.0f - weight) + color * weight;
        color = max(color, vec3f(0.0f));
    }

    if global_id.x < output_dim.x && global_id.y < output_dim.y {
        textureStore(output_texture, global_id.xy, vec4f(color, 1.0f));
    }
}

@compute
@workgroup_size(8, 8, 1)
fn main_(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id: vec3u)
{
    let output_dim = textureDimensions(output_texture).xy;
    init_rng(global_id.y * global_id.x + global_id.x, output_dim.y * output_dim.x + output_dim.x);

    var color = vec3f(0.0f);
    let camera_ray = compute_camera_ray(global_id, output_dim);
    let hit = ray_scene_intersection(local_id, camera_ray);
    if hit.dst != F32_MAX
    {
        color = hit.normal;
    }

    if global_id.x < output_dim.x && global_id.y < output_dim.y {
        textureStore(output_texture, global_id.xy, vec4f(color, 1.0f));
    }
}

// Applies pixel jittering.
fn compute_camera_ray(global_id: vec3u, output_dim: vec2u) -> Ray
{
    let frag_coord = vec2f(global_id.xy) + 0.5f;
    let resolution = vec2f(output_dim);

    let rnd_vec = vec2f(random_f32(), random_f32());
    var nudged_uv = frag_coord + (rnd_vec - 0.5f);  // Move 0.5 to the left and right
    nudged_uv = clamp(nudged_uv, vec2(0.0f), resolution.xy) / resolution;

    let uv = frag_coord / resolution;
    var coord = 2.0f * nudged_uv - 1.0f;
    coord.y *= -resolution.y / resolution.x;

    let look_at = normalize(vec3(coord, 1.0f));

    let res = Ray(vec3f(0.0f, 0.0f, 0.0f), look_at, 1.0f / look_at);
    return transform_ray(res, camera_transform);
}

//////////////////////////////////////////////
// Pathtracing
//////////////////////////////////////////////

const MAX_BOUNCES: u32 = 5;

fn pathtrace(local_id: vec3u, start_ray: Ray) -> vec3f
{
    var ray = start_ray;
    var weight = vec3f(1.0f);  // Multiplicative terms.
    var radiance = vec3f(0.0f);
    var op_bounce: u32 = 0;
    for(var bounce = 0u; bounce <= MAX_BOUNCES; bounce++)
    {
        let hit = ray_scene_intersection(local_id, ray);
        if hit.dst == F32_MAX  // Missed.
        {
            radiance += sample_env_map(ray.dir) * weight;
            break;
        }

        // Ray hit something.

        let mat = materials[hit.mat_idx];

        // TODO: Handle volume transmission.

        const in_volume = false;
        if !in_volume
        {
            const mat_sampler_idx: u32 = 0;  // TODO!

            // Eval material.
            let color = textureSampleLevel(textures[mat.color_tex_idx], samplers[mat_sampler_idx], hit.tex_coords, 0.0f) * mat.color;

            // Set next ray's origin at point of contact.
            ray.ori = ray.ori + ray.dir * hit.dst;

            // TODO: No caustics param.

            // Handle coverage.
            if color.a < 1.0f && random_f32() >= color.a
            {
                op_bounce++;
                if op_bounce > 128 { break; }
                bounce -= 1;
                continue;
            }

            //let emission = textureSampleLevel(textures[mat.emission_tex_idx], samplers[mat_sampler_idx], hit.tex_coords, 0.0f).rgb * mat.emission.rgb;
            let emission = mat.emission.rgb;
            //let roughness = textureSampleLevel(textures[mat.roughness_tex_idx], samplers[mat_sampler_idx], hit.tex_coords, 0.0f).r * mat.roughness;
            let roughness = mat.roughness;

            // Accumulate emission.
            radiance += weight * emission;

            let outgoing = -ray.dir;
            let sample = sample_bsdf(mat.mat_type, color.rgb, hit.normal, roughness, mat.ior, outgoing);
            weight *= sample.weight / sample.prob;

            // Update volume stack. TODO

            ray.dir = sample.incoming;
            ray.inv_dir = 1.0f / ray.dir;
        }
        else  // Volumetric rendering. TODO
        {

        }

        // Check weight.
        if all(weight == vec3f(0.0f)) || !is_vec3f_finite(weight) { break; }

        // Russian roulette.
        if bounce > 3
        {
            let survive_prob = min(0.99f, max(weight.x, max(weight.y, weight.z)));
            if random_f32() >= survive_prob { break; }
            weight *= 1.0f / survive_prob;
        }
    }

    return radiance;
}

fn pathtrace_mis(local_id: vec3u, ray: Ray) -> vec3f
{
    return vec3f(0.0f);
}

fn sample_env_map(dir: vec3f) -> vec3f
{
    let coords = vec2f((atan2(dir.x, dir.z) + PI) / (2*PI), acos(dir.y / PI));
    return textureSampleLevel(env_map, env_map_sampler, coords, 0.0f).rgb;
}

struct BsdfSample
{
    incoming: vec3f,  // Keep in mind this also points outwards.
    weight: vec3f,
    prob: f32,
}

fn sample_bsdf(mat_type: u32, color: vec3f, normal: vec3f, roughness: f32, ior: f32, outgoing: vec3f) -> BsdfSample
{
    var res = BsdfSample();

    let up_normal = select(-normal, normal, dot(normal, outgoing) > 0.0f);

    switch(mat_type)
    {
        case MAT_TYPE_MATTE:
        {
            res.incoming = random_direction_cos(up_normal);
            if !same_hemisphere(up_normal, outgoing, res.incoming) { return res; }

            res.weight = color / PI * abs(dot(up_normal, res.incoming));
            let cosw = dot(up_normal, res.incoming);
            res.prob = select(cosw / PI, 0.0f, cosw <= 0.0f);
        }
        case MAT_TYPE_REFLECTIVE:
        {
            // Microfacet sampling is pretty expensive so we do it conditionally.
            let microfacet_normal = up_normal;
            res.prob = 1.0f;
            if roughness > 0.0f
            {

            }

            res.incoming = reflect(-outgoing, up_normal);
            if !same_hemisphere(up_normal, outgoing, res.incoming) { return res; }
            res.weight = fresnel_conductor(reflectivity_to_eta(color), vec3f(0.0f), up_normal, outgoing);
        }
        case MAT_TYPE_TRANSPARENT:
        {
            // Microfacet sampling is pretty expensive so we do it conditionally.
            let microfacet_normal = normal;
            let microfacet_prob = 1.0f;
            if roughness > 0.0f
            {

            }

            let fresnel = fresnel_dielectric(ior, up_normal, outgoing);
            if random_f32() < fresnel
            {
                res.incoming = reflect(-outgoing, up_normal);
                res.weight = vec3f(1.0f) * fresnel;
                res.prob = fresnel;
            }
            else
            {
                res.incoming = -outgoing;
                res.weight = color * (1.0f - fresnel);
                res.prob = 1.0f - fresnel;
            }
        }
        case MAT_TYPE_REFRACTIVE:
        {
            // Microfacet sampling is pretty expensive so we do it conditionally.

        }
        case MAT_TYPE_VOLUMETRIC:
        {

        }
        case default: {}
    }

    return res;
}

fn reflectivity_to_eta(reflectivity: vec3f) -> vec3f
{
    var r = clamp(reflectivity, vec3f(0.0f), vec3f(0.99f));
    return (1.0f + sqrt(r)) / (1.0f - sqrt(r));
}

// All Fresnel functions are from:
// https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
fn fresnel_schlick_vec3f(color: vec3f, normal: vec3f, out_dir: vec3f) -> vec3f
{
    if all(color == vec3f(0.0f)) { return vec3f(0.0f); }

    let cosine = dot(normal, out_dir);
    return color + (1.0 - color) * pow(clamp(1.0f - abs(cosine), 0.0f, 1.0f), 5.0f);
}

fn fresnel_schlick(eta: f32, normal: vec3f, out_dir: vec3f) -> f32
{
    if eta == 0.0f { return 0.0f; }

    let cosine = dot(normal, out_dir);
    return eta + (1.0 - eta) * pow(clamp(1.0f - abs(cosine), 0.0f, 1.0f), 5.0f);
}

fn fresnel_dielectric(eta: f32, normal: vec3f, outgoing: vec3f) -> f32
{
    let cosw = abs(dot(normal, outgoing));

    let sin2 = 1.0f - cosw * cosw;
    let eta2 = eta * eta;

    let cos2t = 1.0f - sin2 / eta2;
    if cos2t < 0.0f { return 1.0f; }  // tir

    let t0 = sqrt(cos2t);
    let t1 = eta * t0;
    let t2 = eta * cosw;

    let rs = (cosw - t1) / (cosw + t1);
    let rp = (t0 - t2) / (t0 + t2);

    return (rs * rs + rp * rp) / 2.0f;
}

fn fresnel_conductor(eta: vec3f, etak: vec3f, normal: vec3f, outgoing: vec3f) -> vec3f
{
    var cosw = dot(normal, outgoing);
    if cosw <= 0.0f { return vec3f(0.0f); }

    cosw      = clamp(cosw, -1.0f, 1.0f);
    let cos2  = cosw * cosw;
    let sin2  = clamp(1.0f - cos2, 0.0f, 1.0f);
    let eta2  = eta * eta;
    let etak2 = etak * etak;

    let t0       = eta2 - etak2 - sin2;
    let a2plusb2 = sqrt(t0 * t0 + 4 * eta2 * etak2);
    let t1       = a2plusb2 + cos2;
    let a        = sqrt((a2plusb2 + t0) / 2.0f);
    let t2       = 2.0f * a * cosw;
    let rs       = (t1 - t2) / (t1 + t2);

    let t3 = cos2 * a2plusb2 + sin2 * sin2;
    let t4 = t2 * sin2;
    let rp = rs * (t3 - t4) / (t3 + t4);

    return (rp + rs) / 2.0f;
}

fn microfacet_distribution(roughness: f32, normal: vec3f, halfway: vec3f, ggx: bool) -> f32
{
    // From:
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    let cosine = dot(normal, halfway);
    if cosine <= 0.0f { return 0.0f; }
    let roughness2 = roughness * roughness;
    let cosine2    = cosine * cosine;
    if (ggx) {
        return roughness2 / (PI * (cosine2 * roughness2 + 1 - cosine2) * (cosine2 * roughness2 + 1 - cosine2));
    } else {
        return exp((cosine2 - 1) / (roughness2 * cosine2)) / (PI * roughness2 * cosine2 * cosine2);
    }
}

fn microfacet_shadowing1(roughness: f32, normal: vec3f, halfway: vec3f, direction: vec3f, ggx: bool) -> f32
{
    // From:
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation

    let cosine  = dot(normal, direction);
    let cosineh = dot(halfway, direction);
    if cosine * cosineh <= 0.0f { return 0.0f; }

    let roughness2 = roughness * roughness;
    let cosine2    = cosine * cosine;
    if ggx
    {
        return 2.0f * abs(cosine) / (abs(cosine) + sqrt(cosine2 - roughness2 * cosine2 + roughness2));
    }
    else
    {
        let ci = abs(cosine) / (roughness * sqrt(1 - cosine2));
        if ci < 1.6f {
            return (3.535f * ci + 2.181f * ci * ci) / (1.0f + 2.276f * ci + 2.577f * ci * ci);
        } else {
            return 1.0f;
        }
    }
}

fn microfacet_shadowing(roughness: f32, normal: vec3f, halfway: vec3f, outgoing: vec3f, incoming: vec3f, ggx: bool) -> f32
{
    return microfacet_shadowing1(roughness, normal, halfway, outgoing, ggx) *
           microfacet_shadowing1(roughness, normal, halfway, incoming, ggx);
}

//////////////////////////////////////////////
// Random number generation
//////////////////////////////////////////////

var<private> RNG_STATE: u32 = 0u;

fn init_rng(global_id: u32, last_id: u32)
{
    RNG_STATE = global_id + (last_id + 1u) * accum_counter;
}

// PCG Random number generator.
// From: www.pcg-random.org and www.shadertoy.com/view/XlGcRh
fn random_u32() -> u32
{
    RNG_STATE = RNG_STATE * 747796405u + 2891336453u;
    var result = ((RNG_STATE >> ((RNG_STATE >> 28u) + 4u)) ^ RNG_STATE) * 277803737u;
    result = (result >> 22u) ^ result;
    return result;
}

// From 0 (inclusive) to 1 (inclusive)
fn random_f32() -> f32
{
    RNG_STATE = RNG_STATE * 747796405u + 2891336453u;
    var result: u32 = ((RNG_STATE >> ((RNG_STATE >> 28u) + 4u)) ^ RNG_STATE) * 277803737u;
    result = (result >> 22u) ^ result;
    return f32(result) / 4294967295.0f;
}

fn random_f32_normal_dist() -> f32
{
    let theta = 2.0f * PI * random_f32();
    let rho = sqrt(-2.0f * log(random_f32()));
    return rho * cos(theta);
}

fn random_in_circle() -> vec2f
{
    let angle: f32 = random_f32() * 2.0f * PI;
    var res: vec2f = vec2f(cos(angle), sin(angle));
    res *= sqrt(random_f32());
    return res;
}

fn random_direction() -> vec3f
{
    // This is the same as sampling a random
    // point along a unit sphere. Since the
    // multivariate standard normal distribution
    // is spherically symmetric, we can just sample
    // the normal distribution 3 times to get our
    // direction result.
    let x = random_f32_normal_dist();
    let y = random_f32_normal_dist();
    let z = random_f32_normal_dist();
    return normalize(vec3(x, y, z));
}

fn random_in_hemisphere(normal: vec3f) -> vec3f
{
    let dir = random_direction();
    return dir * sign(dot(normal, dir));
}

// TODO: Can we make this faster?
fn random_direction_cos(normal: vec3f) -> vec3f
{
    let r1 = random_f32();
    let r2 = random_f32();

    // Spherical coordinates
    let theta = acos(sqrt(1.0f - r1));
    let phi = 2.0f * PI * r2;

    // Convert to Cartesian coordinates
    let x = sin(theta) * cos(phi);
    let y = sin(theta) * sin(phi);
    let z = cos(theta);

    // Transform to world space
    let w = normal;
    let axis = select(vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), abs(w.x) > 0.1f);
    let u = normalize(cross(axis, w));
    let v = cross(w, u);
    return normalize(x * u + y * v + z * w);
}

//////////////////////////////////////////////
// Acceleration structures
//////////////////////////////////////////////

struct Ray
{
    ori: vec3f,
    dir: vec3f,
    inv_dir: vec3f  // Precomputed inverse of the ray direction, for performance.
}

const RAY_HIT_MIN_DIST: f32 = 0.0001;

// From: https://tavianator.com/2011/ray_box.html
// For misses, t = F32_MAX
fn ray_aabb_dst(ray: Ray, aabb_min: vec3f, aabb_max: vec3f)->f32
{
    let t_min: vec3f = (aabb_min - 0.001 - ray.ori) * ray.inv_dir;
    let t_max: vec3f = (aabb_max + 0.001 - ray.ori) * ray.inv_dir;
    let t1: vec3f = min(t_min, t_max);
    let t2: vec3f = max(t_min, t_max);
    let dst_far: f32  = min(min(t2.x, t2.y), t2.z);
    let dst_near: f32 = max(max(t1.x, t1.y), t1.z);

    let did_hit: bool = dst_far >= dst_near && dst_far > 0.0f;
    return select(F32_MAX, dst_near, did_hit);
}

// From: https://www.shadertoy.com/view/MlGcDz
// Triangle intersection. Returns { t, u, v }
// For misses, t = F32_MAX
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
    let det = dot(ray.dir, n);
    let d = 1.0 / det;
    let u = d * dot(-q, v2v0);
    let v = d * dot(q, v1v0);
    var t = d * dot(-n, rov0);

    if min(u, v) < 0.0 || (u+v) > 1.0 || t < RAY_HIT_MIN_DIST { t = F32_MAX; }
    return vec3f(t, u, v);
}

struct HitInfo
{
    normal: vec3f,
    dst: f32,
    tex_coords: vec2f,
    mat_idx: u32,
}

const invalid_hit: HitInfo = HitInfo(vec3f(), F32_MAX, vec2f(), 0);

const MAX_TLAS_DEPTH: u32 = 20;  // This supports 2^MAX_TLAS_DEPTH objects.
const TLAS_STACK_SIZE: u32 = (MAX_TLAS_DEPTH + 1) * 8 * 8;  // shared memory
var<workgroup> tlas_stack: array<u32, TLAS_STACK_SIZE>;     // shared memory

fn ray_scene_intersection(local_id: vec3u, ray: Ray)->HitInfo
{
    // Comment/Uncomment to test the performance of shared memory
    // vs local array (registers or global memory)
    // Shared memory is faster (on a GTX 1070).
    //let offset: u32 = 0u;                                                // local
    let offset = (local_id.y * 8 + local_id.x) * (MAX_TLAS_DEPTH + 1);  // shared memory

    //var tlas_stack: array<u32, 26>;  // local
    var stack_idx: u32 = 2;
    tlas_stack[0 + offset] = 0u;
    tlas_stack[1 + offset] = 0u;

    // t, u, v
    var min_hit = vec3f(F32_MAX, 0.0f, 0.0f);
    var tri_idx: u32 = 0;
    var mesh_idx: u32 = 0;
    var mat_idx: u32 = 0;
    while stack_idx > 1
    {
        stack_idx--;
        let node = tlas_nodes[tlas_stack[stack_idx + offset]];

        if node.left_right == 0u  // Leaf node
        {
            let instance = instances[node.instance_idx];
            let ray_trans = transform_ray(ray, instances[node.instance_idx].inv_transform);
            let result = ray_mesh_intersection(local_id, ray_trans, instance.mesh_idx);
            if result.hit.x < min_hit.x
            {
                min_hit  = result.hit;
                tri_idx  = result.tri_idx;
                mesh_idx = instance.mesh_idx;
                mat_idx  = instance.mat_idx;
            }
        }
        else  // Non-leaf node
        {
            let left_child  = node.left_right >> 16;
            let right_child = node.left_right & 0x0000FFFF;
            let left_child_node  = tlas_nodes[left_child];
            let right_child_node = tlas_nodes[right_child];

            let left_dst  = ray_aabb_dst(ray, left_child_node.aabb_min,  left_child_node.aabb_max);
            let right_dst = ray_aabb_dst(ray, right_child_node.aabb_min, right_child_node.aabb_max);

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
                    tlas_stack[stack_idx + offset] = right_child;
                    stack_idx++;
                }

                if push_left
                {
                    tlas_stack[stack_idx + offset] = left_child;
                    stack_idx++;
                }
            }
            else
            {
                if push_left
                {
                    tlas_stack[stack_idx + offset] = left_child;
                    stack_idx++;
                }

                if push_right
                {
                    tlas_stack[stack_idx + offset] = right_child;
                    stack_idx++;
                }
            }
        }
    }

    var hit_info: HitInfo = invalid_hit;
    if min_hit.x != F32_MAX
    {
        let vert0: Vertex = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
        let vert1: Vertex = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
        let vert2: Vertex = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];
        let u = min_hit.y;
        let v = min_hit.z;
        let w = 1.0 - u - v;

        hit_info.dst = min_hit.x;
        hit_info.normal = normalize(vert0.normal*w + vert1.normal*u + vert2.normal*v);
        hit_info.tex_coords = vert0.tex_coords*w + vert1.tex_coords*u + vert2.tex_coords*v;
        hit_info.mat_idx = mat_idx;
    }

    return hit_info;
}

const MAX_BVH_DEPTH: u32 = 25;
const BVH_STACK_SIZE: u32 = (MAX_BVH_DEPTH + 1) * 8 * 8;  // shared memory
var<workgroup> bvh_stack: array<u32, BVH_STACK_SIZE>;     // shared memory

struct RayMeshIntersectionResult
{
    hit: vec3f,  // t, u, v
    tri_idx: u32
}

fn ray_mesh_intersection(local_id: vec3u, ray: Ray, mesh_idx: u32) -> RayMeshIntersectionResult
{
    // Comment/Uncomment to test the performance of shared memory
    // vs local array (registers or global memory)
    // Shared memory is faster (on a GTX 1070).
    //let offset: u32 = 0u;                                                // local
    let offset = (local_id.y * 8 + local_id.x) * (MAX_BVH_DEPTH + 1);  // shared memory

    //var bvh_stack: array<u32, 26>;  // local
    var stack_idx: u32 = 2;
    bvh_stack[0 + offset] = 0u;
    bvh_stack[1 + offset] = 0u;

    // t, u, v
    var min_hit = vec3f(F32_MAX, 0.0f, 0.0f);
    var tri_idx: u32 = 0;
    while stack_idx > 1
    {
        stack_idx--;
        let node = bvh_nodes_array[mesh_idx].data[bvh_stack[stack_idx + offset]];

        if node.tri_count > 0u  // Leaf node
        {
            let tri_begin = node.tri_begin_or_first_child;
            let tri_count = node.tri_count;
            for(var i: u32 = tri_begin; i < tri_begin + tri_count; i++)
            {
                let v0: vec3f = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[i*3 + 0]];
                let v1: vec3f = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[i*3 + 1]];
                let v2: vec3f = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[i*3 + 2]];
                let hit: vec3f = ray_tri_dst(ray, v0, v1, v2);
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
            let left_child_node  = bvh_nodes_array[mesh_idx].data[left_child];
            let right_child_node = bvh_nodes_array[mesh_idx].data[right_child];

            let left_dst  = ray_aabb_dst(ray, left_child_node.aabb_min,  left_child_node.aabb_max);
            let right_dst = ray_aabb_dst(ray, right_child_node.aabb_min, right_child_node.aabb_max);

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
                    bvh_stack[stack_idx + offset] = right_child;
                    stack_idx++;
                }

                if push_left
                {
                    bvh_stack[stack_idx + offset] = left_child;
                    stack_idx++;
                }
            }
            else
            {
                if push_left
                {
                    bvh_stack[stack_idx + offset] = left_child;
                    stack_idx++;
                }

                if push_right
                {
                    bvh_stack[stack_idx + offset] = right_child;
                    stack_idx++;
                }
            }
        }
    }

    return RayMeshIntersectionResult(min_hit, tri_idx);
}

//////////////////////////////////////////////
// Utils and constants
//////////////////////////////////////////////

const F32_MAX: f32 = 0x1.fffffep+127;  // WGSL does not yet have a "max(f32)"
const PI: f32 = 3.14159265358979323846264338327950288;

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

fn transform_ray(ray: Ray, transform: mat4x4f) -> Ray
{
    var res = ray;
    res.ori = transform_point(res.ori, transform);
    res.dir = transform_dir(res.dir, transform);
    res.inv_dir = 1.0f / res.dir;
    return res;
}

fn is_f32_finite(v: f32) -> bool
{
    return v == v && abs(v) <= F32_MAX;
}

fn is_vec3f_finite(v: vec3f) -> bool
{
    return is_f32_finite(v.x) && is_f32_finite(v.y) && is_f32_finite(v.z);
}

fn same_hemisphere(normal: vec3f, outgoing: vec3f, incoming: vec3f) -> bool
{
    return dot(normal, outgoing) * dot(normal, incoming) >= 0;
}
