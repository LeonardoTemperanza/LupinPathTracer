
// Group 0: Scene Description
// Mesh
@group(0) @binding(0) var<storage, read> verts_pos_array: binding_array<VertsPos>;
@group(0) @binding(1) var<storage, read> verts_array: binding_array<Verts>;
@group(0) @binding(2) var<storage, read> indices_array: binding_array<Indices>;
@group(0) @binding(3) var<storage, read> bvh_nodes_array: binding_array<BvhNodes>;
// Instances
@group(0) @binding(4) var<storage, read> tlas_nodes: array<TlasNode>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> materials: array<Material>;
// Textures
@group(0) @binding(7) var textures: binding_array<texture_2d<f32>>;
@group(0) @binding(8) var samplers: binding_array<sampler>;
// Environments
@group(0) @binding(9) var<storage, read> environments: array<Environment>;
// Lights
@group(0) @binding(10) var<storage, read> lights: array<Light>;
@group(0) @binding(11) var<storage, read> alias_tables: binding_array<AliasTable>;
@group(0) @binding(12) var<storage, read> env_alias_tables: binding_array<AliasTable>;

// Group 1: Pathtrace settings
@group(1) @binding(0) var prev_frame: texture_2d<f32>;

// Group 2: Render targets
@group(2) @binding(0) var output_hdr: texture_storage_2d<rgba16float, write>;
@group(2) @binding(1) var output_rgba8unorm: texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(2) var output_rgba8snorm: texture_storage_2d<rgba8snorm, write>;

// Push constants
struct PushConstants
{
    camera_transform: mat4x4f,
    camera_lens: f32,
    camera_film: f32,
    camera_aspect: f32,
    camera_focus: f32,
    camera_aperture: f32,

    flags: u32,

    id_offset: vec2u,
    accum_counter: u32,  // If this is 0, nothing is taken from the previous frame.

    // Debug params
    heatmap_min: f32,
    heatmap_max: f32,
}

var<push_constant> constants: PushConstants;

// Override constants
override MAX_BOUNCES: u32 = 5;
override SAMPLES_PER_PIXEL: u32 = 1;
override DEBUG: bool = false;  // Relying on constant propagation for good performance.

// Constants
const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;
const SENTINEL_IDX: u32 = U32_MAX;

// We need these wrappers, or we get a weird
// compilation error, for some reason...
struct VertsPos   { data: array<vec3f>    }
struct Verts      { data: array<Vertex>   }
struct Indices    { data: array<u32>      }
struct BvhNodes   { data: array<BvhNode>  }
struct AliasTable { data: array<AliasBin> }

//////////////////////////////////////////////
// Scene description
//////////////////////////////////////////////

// This doesn't include positions; those are
// stored in a separate buffer for cache-locality.
struct Vertex
{
    color: vec4f,
    normal: vec3f,
    // 4 bytes padding
    tex_coords: vec2f
    // 8 bytes padding
}

// This lets us fetch the optional vertex attributes.
struct MeshInfo
{
    color_buf_idx: u32,
    tex_coords_buf_idx: u32,
    normal_buf_idx: u32,
}

struct Instance
{
    // NOTE: We want the inverse because it's used much more frequently,
    // and we want its transpose to take advantage of the reduced size
    // of the layout of a mat3x4f vs mat4x3f.
    transpose_inverse_transform: mat3x4f,
    mesh_idx: u32,
    mat_idx: u32,
    // 8 bytes padding
}

// WGSL doesn't have enums...
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

struct Environment
{
    emission: vec3f,
    emission_tex_idx: u32,
    transform: mat4x4f,
}

struct Light
{
    instance_idx: u32,
    area: f32,
}

struct AliasBin
{
    prob: f32,
    alias_threshold: f32,
    alias_idx: u32,
}

// NOTE: The odd ordering of the fields
// ensures that the struct is 32 bytes wide,
// as vec3f has 16-byte alignment.
struct BvhNode
{
    aabb_min: vec3f,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    tri_begin_or_first_child: u32,
    aabb_max: vec3f,
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

// NOTE: Coupled to constants in renderer.rs
const FLAG_CAMERA_ORTHO: u32            = 1 << 0;
const FLAG_ENVS_EMPTY: u32              = 1 << 1;
const FLAG_LIGHTS_EMPTY: u32            = 1 << 2;
// Debug flags
const FLAG_DEBUG_TRI_CHECKS:  u32       = 1 << 3;
const FLAG_DEBUG_AABB_CHECKS: u32       = 1 << 4;
const FLAG_DEBUG_NUM_BOUNCES: u32       = 1 << 5;
const FLAG_DEBUG_FIRST_HIT_ONLY: u32    = 1 << 6;

//////////////////////////////////////////////
// Entrypoints
//////////////////////////////////////////////

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn pathtrace_main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id_: vec3u)
{
    let global_id = global_id_.xy + constants.id_offset;
    let output_dim = textureDimensions(output_hdr).xy;
    init_rng(global_id.y * output_dim.x + global_id.x);

    var color = vec3f(0.0f);
    for(var sample: u32 = 0; sample < SAMPLES_PER_PIXEL; sample++)
    {
        let pixel_offset = random_vec2f() - 0.5f;
        let camera_ray = compute_camera_ray(global_id, output_dim, pixel_offset);
        color += pathtrace(local_id, camera_ray);
    }
    color /= f32(SAMPLES_PER_PIXEL);
    color = max(color, vec3f(0.0f));

    // Progressive rendering.
    if constants.accum_counter != 0
    {
        let weight = 1.0f / f32(constants.accum_counter);
        let prev_color = textureLoad(prev_frame, global_id.xy, 0).rgb;
        color = prev_color * (1.0f - weight) + color * weight;
        color = max(color, vec3f(0.0f));
    }

    if all(global_id < output_dim) {
        textureStore(output_hdr, global_id.xy, vec4f(color, 1.0f));
    }

    write_debug_color(global_id, output_dim);
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn pathtrace_debug_main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id_: vec3u)
{
    if !DEBUG { return; }

    let global_id = global_id_.xy + constants.id_offset;
    let output_dim = textureDimensions(output_rgba8unorm).xy;
    init_rng(global_id.y * output_dim.x + global_id.x);

    let pixel_offset = random_vec2f() - 0.5f;
    let camera_ray = compute_camera_ray(global_id, output_dim, pixel_offset);

    let first_hit_only = (constants.flags & FLAG_DEBUG_FIRST_HIT_ONLY) != 0;
    let debug_num_bounces = (constants.flags & FLAG_DEBUG_NUM_BOUNCES) != 0;

    if first_hit_only && !debug_num_bounces {
        ray_scene_intersection(local_id, camera_ray);
    } else {
        pathtrace(local_id, camera_ray);
    }

    var val = 0.0f;
    if (constants.flags & FLAG_DEBUG_TRI_CHECKS) != 0 {
        val = f32(RAY_DEBUG_INFO.num_tri_checks);
    } else if (constants.flags & FLAG_DEBUG_AABB_CHECKS) != 0 {
        val = f32(RAY_DEBUG_INFO.num_aabb_checks);
    } else if debug_num_bounces {
        val = f32(DEBUG_NUM_BOUNCES);
    }

    let color = get_heatmap_color(val, constants.heatmap_min, constants.heatmap_max);

    if all(global_id < output_dim) {
        textureStore(output_rgba8unorm, global_id.xy, vec4f(color, 1.0f));
    }
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn gbuffer_albedo_main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id_: vec3u)
{
    let global_id = global_id_.xy + constants.id_offset;
    let output_dim = textureDimensions(output_rgba8unorm).xy;
    init_rng(global_id.y * output_dim.x + global_id.x);

    const NUM_AA_SAMPLES_PER_DIR: u32 = 1;  // Antialiasing samples.
    var color = vec3f(0.0f);
    for(var sample_y: u32 = 0; sample_y < NUM_AA_SAMPLES_PER_DIR; sample_y++)
    {
        for(var sample_x: u32 = 0; sample_x < NUM_AA_SAMPLES_PER_DIR; sample_x++)
        {
            let offset_x = f32(sample_x) / f32(NUM_AA_SAMPLES_PER_DIR);
            let offset_y = f32(sample_y) / f32(NUM_AA_SAMPLES_PER_DIR);
            let pixel_offset = vec2f(offset_x, offset_y) - 0.5f * 0.9f;
            let camera_ray = compute_camera_ray(global_id, output_dim, pixel_offset);

            let hit = ray_scene_intersection(local_id, camera_ray);
            if hit.dst != F32_MAX
            {
                let mat_point = get_material_point(materials[hit.mat_idx], hit.tex_coords, hit.color);
                color += mat_point.color.rgb;
                //color += vec3f(mat_point.opacity);
                //color += hit.normal * 0.5f + 0.5f;
                //color += vec3f(mat_point.roughness);
                //color += vec3f(mat_point.metallic);
            }
        }
    }
    color /= f32(NUM_AA_SAMPLES_PER_DIR * NUM_AA_SAMPLES_PER_DIR);
    color = clamp(color, vec3f(0.0f), vec3f(1.0f));

    if all(global_id < output_dim) {
        textureStore(output_rgba8unorm, global_id.xy, vec4f(color, 1.0f));
    }
}

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn gbuffer_normals_main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id_: vec3u)
{
    let global_id = global_id_.xy + constants.id_offset;
    let output_dim = textureDimensions(output_rgba8snorm).xy;
    init_rng(global_id.y * output_dim.x + global_id.x);

    const NUM_AA_SAMPLES_PER_DIR: u32 = 1;  // Antialiasing samples.
    var color = vec3f(0.0f);
    for(var sample_y: u32 = 0; sample_y < NUM_AA_SAMPLES_PER_DIR; sample_y++)
    {
        for(var sample_x: u32 = 0; sample_x < NUM_AA_SAMPLES_PER_DIR; sample_x++)
        {
            let offset_x = f32(sample_x) / f32(NUM_AA_SAMPLES_PER_DIR);
            let offset_y = f32(sample_y) / f32(NUM_AA_SAMPLES_PER_DIR);
            let pixel_offset = vec2f(offset_x, offset_y) - 0.5f * 0.9f;
            let camera_ray = compute_camera_ray(global_id, output_dim, pixel_offset);

            let hit = ray_scene_intersection(local_id, camera_ray);
            if hit.dst != F32_MAX
            {
                let normal = get_shading_normal(hit);
                color += normal * 0.5f + 0.5f;
            }
        }
    }
    color /= f32(NUM_AA_SAMPLES_PER_DIR * NUM_AA_SAMPLES_PER_DIR);
    color = clamp(color, vec3f(-1.0f), vec3f(1.0f));

    if all(global_id < output_dim) {
        textureStore(output_rgba8snorm, global_id.xy, vec4f(color, 1.0f));
    }
}

// pixel_offset is expected to be in the [-0.5, 0.5] range.
fn compute_camera_ray(global_id: vec2u, output_dim: vec2u, pixel_offset: vec2f) -> Ray
{
    let resolution = vec2f(output_dim);
    let pixel_coord = vec2f(f32(global_id.x), resolution.y - f32(global_id.y)) + vec2f(0.5f);
    let nudged_uv = (pixel_coord + pixel_offset) / resolution;

    let camera_lens = constants.camera_lens;
    let camera_film = constants.camera_film;
    let camera_aspect = constants.camera_aspect;
    let camera_focus = constants.camera_focus;
    let camera_aperture = constants.camera_aperture;

    let film_size = select(vec2f(camera_film * camera_aspect, camera_film), vec2f(camera_film, camera_film / camera_aspect), camera_aspect >= 1);
    let lens_uv = random_in_disk();

    if (constants.flags & FLAG_CAMERA_ORTHO) != 0
    {
        let scale = 1.0 / camera_lens;
        let q = vec3f(film_size.x * (0.5f - nudged_uv.x) * scale, film_size.y * (0.5f - nudged_uv.y) * scale, camera_lens);
        let e = vec3f(-q.x, -q.y, 0) + vec3f(lens_uv.x * camera_aperture / 2.0f,
                                             lens_uv.y * camera_aperture / 2.0f, 0);
        let p = vec3f(-q.x, -q.y, -camera_focus);
        let d = normalize(p - e) * vec3f(1.0f, 1.0f, -1.0f);
        let res = Ray(e, d, 1.0f / d);
        return transform_ray(res, constants.camera_transform);
    }
    else
    {
        let q = vec3f(film_size * vec2f(0.5f - nudged_uv.x, 0.5f - nudged_uv.y), camera_lens);
        let look_at = -normalize(q);
        let lens_point = vec3f(lens_uv * vec2f(camera_aperture / 2.0f), 0.0f);
        let focus_point = vec3f(look_at * camera_focus / abs(look_at.z));
        let final_dir = normalize(focus_point - lens_point) * vec3f(1.0f, 1.0f, -1.0f);

        let res = Ray(lens_point, final_dir, 1.0f / final_dir);
        return transform_ray(res, constants.camera_transform);
    }
}

//////////////////////////////////////////////
// Pathtracing
//////////////////////////////////////////////

// General algorithms and structure from:
// https://github.com/xelatihy/yocto-gl

var<private> DEBUG_NUM_BOUNCES: u32 = 0;

fn pathtrace_naive(local_id: vec3u, start_ray: Ray) -> vec3f
{
    var ray = start_ray;
    var weight = vec3f(1.0f);  // Multiplicative terms.
    var radiance = vec3f(0.0f);
    var opacity_bounce: u32 = 0;
    for(var bounce = 0u; bounce <= MAX_BOUNCES; bounce++)
    {
        let hit = ray_scene_intersection(local_id, ray);
        if hit.dst == F32_MAX  // Missed.
        {
            radiance += weight * sample_environments(ray.dir);
            break;
        }

        // Ray hit something.
        if DEBUG {
            DEBUG_NUM_BOUNCES++;
        }

        let hit_pos = ray.ori + ray.dir * hit.dst;
        let outgoing = -ray.dir;

        let mat_point = get_material_point(materials[hit.mat_idx], hit.tex_coords, hit.color);

        // TODO: No caustics param.

        // Handle coverage.
        if mat_point.opacity < 1.0f && random_f32() >= mat_point.opacity
        {
            opacity_bounce++;
            if opacity_bounce > 128 { break; }
            ray.ori = hit_pos;
            bounce--;
            continue;
        }

        let normal = get_shading_normal(hit);

        // Accumulate emission.
        radiance += weight * mat_point.emission /** f32(dot(normal, outgoing) >= 0.0f)*/;

        // Compute next direction.
        var incoming = vec3f();
        if !is_mat_delta(mat_point)
        {
            let rnd0 = random_f32();
            let rnd1 = random_vec2f();
            incoming = sample_bsdfcos(mat_point, normal, outgoing, rnd0, rnd1);
            if all(incoming == vec3f(0.0f)) { break; }

            weight *= eval_bsdfcos(mat_point, normal, outgoing, incoming) / sample_bsdfcos_pdf(mat_point, normal, outgoing, incoming);
        }
        else
        {
            incoming = sample_delta(mat_point, normal, outgoing, random_f32());
            if all(incoming == vec3f(0.0f)) { break; }

            weight *= eval_delta(mat_point, normal, outgoing, incoming) / sample_delta_pdf(mat_point, normal, outgoing, incoming);
        }

        ray.ori = hit_pos;
        ray.dir = incoming;
        ray.inv_dir = 1.0f / ray.dir;

        // Check weight.
        if all(weight == vec3f(0.0f)) || !vec3f_is_finite(weight) { break; }

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

fn pathtrace(local_id: vec3u, start_ray: Ray) -> vec3f
{
    var ray = start_ray;
    var weight = vec3f(1.0f);  // Multiplicative terms.
    var radiance = vec3f(0.0f);
    var opacity_bounce: u32 = 0;
    for(var bounce = 0u; bounce <= MAX_BOUNCES; bounce++)
    {
        let hit = ray_scene_intersection(local_id, ray);
        if hit.dst == F32_MAX  // Missed.
        {
            radiance += weight * sample_environments(ray.dir);
            break;
        }

        // Ray hit something.
        if DEBUG {
            DEBUG_NUM_BOUNCES++;
        }

        let hit_pos = ray.ori + ray.dir * hit.dst;
        let outgoing = -ray.dir;

        let mat_point = get_material_point(materials[hit.mat_idx], hit.tex_coords, hit.color);

        // Handle coverage.
        if mat_point.opacity < 1.0f && random_f32() >= mat_point.opacity
        {
            opacity_bounce++;
            if opacity_bounce > 128 { break; }
            ray.ori = hit_pos;
            bounce--;
            continue;
        }

        let normal = get_shading_normal(hit);

        // Accumulate emission.
        radiance += weight * mat_point.emission /** f32(dot(normal, outgoing) >= 0.0f)*/;

        // Compute next direction.
        var incoming = vec3f();
        if !is_mat_delta(mat_point)
        {
            const light_prob = 0.5f;
            const bsdf_prob = 1.0f - light_prob;
            if random_f32() < bsdf_prob
            {
                let rnd0 = random_f32();
                let rnd1 = random_vec2f();
                incoming = sample_bsdfcos(mat_point, normal, outgoing, rnd0, rnd1);
            }
            else
            {
                incoming = sample_lights(hit_pos, normal, outgoing);
            }

            if all(incoming == vec3f(0.0f)) { break; }

            let prob = bsdf_prob  * sample_bsdfcos_pdf(mat_point, normal, outgoing, incoming) +
                       light_prob * sample_lights_pdf(local_id, hit_pos, incoming);
            weight *= eval_bsdfcos(mat_point, normal, outgoing, incoming) / prob;
        }
        else
        {
            incoming = sample_delta(mat_point, normal, outgoing, random_f32());
            if all(incoming == vec3f(0.0f)) { break; }

            weight *= eval_delta(mat_point, normal, outgoing, incoming) /
                      sample_delta_pdf(mat_point, normal, outgoing, incoming);
        }

        ray.ori = hit_pos;
        ray.dir = incoming;
        ray.inv_dir = 1.0f / ray.dir;

        // Check weight.
        if all(weight == vec3f(0.0f)) || !vec3f_is_finite(weight) { break; }

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

struct MaterialPoint
{
    mat_type: u32,
    emission: vec3f,
    color: vec3f,
    opacity: f32,
    roughness: f32,
    metallic: f32,
    ior: f32,
    density: vec3f,
    scattering: vec3f,
    sc_anisotropy: f32,
    tr_depth: f32,
}

const MIN_ROUGHNESS: f32 = 0.03f * 0.03f;

fn get_material_point(mat: Material, uv: vec2f, hit_color: vec4f) -> MaterialPoint
{
    var res = MaterialPoint();
    res.mat_type = mat.mat_type;

    const mat_sampler_idx: u32 = 0;  // TODO!

    // Sample textures.
    var color_sample = vec4f(1.0f);
    if mat.color_tex_idx != SENTINEL_IDX {
        color_sample = textureSampleLevel(textures[mat.color_tex_idx], samplers[mat_sampler_idx], uv, 0.0f);
        color_sample = vec4f(vec3f_srgb_to_linear(color_sample.rgb), color_sample.a);
    }
    var emission_sample = vec3f(1.0f);
    if mat.emission_tex_idx != SENTINEL_IDX {
        emission_sample = textureSampleLevel(textures[mat.emission_tex_idx], samplers[mat_sampler_idx], uv, 0.0f).rgb;
    }
    var roughness_sample = 1.0f;
    var metallic_sample = 1.0f;
    if mat.roughness_tex_idx != SENTINEL_IDX {
        let tex_sample = textureSampleLevel(textures[mat.roughness_tex_idx], samplers[mat_sampler_idx], uv, 0.0f).rgb;
        let sample_linear = vec3f_srgb_to_linear(tex_sample);
        roughness_sample = sample_linear.g;
        metallic_sample = sample_linear.b;
    }
    var scattering_sample = vec3f(1.0f);
    if mat.scattering_tex_idx != SENTINEL_IDX {
        scattering_sample = textureSampleLevel(textures[mat.scattering_tex_idx], samplers[mat_sampler_idx], uv, 0.0f).rgb;
    }

    // Fill in material.
    res.color = color_sample.rgb * mat.color.rgb * hit_color.rgb;
    res.opacity = color_sample.a * mat.color.a * hit_color.a;
    res.emission = emission_sample * mat.emission.rgb;
    res.roughness = roughness_sample * mat.roughness;
    res.roughness *= res.roughness;
    res.density = vec3f(0.0f);
    if mat.mat_type == MAT_TYPE_REFRACTIVE ||
       mat.mat_type == MAT_TYPE_VOLUMETRIC ||
       mat.mat_type == MAT_TYPE_SUBSURFACE {
        res.density = -log(clamp(res.color.rgb, vec3f(0.0001f), vec3f(1.0f))) / mat.tr_depth;
    }
    res.ior = mat.ior;
    res.scattering = scattering_sample * mat.scattering.xyz;
    res.sc_anisotropy = mat.sc_anisotropy;
    res.tr_depth = mat.tr_depth;
    res.metallic = metallic_sample * mat.metallic;

    // Clean up values.
    if res.mat_type == MAT_TYPE_MATTE   ||
       res.mat_type == MAT_TYPE_GLTFPBR ||
       res.mat_type == MAT_TYPE_GLOSSY {
        res.roughness = clamp(res.roughness, MIN_ROUGHNESS, 1.0f);
    } else if res.mat_type == MAT_TYPE_VOLUMETRIC {
        res.roughness = 0.0f;
    } else {
        if res.roughness < MIN_ROUGHNESS { res.roughness = 0.0f; }
    }

    return res;
}

fn get_shading_normal(hit: HitInfo) -> vec3f
{
    let mat = materials[hit.mat_idx];
    let uv = hit.tex_coords;

    var res = hit.normal;

    // Sample normalmap.
    if mat.normal_tex_idx != SENTINEL_IDX
    {
        const mat_sampler_idx: u32 = 0;  // TODO!

        let normalmap_sample = textureSampleLevel(textures[mat.normal_tex_idx], samplers[mat_sampler_idx], uv, 0.0f).xyz;
        var normal_local = -1.0 + 2.0 * normalmap_sample;
        var frame = mat3x3f(hit.tangent, hit.bitangent, res);
        frame[0] = orthonormalize(frame[0], frame[2]);
        frame[1] = normalize(cross(frame[2], frame[0]));

        let should_flip_v = dot(frame[1], hit.bitangent) < 0.0f;
        if should_flip_v { normal_local *= -1.0f; }

        res = normalize(frame * normal_local);
    }

    return res;
}

fn sample_environments(dir: vec3f) -> vec3f
{
    if (constants.flags & FLAG_ENVS_EMPTY) != 0 {
        return vec3f(0.0f);
    }

    var emission = vec3f(0.0f);
    for(var i = 0u; i < arrayLength(&environments); i++) {
        emission += sample_environment(dir, i);
    }
    return emission;
}

fn sample_environment(dir: vec3f, env_idx: u32) -> vec3f
{
    let env = environments[env_idx];
    let uv = dir_to_env_uv(dir, env_idx);
    const sampler_idx = 0u;  // TODO

    var res = env.emission.rgb;
    if env.emission_tex_idx != SENTINEL_IDX {
        res *= textureSampleLevel(textures[env.emission_tex_idx], samplers[sampler_idx], uv, 0.0f).rgb;
    }
    return res;
}

fn is_mat_delta(mat: MaterialPoint) -> bool
{
    return (mat.mat_type == MAT_TYPE_REFLECTIVE  && mat.roughness == 0.0f) ||
           (mat.mat_type == MAT_TYPE_REFRACTIVE  && mat.roughness == 0.0f) ||
           (mat.mat_type == MAT_TYPE_TRANSPARENT && mat.roughness == 0.0f) ||
           (mat.mat_type == MAT_TYPE_VOLUMETRIC);
}

fn reflectivity_to_eta(reflectivity: vec3f) -> vec3f
{
    var r = clamp(reflectivity, vec3f(0.0f), vec3f(0.99f));
    return (1.0f + sqrt(r)) / (1.0f - sqrt(r));
}

fn eta_to_reflectivity(eta: vec3f) -> vec3f {
  return ((eta - 1.0f) * (eta - 1.0f)) / ((eta + 1.0f) * (eta + 1.0f));
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
    if ggx {
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

fn init_rng(global_id: u32)
{
    let seed = 0u;
    RNG_STATE = hash_u32(global_id * 19349663u ^ constants.accum_counter * 83492791u ^ seed * 73856093u);
}

fn hash_u32(seed: u32) -> u32
{
    var x = seed;
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
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

// From 0 (inclusive) to 1 (exclusive)
fn random_f32() -> f32
{
    RNG_STATE = RNG_STATE * 747796405u + 2891336453u;
    var result: u32 = ((RNG_STATE >> ((RNG_STATE >> 28u) + 4u)) ^ RNG_STATE) * 277803737u;
    result = (result >> 22u) ^ result;
    return f32(result) / 4294967295.0f;
}

// NOTE: max_exclusive must be > 0!
fn random_u32_range_unsafe(max_exclusive: u32) -> u32
{
    return min(u32(random_f32() * f32(max_exclusive)), u32(max_exclusive - 1));
}

fn random_vec2f() -> vec2f
{
    // Separate statements to enforce evaluation order.
    let rnd0 = random_f32();
    let rnd1 = random_f32();
    return vec2f(rnd0, rnd1);
}

fn random_f32_normal_dist() -> f32
{
    let theta = 2.0f * PI * random_f32();
    let rho = sqrt(-2.0f * log(random_f32()));
    return rho * cos(theta);
}

fn random_in_disk() -> vec2f
{
    let rnd = random_vec2f();
    let r   = sqrt(rnd.y);
    let phi = 2.0f * PI * rnd.x;
    return vec2f(cos(phi) * r, sin(phi) * r);
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
    // Separate statements to enforce evaluation order.
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

fn random_tri_uv() -> vec2f
{
    let rnd = random_vec2f();
    return vec2f(1.0f - sqrt(rnd.x), rnd.y * sqrt(rnd.x));
}

fn random_in_tri(v0: vec3f, v1: vec3f, v2: vec3f) -> vec3f
{
    let rnd = random_vec2f();
    let uv = vec2f(1.0f - sqrt(rnd.x), rnd.y * sqrt(rnd.x));
    return v0 * (1.0f - uv.x - uv.y) + v1 * uv.x + v2 * uv.y;
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
    tangent: vec3f,
    mat_idx: u32,
    bitangent: vec3f,
    color: vec4f,
    tex_coords: vec2f,
}

const invalid_hit: HitInfo = HitInfo(vec3f(), F32_MAX, vec3f(), 0, vec3f(), vec4f(), vec2f());

const MAX_TLAS_DEPTH: u32 = 20;  // This supports 2^MAX_TLAS_DEPTH objects.
const TLAS_STACK_SIZE: u32 = (MAX_TLAS_DEPTH + 1) * 8 * 8;  // shared memory
var<workgroup> tlas_stack: array<u32, TLAS_STACK_SIZE>;     // shared memory

struct RayDebugInfo
{
    num_tri_checks: u32,
    num_aabb_checks: u32,
}

var<private> RAY_DEBUG_INFO: RayDebugInfo = RayDebugInfo();

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
    var transform = mat3x4f();
    while stack_idx > 1
    {
        stack_idx--;
        let node = tlas_nodes[tlas_stack[stack_idx + offset]];

        if node.left_right == 0u  // Leaf node
        {
            let instance = instances[node.instance_idx];

            var ray_trans = ray;
            // NOTE: We do vector * matrix because it's transposed.
            ray_trans.ori = (vec4f(ray_trans.ori, 1.0f) * instance.transpose_inverse_transform).xyz;
            // NOTE: We do not normalize because we do want ray.dir's length to change.
            ray_trans.dir = (vec4f(ray_trans.dir, 0.0f) * instance.transpose_inverse_transform).xyz;
            ray_trans.inv_dir = 1.0f / ray_trans.dir;

            let result = ray_mesh_intersection(local_id, ray_trans, min_hit.x, instance.mesh_idx);
            if result.hit.x < min_hit.x
            {
                min_hit   = result.hit;
                tri_idx   = result.tri_idx;
                mesh_idx  = instance.mesh_idx;
                mat_idx   = instance.mat_idx;
                transform = instance.transpose_inverse_transform;
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
        let v0: Vertex = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
        let v1: Vertex = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
        let v2: Vertex = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];
        let u = min_hit.y;
        let v = min_hit.z;
        let w = 1.0 - u - v;

        let normal_local = normalize(v0.normal*w + v1.normal*u + v2.normal*v);
        let normal_mat = mat3x3f(transform[0].xyz, transform[1].xyz, transform[2].xyz);

        hit_info.dst = min_hit.x;
        hit_info.mat_idx = mat_idx;

        hit_info.normal = normalize(normal_mat * normal_local);
        hit_info.color = v0.color*w + v1.color*u + v2.color*v;
        hit_info.tex_coords = v0.tex_coords*w + v1.tex_coords*u + v2.tex_coords*v;

        // Tangent and bitangent from uv.
        {
            let p0 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
            let p1 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
            let p2 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];

            // From YoctoGL.
            // Follows the definition in http://www.terathon.com/code/tangent.html and
            // https://gist.github.com/aras-p/2843984
            // normal points up from texture space
            let p   = p1 - p0;
            let q   = p2 - p0;
            let s   = vec2f(v1.tex_coords.x - v0.tex_coords.x, v2.tex_coords.x - v0.tex_coords.x);
            let t   = vec2f(v1.tex_coords.y - v0.tex_coords.y, v2.tex_coords.y - v0.tex_coords.y);
            let div = s.x * t.y - s.y * t.x;

            var tangent_local   = vec3f(1.0f, 0.0f, 0.0f);
            var bitangent_local = vec3f(0.0f, 1.0f, 0.0f);
            if div != 0
            {
                tangent_local   = vec3f(t.y * p.x - t.x * q.x,
                                        t.y * p.y - t.x * q.y,
                                        t.y * p.z - t.x * q.z) / div;
                bitangent_local = vec3f(s.x * q.x - s.y * p.x,
                                        s.x * q.y - s.y * p.y,
                                        s.x * q.z - s.y * p.z) / div;
            }

            hit_info.tangent = normalize(normal_mat * tangent_local);
            hit_info.bitangent = normalize(normal_mat * bitangent_local);
        }
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

// cur_min_hit_dst should be F32_MAX if absent.
fn ray_mesh_intersection(local_id: vec3u, ray: Ray, cur_min_hit_dst: f32, mesh_idx: u32) -> RayMeshIntersectionResult
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
    var min_hit = vec3f(cur_min_hit_dst, 0.0f, 0.0f);
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

                if DEBUG {
                    RAY_DEBUG_INFO.num_tri_checks++;
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

            if DEBUG {
                RAY_DEBUG_INFO.num_aabb_checks += 2;
            }

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

fn ray_instance_intersection(local_id: vec3u, ray: Ray, cur_min_hit_dst: f32, instance_idx: u32) -> RayMeshIntersectionResult
{
    let instance = instances[instance_idx];

    let test = transpose(instance.transpose_inverse_transform);
    let trans_mat4 = mat4x4f(vec4f(test[0], 0.0f), vec4f(test[1], 0.0f), vec4f(test[2], 0.0f), vec4f(test[3], 1.0f));

    let ray_trans = transform_ray_without_normalizing_direction(ray, trans_mat4);
    let result = ray_mesh_intersection(local_id, ray_trans, cur_min_hit_dst, instance.mesh_idx);
    return result;
}

//////////////////////////////////////////////
// Sampling functions
//////////////////////////////////////////////

fn sample_bsdfcos(material: MaterialPoint, normal: vec3f, outgoing: vec3f, rnl: f32, rn: vec2f) -> vec3f
{
    if material.roughness == 0.0f { return vec3f(); }

    switch material.mat_type
    {
        case MAT_TYPE_MATTE:       { return sample_matte(material.color, normal, outgoing, rn); }
        case MAT_TYPE_GLOSSY:      { return sample_glossy(material.color, material.ior, material.roughness, normal, outgoing, rnl, rn); }
        case MAT_TYPE_REFLECTIVE:  { return sample_reflective(material.color, material.roughness, normal, outgoing, rn); }
        case MAT_TYPE_TRANSPARENT: { return sample_transparent(material.color, material.ior, material.roughness, normal, outgoing, rnl, rn); }
        case MAT_TYPE_REFRACTIVE:  { return sample_refractive(material.color, material.ior, material.roughness, normal, outgoing, rnl, rn); }
        case MAT_TYPE_SUBSURFACE:  { return sample_refractive(material.color, material.ior, material.roughness, normal, outgoing, rnl, rn); }
        case MAT_TYPE_GLTFPBR:     { return sample_gltfpbr(material.color, material.ior, material.roughness, material.metallic, normal, outgoing, rnl, rn); }
        default: { return vec3f(); }
    }

    return vec3f();
}

fn sample_matte(color: vec3f, normal: vec3f, outgoing: vec3f, rn: vec2f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    return sample_hemisphere_cos(up_normal, rn);
}

fn sample_glossy(color: vec3f, ior: f32, roughness: f32, normal: vec3f, outgoing: vec3f,
                 rnl: f32, rn: vec2f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    if rnl < fresnel_dielectric(ior, up_normal, outgoing)
    {
        let halfway  = sample_microfacet(roughness, up_normal, rn, true);
        let incoming = reflect_(outgoing, halfway);
        if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
        return incoming;
    }
    else
    {
        return sample_hemisphere_cos(up_normal, rn);
    }
}

fn sample_reflective(color: vec3f, roughness: f32, normal: vec3f, outgoing: vec3f,
                     rn: vec2f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let halfway   = sample_microfacet(roughness, up_normal, rn, true);
    let incoming  = reflect_(outgoing, halfway);
    if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
    return incoming;
}

fn sample_transparent(color: vec3f, ior: f32, roughness: f32, normal: vec3f, outgoing: vec3f,
                      rnl: f32, rn: vec2f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let halfway   = sample_microfacet(roughness, up_normal, rn, true);
    if rnl < fresnel_dielectric(ior, halfway, outgoing)
    {
        let incoming = reflect_(outgoing, halfway);
        if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
        return incoming;
    }
    else
    {
        let reflected = reflect_(outgoing, halfway);
        let incoming  = -reflect_(reflected, up_normal);
        if same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
        return incoming;
    }
}

fn sample_refractive(color: vec3f, ior: f32, roughness: f32, normal: vec3f, outgoing: vec3f,
                     rnl: f32, rn: vec2f) -> vec3f
{
    let entering  = dot(normal, outgoing) >= 0;
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let halfway   = sample_microfacet(roughness, up_normal, rn, true);
    // let halfway = sample_microfacet(roughness, up_normal, outgoing, rn, true);
    if rnl < fresnel_dielectric(select(1.0f / ior, ior, entering), halfway, outgoing)
    {
        let incoming = reflect_(outgoing, halfway);
        if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
        return incoming;
    }
    else
    {
        let incoming = refract_(outgoing, halfway, select(ior, 1.0f / ior, entering));
        if same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
        return incoming;
    }
}

fn sample_gltfpbr(color: vec3f, ior: f32, roughness: f32,
                  metallic: f32, normal: vec3f, outgoing: vec3f,
                  rnl: f32, rn: vec2f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let reflectivity = mix(eta_to_reflectivity(vec3f(ior)), color, metallic);
    let fresnel_schlick = fresnel_schlick_vec3f(reflectivity, up_normal, outgoing);
    if rnl < (fresnel_schlick.x + fresnel_schlick.y + fresnel_schlick.z) / 3.0f
    {
        let halfway  = sample_microfacet(roughness, up_normal, rn, true);
        let incoming = reflect_(outgoing, halfway);
        if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(); }
        return incoming;
    }
    else
    {
        return sample_hemisphere_cos(up_normal, rn);
    }
}

fn sample_microfacet(roughness: f32, normal: vec3f, rn: vec2f, ggx: bool) -> vec3f
{
    let phi   = 2.0f * PI * rn.x;
    var theta = 0.0f;
    if ggx
    {
        theta = atan(roughness * sqrt(rn.y / (1 - rn.y)));
    }
    else
    {
        let roughness2 = roughness * roughness;
        theta           = atan(sqrt(-roughness2 * log(1 - rn.y)));
    }

    let local_half_vector = vec3f(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
    return normalize(basis_fromz(normal) * local_half_vector);
}

fn eval_bsdfcos(material: MaterialPoint, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if material.roughness == 0.0f { return vec3f(); }

    switch material.mat_type
    {
        case MAT_TYPE_MATTE:       { return eval_matte(material.color, normal, outgoing, incoming); }
        case MAT_TYPE_GLOSSY:      { return eval_glossy(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_REFLECTIVE:  { return eval_reflective(material.color, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_TRANSPARENT: { return eval_transparent(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_REFRACTIVE:  { return eval_refractive(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_SUBSURFACE:  { return eval_refractive(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_GLTFPBR:     { return eval_gltfpbr(material.color, material.ior, material.roughness, material.metallic, normal, outgoing, incoming); }
        default: { return vec3f(); }
    }

    return vec3f(0.0f);
}

fn eval_matte(color: vec3f, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0 { return vec3f(); }
    return color / PI * abs(dot(normal, incoming));
}

fn eval_glossy(color: vec3f, ior: f32, roughness: f32, normal: vec3f,
               outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0 { return vec3f(); }
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0);
    let F1        = fresnel_dielectric(ior, up_normal, outgoing);
    let halfway   = normalize(incoming + outgoing);
    let F         = fresnel_dielectric(ior, halfway, incoming);
    let D         = microfacet_distribution(roughness, up_normal, halfway, true);
    let G         = microfacet_shadowing(
                roughness, up_normal, halfway, outgoing, incoming, true);
    return color * (1.0f - F1) / PI * abs(dot(up_normal, incoming)) +
           vec3f(1.0f) * F * D * G /
               (4.0f * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
               abs(dot(up_normal, incoming));
}

fn eval_reflective(color: vec3f, roughness: f32, normal: vec3f,
                   outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0 { return vec3f(); };
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0);
    let halfway   = normalize(incoming + outgoing);
    let F         = fresnel_conductor(
                reflectivity_to_eta(color), vec3f(), halfway, incoming);
    let D = microfacet_distribution(roughness, up_normal, halfway, true);
    let G = microfacet_shadowing(
        roughness, up_normal, halfway, outgoing, incoming, true);
    return F * D * G / (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
           abs(dot(up_normal, incoming));
}

fn eval_transparent(color: vec3f, ior: f32, roughness: f32, normal: vec3f,
                    outgoing: vec3f, incoming: vec3f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f
    {
        let halfway = normalize(incoming + outgoing);
        let F       = fresnel_dielectric(ior, halfway, outgoing);
        let D       = microfacet_distribution(roughness, up_normal, halfway, true);
        let G       = microfacet_shadowing(
                  roughness, up_normal, halfway, outgoing, incoming, true);
        return vec3f(1.0f) * F * D * G /
               (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
               abs(dot(up_normal, incoming));
    }
    else
    {
        let reflected = reflect_(-incoming, up_normal);
        let halfway   = normalize(reflected + outgoing);
        let F         = fresnel_dielectric(ior, halfway, outgoing);
        let D         = microfacet_distribution(roughness, up_normal, halfway, true);
        let G         = microfacet_shadowing(
                    roughness, up_normal, halfway, outgoing, reflected, true);
        return color * (1.0f - F) * D * G /
               (4.0f * dot(up_normal, outgoing) * dot(up_normal, reflected)) *
               (abs(dot(up_normal, reflected)));
    }
}

fn eval_refractive(color: vec3f, ior: f32, roughness: f32,
    normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    let entering  = dot(normal, outgoing) >= 0;
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0);
    let rel_ior   = select(1.0f / ior, ior, entering);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0
    {
        let halfway = normalize(incoming + outgoing);
        let F       = fresnel_dielectric(rel_ior, halfway, outgoing);
        let D       = microfacet_distribution(roughness, up_normal, halfway, true);
        let G       = microfacet_shadowing(
                  roughness, up_normal, halfway, outgoing, incoming, true);
        return vec3f(1.0f * F * D * G /
               abs(4.0f * dot(normal, outgoing) * dot(normal, incoming)) *
               abs(dot(normal, incoming)));
    }
    else
    {
        let halfway = -normalize(rel_ior * incoming + outgoing) *
                       (select(-1.0f, 1.0f, entering));
        let F = fresnel_dielectric(rel_ior, halfway, outgoing);
        let D = microfacet_distribution(roughness, up_normal, halfway, true);
        let G = microfacet_shadowing(
            roughness, up_normal, halfway, outgoing, incoming, true);
        // [Walter 2007] equation 21
        return vec3f(1.0f) *
               abs((dot(outgoing, halfway) * dot(incoming, halfway)) /
                   (dot(outgoing, normal) * dot(incoming, normal))) *
               (1 - F) * D * G /
               pow(rel_ior * dot(halfway, incoming) + dot(halfway, outgoing),
                   2.0f) *
               abs(dot(normal, incoming));
    }
}

fn eval_gltfpbr(color: vec3f, ior: f32, roughness: f32, metallic: f32, normal: vec3f,
                outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0.0f { return vec3f(); }
    let reflectivity = mix(
        eta_to_reflectivity(vec3f(ior)), color, metallic);
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0);
    let F1        = fresnel_schlick_vec3f(reflectivity, up_normal, outgoing);
    let halfway   = normalize(incoming + outgoing);
    let F         = fresnel_schlick_vec3f(reflectivity, halfway, incoming);
    let D         = microfacet_distribution(roughness, up_normal, halfway, true);
    let G         = microfacet_shadowing(
                roughness, up_normal, halfway, outgoing, incoming, true);
    return color * (1 - metallic) * (1 - F1) / PI *
               abs(dot(up_normal, incoming)) +
           F * D * G / (4 * dot(up_normal, outgoing) * dot(up_normal, incoming)) *
               abs(dot(up_normal, incoming));
}

fn sample_bsdfcos_pdf(material: MaterialPoint, normal: vec3f,
                      outgoing: vec3f, incoming: vec3f) -> f32
{
    if material.roughness == 0.0f { return 0.0f; }

    switch material.mat_type
    {
        case MAT_TYPE_MATTE:        { return sample_matte_pdf(material.color, normal, outgoing, incoming); }
        case MAT_TYPE_GLOSSY:       { return sample_glossy_pdf(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_REFLECTIVE:   { return sample_reflective_pdf(material.color, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_TRANSPARENT:  { return sample_transparent_pdf(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_REFRACTIVE:   { return sample_refractive_pdf(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_SUBSURFACE:   { return sample_refractive_pdf(material.color, material.ior, material.roughness, normal, outgoing, incoming); }
        case MAT_TYPE_GLTFPBR:      { return sample_gltfpbr_pdf(material.color, material.ior, material.roughness, material.metallic, normal, outgoing, incoming); }
        default: { return 0.0f; }
    }

    return 0.0f;
}

fn sample_matte_pdf(color: vec3f, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> f32
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0.0f { return 0.0f; }
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    return sample_hemisphere_cos_pdf(up_normal, incoming);
}

fn sample_glossy_pdf(color: vec3f, ior: f32, roughness: f32, normal: vec3f,
                     outgoing: vec3f, incoming: vec3f) -> f32
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0.0f { return 0.0f; }
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let halfway   = normalize(outgoing + incoming);
    let F         = fresnel_dielectric(ior, up_normal, outgoing);
    return F * sample_microfacet_pdf(roughness, up_normal, halfway, true) /
               (4 * abs(dot(outgoing, halfway))) +
           (1 - F) * sample_hemisphere_cos_pdf(up_normal, incoming);
}

fn sample_reflective_pdf(color: vec3f, roughness: f32, normal: vec3f,
                         outgoing: vec3f, incoming: vec3f) -> f32
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0 { return 0.0f; }
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let halfway   = normalize(outgoing + incoming);
    return sample_microfacet_pdf(roughness, up_normal, halfway, true) /
           (4 * abs(dot(outgoing, halfway)));
}

fn sample_transparent_pdf(color: vec3f, ior: f32, roughness: f32, normal: vec3f,
                          outgoing: vec3f, incoming: vec3f) -> f32
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0
    {
        let halfway = normalize(incoming + outgoing);
        return fresnel_dielectric(ior, halfway, outgoing) *
               sample_microfacet_pdf(roughness, up_normal, halfway, true) /
               (4 * abs(dot(outgoing, halfway)));
    }
    else
    {
        let reflected = reflect_(-incoming, up_normal);
        let halfway   = normalize(reflected + outgoing);
        let d         = (1 - fresnel_dielectric(ior, halfway, outgoing)) *
                 sample_microfacet_pdf(roughness, up_normal, halfway, true);
        return d / (4 * abs(dot(outgoing, halfway)));
    }
}

fn sample_refractive_pdf(color: vec3f, ior: f32, roughness: f32, normal: vec3f,
                         outgoing: vec3f, incoming: vec3f) -> f32
{
    let entering  = dot(normal, outgoing) >= 0;
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let rel_ior   = select(1.0f / ior, ior, entering);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f
    {
        let halfway = normalize(incoming + outgoing);
        return fresnel_dielectric(rel_ior, halfway, outgoing) *
               sample_microfacet_pdf(roughness, up_normal, halfway, true) /
               //  sample_microfacet_pdf(roughness, up_normal, halfway, outgoing) /
               (4 * abs(dot(outgoing, halfway)));
    }
    else
    {
        let halfway = -normalize(rel_ior * incoming + outgoing) *
                       (select(-1.0f, 1.0f, entering));
        // [Walter 2007] equation 17
        return (1 - fresnel_dielectric(rel_ior, halfway, outgoing)) *
               sample_microfacet_pdf(roughness, up_normal, halfway, true) *
               //  sample_microfacet_pdf(roughness, up_normal, halfway, outgoing, true) /
               abs(dot(halfway, incoming)) /  // here we use incoming as from pbrt
               pow(rel_ior * dot(halfway, incoming) + dot(halfway, outgoing), 2.0f);
    }
}

fn sample_gltfpbr_pdf(color: vec3f, ior: f32, roughness: f32, metallic: f32, normal: vec3f,
                      outgoing: vec3f, incoming: vec3f) -> f32
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0 { return 0.0f; }
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let halfway      = normalize(outgoing + incoming);
    let reflectivity = mix(
        eta_to_reflectivity(vec3f(ior)), color, metallic);
    let fresnel_schlick = fresnel_schlick_vec3f(reflectivity, up_normal, outgoing);
    let F = (fresnel_schlick.x + fresnel_schlick.y + fresnel_schlick.z) / 3.0f;
    return F * sample_microfacet_pdf(roughness, up_normal, halfway, true) /
               (4 * abs(dot(outgoing, halfway))) +
           (1 - F) * sample_hemisphere_cos_pdf(up_normal, incoming);
}

fn sample_microfacet_pdf(roughness: f32, normal: vec3f, halfway: vec3f, ggx: bool) -> f32
{
    let cosine = dot(normal, halfway);
    if cosine < 0 { return 0.0f; }
    return microfacet_distribution(roughness, normal, halfway, ggx) * cosine;
}

fn sample_hemisphere_cos(normal: vec3f, ruv: vec2f) -> vec3f
{
    let z               = sqrt(ruv.y);
    let r               = sqrt(1 - z * z);
    let phi             = 2 * PI * ruv.x;
    let local_direction = vec3f(r * cos(phi), r * sin(phi), z);
    return normalize(basis_fromz(normal) * local_direction);
}

fn sample_hemisphere_cos_pdf(normal: vec3f, direction: vec3f) -> f32
{
    let cosw = dot(normal, direction);
    return select(cosw / PI, 0.0f, cosw <= 0.0f);
}

fn sample_delta(material: MaterialPoint, normal: vec3f, outgoing: vec3f, rnl: f32) -> vec3f
{
    if material.roughness != 0.0f { return vec3f(); }

    switch material.mat_type
    {
        case MAT_TYPE_REFLECTIVE:  { return sample_reflective_delta(material.color, normal, outgoing); }
        case MAT_TYPE_TRANSPARENT: { return sample_transparent_delta(material.color, material.ior, normal, outgoing, rnl); }
        case MAT_TYPE_REFRACTIVE:  { return sample_refractive_delta(material.color, material.ior, normal, outgoing, rnl); }
        case MAT_TYPE_VOLUMETRIC:  { return sample_passthrough(material.color, normal, outgoing); }
        default: { return vec3f(); }
    }

    return vec3f();
}

fn sample_reflective_delta(color: vec3f, normal: vec3f, outgoing: vec3f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    return reflect_(outgoing, up_normal);
}

fn sample_transparent_delta(color: vec3f, ior: f32, normal: vec3f, outgoing: vec3f, rnl: f32) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    if rnl < fresnel_dielectric(ior, up_normal, outgoing) {
      return reflect_(outgoing, up_normal);
    } else {
      return -outgoing;
    }
}

fn sample_refractive_delta(color: vec3f, ior: f32, normal: vec3f, outgoing: vec3f, rnl: f32) -> vec3f
{
    if abs(ior - 1) < 1e-3 { return -outgoing; }
    let entering  = dot(normal, outgoing) >= 0;
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let rel_ior   = select(1.0f / ior, ior, entering);
    if rnl < fresnel_dielectric(rel_ior, up_normal, outgoing) {
        return reflect_(outgoing, up_normal);
    } else {
        return refract_(outgoing, up_normal, 1 / rel_ior);
    }
}

fn sample_passthrough(color: vec3f, normal: vec3f, outgoing: vec3f) -> vec3f
{
    return -outgoing;
}

fn eval_delta(material: MaterialPoint, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if material.roughness != 0.0f { return vec3f(); }

    switch material.mat_type
    {
        case MAT_TYPE_REFLECTIVE:  { return eval_reflective_delta(material.color, normal, outgoing, incoming); }
        case MAT_TYPE_TRANSPARENT: { return eval_transparent_delta(material.color, material.ior, normal, outgoing, incoming); }
        case MAT_TYPE_REFRACTIVE:  { return eval_refractive_delta(material.color, material.ior, normal, outgoing, incoming); }
        case MAT_TYPE_VOLUMETRIC:  { return eval_passthrough(material.color, normal, outgoing, incoming); }
        default: { return vec3f(); }
    }

    return vec3f();
}

fn eval_reflective_delta(color: vec3f, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0.0f { return vec3f(); }
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    return fresnel_conductor(reflectivity_to_eta(color), vec3f(), up_normal, outgoing);
}

fn eval_transparent_delta(color: vec3f, ior: f32, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0 {
        return vec3f(1.0f) * fresnel_dielectric(ior, up_normal, outgoing);
    } else {
        return color * (1 - fresnel_dielectric(ior, up_normal, outgoing));
    }
}

fn eval_refractive_delta(color: vec3f, ior: f32, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if abs(ior - 1.0f) < 1e-3f {
        return select(vec3f(0.0f), vec3f(1.0f), dot(normal, incoming) * dot(normal, outgoing) <= 0);
    }
    let entering  = dot(normal, outgoing) >= 0;
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let rel_ior   = select(1.0f / ior, ior, entering);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f {
        return vec3f(1.0f) * fresnel_dielectric(rel_ior, up_normal, outgoing);
    } else {
        return vec3f(1.0f) * (1 / (rel_ior * rel_ior)) *
               (1 - fresnel_dielectric(rel_ior, up_normal, outgoing));
    }
}

fn eval_passthrough(color: vec3f, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> vec3f
{
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f {
        return vec3f(0.0f);
    } else {
        return vec3f(1.0f);
    }
}

fn sample_delta_pdf(material: MaterialPoint, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> f32
{
    if material.roughness != 0.0f { return 0.0f; }

    switch material.mat_type
    {
        case MAT_TYPE_REFLECTIVE:  { return sample_reflective_delta_pdf(material.color, normal, outgoing, incoming); }
        case MAT_TYPE_TRANSPARENT: { return sample_transparent_delta_pdf(material.color, material.ior, normal, outgoing, incoming); }
        case MAT_TYPE_REFRACTIVE:  { return sample_refractive_delta_pdf(material.color, material.ior, normal, outgoing, incoming); }
        case MAT_TYPE_VOLUMETRIC:  { return sample_passthrough_pdf(material.color, normal, outgoing, incoming); }
        default: { return 0.0f; }
    }

    return 0.0f;
}

fn sample_reflective_delta_pdf(color: vec3f, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> f32
{
    if dot(normal, incoming) * dot(normal, outgoing) <= 0.0f { return 0.0f; }
    return 1.0f;
}

fn sample_transparent_delta_pdf(color: vec3f, ior: f32, normal: vec3f,
                                outgoing: vec3f, incoming: vec3f) -> f32
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f {
        return fresnel_dielectric(ior, up_normal, outgoing);
    } else {
        return 1.0f - fresnel_dielectric(ior, up_normal, outgoing);
    }
}

fn sample_refractive_delta_pdf(color: vec3f, ior: f32, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> f32
{
    if abs(ior - 1) < 1e-3f {
        return select(0.0f, 1.0f, dot(normal, incoming) * dot(normal, outgoing) < 0.0f);
    }
    let entering  = dot(normal, outgoing) >= 0;
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);
    let rel_ior   = select(1.0f / ior, ior, entering);
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f {
        return fresnel_dielectric(rel_ior, up_normal, outgoing);
    } else {
        return (1 - fresnel_dielectric(rel_ior, up_normal, outgoing));
    }
}

fn sample_passthrough_pdf(color: vec3f, normal: vec3f, outgoing: vec3f, incoming: vec3f) -> f32
{
    if dot(normal, incoming) * dot(normal, outgoing) >= 0.0f {
        return 0.0f;
    } else {
        return 1.0f;
    }
}

fn basis_fromz(v: vec3f) -> mat3x3f
{
    // https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    let z    = normalize(v);
    let sign = copysignf(1.0f, z.z);
    let a    = -1.0f / (sign + z.z);
    let b    = z.x * z.y * a;
    let x    = vec3f(1.0f + sign * z.x * z.x * a, sign * b, -sign * z.x);
    let y    = vec3f(b, sign + z.y * z.y * a, -z.y);
    return mat3x3f(x, y, z);  // TODO: Does this do what i think it does?
}

fn copysignf(mag: f32, sgn: f32) -> f32 { return select(mag, -mag, sgn < 0.0f); }

// TODO: Investigate what makes these two different from the built-ins of the same name.
fn reflect_(w: vec3f, n: vec3f) -> vec3f
{
    return -w + 2 * dot(n, w) * n;
}

fn refract_(w: vec3f, n: vec3f, inv_eta: f32) -> vec3f
{
    let cosine = dot(n, w);
    let k      = 1 + inv_eta * inv_eta * (cosine * cosine - 1);
    if k < 0.0f { return vec3f(); }  // tir
    return -w * inv_eta + (inv_eta * cosine - sqrt(k)) * n;
}

// LIGHT SAMPLING

fn sample_lights(pos: vec3f, normal: vec3f, outgoing: vec3f) -> vec3f
{
    let up_normal = select(normal, -normal, dot(normal, outgoing) <= 0.0f);

    let num_lights = select(arrayLength(&lights), 0u, (constants.flags & FLAG_LIGHTS_EMPTY)  != 0);
    let num_envs = select(arrayLength(&environments), 0u, (constants.flags & FLAG_ENVS_EMPTY) != 0);
    if num_lights + num_envs <= 0 { return vec3f(); }

    let light_idx = random_u32_range_unsafe(num_lights + num_envs);
    if light_idx < num_lights
    {
        let tri_idx = sample_instance_alias_table(light_idx);
        let instance_idx = lights[light_idx].instance_idx;
        let mesh_idx = instances[instance_idx].mesh_idx;
        let uv = random_tri_uv();

        let incoming = compute_dir_from_point_to_tri(instance_idx, tri_idx, uv, pos);

        if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(0.0f); }
        return incoming;
    }
    else
    {
        let env_idx = light_idx - num_lights;
        let env_tex_idx = environments[env_idx].emission_tex_idx;
        let env_tex_size = textureDimensions(textures[env_tex_idx]);

        let sample = sample_env_alias_table(env_idx);
        let coords = vec2u(sample % env_tex_size.x, sample / env_tex_size.x);
        let uv = (vec2f(coords) + 0.5f) / vec2f(env_tex_size);

        let incoming = env_idx_to_dir(sample, env_idx);

        if !same_hemisphere(up_normal, outgoing, incoming) { return vec3f(0.0f); }
        return incoming;
    }
}

fn sample_lights_pdf(local_id: vec3u, pos: vec3f, incoming: vec3f) -> f32
{
    var pdf = 0.0f;

    let num_lights = select(arrayLength(&lights), 0u, (constants.flags & FLAG_LIGHTS_EMPTY) != 0);
    let num_envs = select(arrayLength(&environments), 0u, (constants.flags & FLAG_ENVS_EMPTY) != 0);

    for(var i = 0u; i < num_lights; i++)
    {
        let light = lights[i];
        let instance_idx = light.instance_idx;

        var light_pdf = 0.0f;
        var next_pos = pos;
        for(var bounce = 0u; bounce < 100; bounce++)
        {
            var ray = Ray(next_pos, incoming, 1.0f / incoming);
            let hit_info = ray_instance_intersection(local_id, ray, F32_MAX, instance_idx);
            let hit_dst = hit_info.hit.x;
            let hit_uv  = hit_info.hit.yz;
            if hit_dst == F32_MAX { break; }  // No intersection.

            let light_normal = compute_tri_geom_normal(instance_idx, hit_info.tri_idx);

            let light_pos = ray.ori + ray.dir * hit_dst;

            let prob = alias_tables[i].data[hit_info.tri_idx].prob;
            let dist2 = dot(light_pos - pos, light_pos - pos);
            let cos_theta = abs(dot(light_normal, incoming));

            light_pdf += dist2 / (cos_theta * light.area);
            next_pos = light_pos + incoming;
        }

        pdf += light_pdf;
    }

    for(var i = 0u; i < num_envs; i++)
    {
        let env_tex_idx = environments[i].emission_tex_idx;
        let env_tex_size = textureDimensions(textures[env_tex_idx]);

        let pixel_coords = dir_to_env_coords(incoming, i);
        let pixel_idx = pixel_coords.y * env_tex_size.x + pixel_coords.x;
        let prob = env_alias_tables[i].data[pixel_idx].prob;

        let solid_angle = (2.0f * PI / f32(env_tex_size.x)) *
                          (PI / f32(env_tex_size.y)) *
                          sin(PI * (f32(pixel_coords.y) + 0.5f) / f32(env_tex_size.y));

        pdf += prob / solid_angle;
    }

    pdf /= f32(num_lights + num_envs);

    return pdf;
}

fn dir_to_env_coords(dir: vec3f, env: u32) -> vec2u
{
    let env_tex_idx = environments[env].emission_tex_idx;
    let env_tex_size = textureDimensions(textures[env_tex_idx]);

    let uv = dir_to_env_uv(dir, env);
    return vec2u(clamp(u32(uv.x * f32(env_tex_size.x)), 0u, env_tex_size.x - 1),
                 clamp(u32(uv.y * f32(env_tex_size.y)), 0u, env_tex_size.y - 1));
}

fn compute_dir_from_point_to_tri(instance_idx: u32, tri_idx: u32, uv: vec2f, p: vec3f) -> vec3f
{
    let instance = instances[instance_idx];
    let mesh_idx = instance.mesh_idx;

    let test = transpose(instance.transpose_inverse_transform);
    let inv_trans = mat4x4f(vec4f(test[0], 0.0f), vec4f(test[1], 0.0f), vec4f(test[2], 0.0f), vec4f(test[3], 1.0f));

    let local_p = (inv_trans * vec4f(p, 1.0f)).xyz;

    let v0: vec3f = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
    let v1: vec3f = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
    let v2: vec3f = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];
    let w = 1.0 - uv.x - uv.y;

    let local_tri_pos = v0*w + v1*uv.x + v2*uv.y;

    let world_tri_pos = (mat4x4f_inverse(inv_trans) * vec4f(local_tri_pos, 1.0f)).xyz;
    return normalize(world_tri_pos - p);

    //let local_dir = normalize(local_tri_pos - local_p);
    //let normal_mat = transpose(mat3x3f(inv_trans[0].xyz, inv_trans[1].xyz, inv_trans[2].xyz));
    //return normalize(normal_mat * local_dir);
}

fn compute_tri_normal(instance_idx: u32, tri_idx: u32, uv: vec2f) -> vec3f
{
    let instance = instances[instance_idx];
    let mesh_idx = instance.mesh_idx;

    let test = transpose(instance.transpose_inverse_transform);
    let inv_trans = mat4x4f(vec4f(test[0], 0.0f), vec4f(test[1], 0.0f), vec4f(test[2], 0.0f), vec4f(test[3], 1.0f));

    let v0 = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
    let v1 = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
    let v2 = verts_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];
    let w = 1.0 - uv.x - uv.y;

    let local_normal = normalize(v0.normal*w + v1.normal*uv.x + v2.normal*uv.y);
    let normal_mat = transpose(mat3x3f(inv_trans[0].xyz, inv_trans[1].xyz, inv_trans[2].xyz));
    return normalize(normal_mat * local_normal).xyz;
}

fn compute_tri_geom_normal(instance_idx: u32, tri_idx: u32) -> vec3f
{
    let instance = instances[instance_idx];
    let mesh_idx = instance.mesh_idx;

    let test = transpose(instance.transpose_inverse_transform);
    let inv_trans = mat4x4f(vec4f(test[0], 0.0f), vec4f(test[1], 0.0f), vec4f(test[2], 0.0f), vec4f(test[3], 1.0f));

    let v0 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
    let v1 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
    let v2 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];

    let local_normal = normalize(cross(v2 - v0, v1 - v0));
    let normal_mat = transpose(mat3x3f(inv_trans[0].xyz, inv_trans[1].xyz, inv_trans[2].xyz));
    return normalize(normal_mat * local_normal);
}

// Safely wraps values in the [0, 1] range.
fn dir_to_env_uv(dir: vec3f, env_idx: u32) -> vec2f
{
    let env = environments[env_idx];
    let trans_dir = transform_direction_inverse(mat3x3f(env.transform[0].xyz, env.transform[1].xyz, env.transform[2].xyz), dir);
    var uv = vec2f(atan2(trans_dir.z, trans_dir.x) / (2.0f * PI), acos(clamp(trans_dir.y, -1.0f, 1.0f)) / PI);
    if uv.x < 0.0f { uv.x += 1.0f; }
    if uv.x > 1.0f { uv.x -= 1.0f; }
    return uv;
}

fn env_idx_to_dir(idx: u32, env: u32) -> vec3f
{
    let env_tex_idx = environments[env].emission_tex_idx;
    let env_tex_size = textureDimensions(textures[env_tex_idx]);

    let coords = vec2u(idx % env_tex_size.x, idx / env_tex_size.x);
    let uv = (vec2f(coords) + 0.5f) / vec2f(env_tex_size);
    return env_uv_to_dir(uv, env);
}

fn env_uv_to_dir(uv: vec2f, env: u32) -> vec3f
{
    let dir = vec3f(cos(uv.x * 2.0f * PI) * sin(uv.y * PI),
                    cos(uv.y * PI),
                    sin(uv.x * 2.0f * PI) * sin(uv.y * PI));
    return transform_dir(dir, environments[env].transform);
}

// As far as I know you need an extra extension to pass
// pointers to arrays in functions, so unless we want to
// require that, this function is duplicated.
fn sample_env_alias_table(env_idx: u32) -> u32
{
    let num_env = arrayLength(&environments);
    if num_env == 0 { return 0u; }

    let alias_table_size = arrayLength(&env_alias_tables[env_idx].data);
    let rnd_idx = random_u32_range_unsafe(alias_table_size);
    let bin = env_alias_tables[env_idx].data[rnd_idx];
    if random_f32() >= bin.alias_threshold {
        return bin.alias_idx;
    } else {
        return rnd_idx;
    }
}

fn sample_instance_alias_table(light_idx: u32) -> u32
{
    let num_lights = arrayLength(&lights);
    if num_lights == 0 { return 0u; }

    let alias_table_size = arrayLength(&alias_tables[light_idx].data);
    let rnd_idx = random_u32_range_unsafe(alias_table_size);
    let bin = alias_tables[light_idx].data[rnd_idx];
    if random_f32() >= bin.alias_threshold {
        return bin.alias_idx;
    } else {
        return rnd_idx;
    }
}

//////////////////////////////////////////////
// Utils and constants
//////////////////////////////////////////////

const F32_MAX: f32 = 0x1.fffffep+127;  // WGSL does not yet have a "max(f32)"
const U32_MAX: u32 = 4294967295;
const PI: f32 = 3.14159265358979323846264338327950288;

fn transform_point(p: vec3f, transform: mat4x4f)->vec3f
{
    let p_vec4 = vec4f(p, 1.0f);
    let transformed = transform * p_vec4;
    return transformed.xyz;
    //return (transformed / transformed.w).xyz;
}

fn transform_dir(dir: vec3f, transform: mat4x4f)->vec3f
{
    let dir_vec4 = vec4f(dir, 0.0f);
    return normalize((transform * dir_vec4).xyz);
}

fn transform_ray(ray: Ray, transform: mat4x4f) -> Ray
{
    var res = ray;
    res.ori = transform_point(res.ori, transform);
    res.dir = transform_dir(res.dir, transform);
    res.inv_dir = 1.0f / res.dir;
    return res;
}

fn transform_ray_without_normalizing_direction(ray: Ray, transform: mat4x4f) -> Ray
{
    var res = ray;
    res.ori = transform_point(res.ori, transform);

    let dir_vec4 = vec4f(res.dir, 0.0f);
    res.dir = (transform * dir_vec4).xyz;
    res.inv_dir = 1.0f / res.dir;
    return res;
}

fn same_hemisphere(normal: vec3f, outgoing: vec3f, incoming: vec3f) -> bool
{
    return dot(normal, outgoing) * dot(normal, incoming) >= 0;
}

// Ideally should not be used, but useful for debugging.
// https://gist.github.com/mattatz/86fff4b32d198d0928d0fa4ff32cf6fa
fn mat4x4f_inverse(m: mat4x4f) -> mat4x4f
{
    let n11 = m[0][0]; let n12 = m[1][0]; let n13 = m[2][0]; let n14 = m[3][0];
    let n21 = m[0][1]; let n22 = m[1][1]; let n23 = m[2][1]; let n24 = m[3][1];
    let n31 = m[0][2]; let n32 = m[1][2]; let n33 = m[2][2]; let n34 = m[3][2];
    let n41 = m[0][3]; let n42 = m[1][3]; let n43 = m[2][3]; let n44 = m[3][3];

    let t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    let t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    let t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    let t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    let det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    let idet = 1.0f / det;

    var ret = mat4x4f();

    ret[0][0] = t11 * idet;
    ret[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * idet;
    ret[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * idet;
    ret[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * idet;

    ret[1][0] = t12 * idet;
    ret[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * idet;
    ret[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * idet;
    ret[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * idet;

    ret[2][0] = t13 * idet;
    ret[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * idet;
    ret[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * idet;
    ret[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * idet;

    ret[3][0] = t14 * idet;
    ret[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * idet;
    ret[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * idet;
    ret[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * idet;

    return ret;
}

fn srgb_to_linear(srgb: f32) -> f32
{
    if srgb <= 0.04045 {
        return srgb / 12.92f;
    } else {
        return pow((srgb + 0.055f) / (1.0f + 0.055f), 2.4f);
    }
}

// TODO: @speed, use vec3b and lerp instead.
fn vec3f_srgb_to_linear(srgb: vec3f) -> vec3f
{
    return vec3f(srgb_to_linear(srgb.x), srgb_to_linear(srgb.y), srgb_to_linear(srgb.z));
}

// From: https://x.com/_Humus_/status/1074973351276371968
// https://sakibsaikia.github.io/graphics/2022/01/04/Nan-Checks-In-HLSL.html
fn is_finite(x: f32) -> bool
{
    return (u32(x) & 0x7F800000) != 0x7F800000;
}

fn is_positive_finite(x: f32) -> bool
{
    return u32(x) < 0x7F800000;
}

fn is_nan(x: f32) -> bool
{
    return (u32(x) & 0x7FFFFFFF) > 0x7F800000;
}

fn is_inf(x: f32) -> bool
{
    return (u32(x) & 0x7FFFFFFF) == 0x7F800000;
}

fn vec3f_is_finite(v: vec3f) -> bool
{
    return all((vec3u(v) & vec3u(0x7F800000)) != vec3u(0x7F800000));
}

fn orthonormalize(a: vec3f, b: vec3f) -> vec3f
{
    return normalize(a - b * dot(a, b));
}

fn transform_vector_inverse(a: mat3x3f, b: vec3f) -> vec3f
{
    return vec3f(dot(a[0], b), dot(a[1], b), dot(a[2], b));
}

fn transform_direction_inverse(a: mat3x3f, b: vec3f) -> vec3f
{
    return normalize(transform_vector_inverse(a, b));
}

/*
fn mat3x4f_inverse(a: mat3x4f) -> mat3x4f
{
    // Inverse of the 3x3 rotation/scale matrix.
    let cross_yz = cross(a[1], a[2]).xyz;
    let cross_zx = cross(a[2], a[0]).xyz;
    let cross_xy = cross(a[0], a[1]).xyz;
    let adjoint = transpose(mat3x3f(cross_yz, cross_zx, cross_xy));
    let determinant = dot(a[0], cross_yz);
    let minv = adjoint * (1.0f / determinant);

    let t = vec3f();
    let test = t * a;

    // Invert translation (easy to do).
    return mat3x4f();
    // return mat3x4f(minv[0], minv[1], minv[2], -(minv * a[3]));
}
*/

// Debug visualization

fn get_heatmap_color(val: f32, min: f32, max: f32) -> vec3f
{
    let wavelength = 380.0f + 370.0f * max(val - min, 0.0f) / max(max - min, 0.0f);
    var color = vec3f();
    if wavelength <= 380.0f
    {
        color.r = 0.0;
		color.g = 0.0;
		color.b = 0.0;
    }
    else if wavelength > 380.0f && wavelength <= 440.0f
    {
        color.r = -(wavelength - 440.0) / (440.0 - 380.0)/3;
        color.g = 0.0;
        color.b = 0.8;
    }
    else if wavelength >= 440.0 && wavelength <= 490.0
    {
        color.r = 0.0;
        color.g = (wavelength - 440.0) / (490.0 - 440.0);
        color.b = 1.0;
    }
    else if wavelength >= 490.0 && wavelength <= 510.0
    {
        color.r = 0.0;
        color.g = 1.0;
        color.b = -(wavelength - 510.0) / (510.0 - 490.0);
    }
    else if wavelength >= 510.0 && wavelength <= 580.0
    {
        color.r = (wavelength - 510.0) / (580.0 - 510.0);
        color.g = 1.0;
        color.b = 0.0;
    }
    else if wavelength >= 580.0 && wavelength <= 645.0
    {
        color.r = 1.0;
        color.g = -(wavelength - 645.0) / (645.0 - 580.0);
        color.b = 0.0;
    }
    else if wavelength >= 645.0 && wavelength <= 780.0
    {
        color.r = 1.0;
        color.g = 0.0;
        color.b = 0.0;
    }
    else
    {
        color = vec3f(1.0f);
    }

    // Gamma correct.
    const gamma = 0.8f;
    var factor = 1.0f;
    let white = vec3f(1.0f);
    if wavelength >= 380 && wavelength < 420 {
        factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380);
    } else if wavelength >= 420 && wavelength < 701 {
        factor = 1.0;
    } else if wavelength >= 701 && wavelength < 781 {
        factor = 0.3 + 0.7*(780 - wavelength) / (780 - 700);
        return pow((color + factor*white), vec3f(gamma));
    } else {
        factor = 1.0f;
    }

    color = pow(factor * color, vec3f(gamma));
    return color;
}

// For quick printf-like debugging.

var<private> debug_color = vec3f();
var<private> debug_enable = false;

fn write_debug_color(global_id: vec2u, output_dim: vec2u)
{
    if debug_enable
    {
        if all(global_id < output_dim) {
            textureStore(output_hdr, global_id.xy, vec4f(debug_color, 1.0f));
        }
    }
}
