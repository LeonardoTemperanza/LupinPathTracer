
@group(3) @binding(0) var rt_tlas: acceleration_structure;

// NOTE: Requires masks to be set up with the following bitflags
// and instance_custom_data to be 0 if it's not a light and the light index
// otherwise. Note that instance_custom_data must only use 24 bits.

// Coupled to RT BVH construction.
const RT_MASK_DEFAULT: u32 = (1 << 0);
const RT_MASK_LIGHT:   u32 = (1 << 1);

/*
RayQuery docs:

struct RayDesc {
    // Contains flags to use for this ray (e.g. consider all `Blas`es opaque)
    flags: u32,
    // If the bitwise and of this and any `TlasInstance`'s `mask` is not zero then the object inside
    // the `Blas` contained within that `TlasInstance` may be hit.
    cull_mask: u32,
    // Only points on the ray whose t is greater than this may be hit.
    t_min: f32,
    // Only points on the ray whose t is less than this may be hit.
    t_max: f32,
    // The origin of the ray.
    origin: vec3<f32>,
    // The direction of the ray, t is calculated as the length down the ray divided by the length of `dir`.
    dir: vec3<f32>,
}

struct RayIntersection {
    // the kind of the hit, no other member of this structure is useful if this is equal
    // to constant `RAY_QUERY_INTERSECTION_NONE`.
    kind: u32,
    // Distance from starting point, measured in units of `RayDesc::dir`.
    t: f32,
    // Corresponds to `instance.custom_data` where `instance` is the `TlasInstance`
    // that the intersected object was contained in.
    instance_custom_data: u32,
    // The index into the `TlasPackage` to get the `TlasInstance` that the hit object is in
    instance_index: u32,
    // The offset into the shader binding table. Currently, this value is always 0.
    sbt_record_offset: u32,
    // The index into the `Blas`'s build descriptor (e.g. if `BlasBuildEntry::geometry` is
    // `BlasGeometries::TriangleGeometries` then it is the index into that contained vector).
    geometry_index: u32,
    // The object hit's index into the provided buffer (e.g. if the object is a triangle
    // then this is the triangle index)
    primitive_index: u32,
    // Two of the barycentric coordinates, the third can be calculated (only useful if this is a triangle).
    barycentrics: vec2<f32>,
    // Whether the hit face is the front (only useful if this is a triangle).
    front_face: bool,
    // Matrix for converting from object-space to world-space.
    //
    // This matrix needs to be on the left side of the multiplication. Using it the other way round will not work.
    // Use it this way: `let transformed_vector = intersecion.object_to_world * vec4<f32>(x, y, z, transform_multiplier);
    object_to_world: mat4x3<f32>,
    // Matrix for converting from world-space to object-space
    //
    // This matrix needs to be on the left side of the multiplication. Using it the other way round will not work.
    // Use it this way: `let transformed_vector = intersecion.world_to_object * vec4<f32>(x, y, z, transform_multiplier);
    world_to_object: mat4x3<f32>,
}

/// -- Flags for `RayDesc::flags` --

// All `Blas`es are marked as opaque.
const FORCE_OPAQUE = 0x1;

// All `Blas`es are marked as non-opaque.
const FORCE_NO_OPAQUE = 0x2;

// Instead of searching for the closest hit return the first hit.
const TERMINATE_ON_FIRST_HIT = 0x4;

// Unused: implemented for raytracing pipelines.
const SKIP_CLOSEST_HIT_SHADER = 0x8;

// If `RayIntersection::front_face` is false do not return a hit.
const CULL_BACK_FACING = 0x10;

// If `RayIntersection::front_face` is true do not return a hit.
const CULL_FRONT_FACING = 0x20;

// If the `Blas` a intersection is checking is marked as opaque do not return a hit.
const CULL_OPAQUE = 0x40;

// If the `Blas` a intersection is checking is not marked as opaque do not return a hit.
const CULL_NO_OPAQUE = 0x80;

// If the `Blas` a intersection is checking contains triangles do not return a hit.
const SKIP_TRIANGLES = 0x100;

// If the `Blas` a intersection is checking contains AABBs do not return a hit.
const SKIP_AABBS = 0x200;

/// -- Constants for `RayIntersection::kind` --

// The ray hit nothing.
const RAY_QUERY_INTERSECTION_NONE = 0;

// The ray hit a triangle.
const RAY_QUERY_INTERSECTION_TRIANGLE = 1;

// The ray hit a custom object, this will only happen in a committed intersection
// if a ray which intersected a bounding box for a custom object which was then committed.
const RAY_QUERY_INTERSECTION_GENERATED = 2;

// The ray hit a AABB, this will only happen in a candidate intersection
// if the ray intersects the bounding box for a custom object.
const RAY_QUERY_INTERSECTION_AABB = 3;
*/

// Will simply return the closest hit.
fn ray_scene_intersection(ray: Ray) -> HitInfo
{
    var rq: ray_query;
    let flags = RAY_FLAG_SKIP_AABBS;
    let mask = 0xFFu;
    rayQueryInitialize(&rq, rt_tlas, RayDesc(flags, mask, RAY_EPSILON, F32_MAX, ray.ori, ray.dir));
    rayQueryProceed(&rq);

    let hit = rayQueryGetCommittedIntersection(&rq);
    if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE { return HitInfo(); }

    var res = HitInfo();
    res.hit = true;
    res.dst = hit.t;
    res.uv  = hit.barycentrics;
    res.instance_idx = hit.instance_index;
    res.tri_idx = hit.primitive_index;
    res.hit_backside = !hit.front_face;
    return res;
}

fn compute_instance_lights_pdf(ray: Ray) -> f32
{
    var rq: ray_query;
    // Consider everything to be not opaque, a.k.a consider all hits.
    let flags = RAY_FLAG_SKIP_AABBS | RAY_FLAG_FORCE_NO_OPAQUE;
    let mask = RT_MASK_LIGHT;
    rayQueryInitialize(&rq, rt_tlas, RayDesc(flags, mask, RAY_EPSILON, F32_MAX, ray.ori, ray.dir));

    var pdf = 0.0f;
    var count = 0.0f;
    while rayQueryProceed(&rq)
    {
        let hit = rayQueryGetCandidateIntersection(&rq);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE { continue; }

        let light_idx = hit.instance_custom_data;
        let instance_idx = hit.instance_index;
        let light = lights[light_idx];
        let mesh_idx = instances[instance_idx].mesh_idx;
        let tri_idx = hit.primitive_index;

        // Compute light geometric normal
        var light_normal = vec3f();
        {
            let v0 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
            let v1 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
            let v2 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];

            let local_normal = normalize(cross(v2 - v0, v1 - v0));
            let normal_mat = transpose(mat3x3f(hit.world_to_object[0].xyz, hit.world_to_object[1].xyz, hit.world_to_object[2].xyz));
            light_normal = normalize(normal_mat * local_normal);
        }

        let light_pos = ray.ori + ray.dir * hit.t;

        let prob = alias_tables[light_idx].data[tri_idx].prob;
        let dist_sqr = dot(light_pos - ray.ori, light_pos - ray.ori);
        let cos_theta = abs(dot(light_normal, ray.dir));

        pdf += dist_sqr / (cos_theta * light.area);
        count += 1.0f;
    }

    return pdf;
}

/*
// Will return the closest hit, while skipping hits based on alpha.
fn ray_skip_alpha_stochastically(start_ray: Ray) -> HitInfo
{
    var rq: ray_query;
    rayQueryInitialize(&rq, rt_tlas, RayDesc(0u, 0xFFu, RAY_EPSILON, F32_MAX, ray.ori, ray.dir));
    rayQueryProceed(&rq);

    var closest_opaque_hit = RayIntersection();
    while rayQueryProceed(&rq)
    {
        let hit = rayQueryGetCandidateIntersection(&rq);
        if hit.kind != RAY_QUERY_INTERSECTION_TRIANGLE { continue; }

        let is_first_hit = closest_opaque_hit.kind == RAY_QUERY_INTERSECTION_NONE;
        if !is_first_hit && hit.t >= closest_opaque_hit.t { continue; }

        // Skip if alpha is exactly 0.0f
        let uv = hit.barycentrics;
        let instance_idx = hit.instance_custom_data;
        let tri_idx = hit.primitive_index;
        let alpha = _get_alpha(uv, tri_idx, instance_idx);
        if alpha == 0.0f { continue; }
    }

    if closest_opaque_hit.kind == RAY_QUERY_INTERSECTION_NONE { return HitInfo(); }

    var res = HitInfo();
    res.hit = true;
    res.dst = closest_opaque_hit.t;
    res.uv  = closest_opaque_hit.barycentrics;
    res.instance_idx = closest_opaque_hit.instance_custom_data;
    res.tri_idx = closest_opaque_hit.primitive_index;
    res.hit_backside = !closest_opaque_hit.front_face;
    return res;
}
*/

// Internals

fn _get_alpha(uv: vec2f, tri_idx: u32, instance_idx: u32) -> f32
{
    let instance = instances[instance_idx];
    let mesh_idx = instance.mesh_idx;
    let mat = materials[instance.mat_idx];
    let mesh_info = mesh_infos[mesh_idx];

    const mat_sampler_idx: u32 = 0;  // TODO!

    // Sample textures.
    var color_sample = vec4f(1.0f);
    if mesh_info.texcoords_buf_idx != SENTINEL_IDX
    {
        let uv0 = verts_texcoord_array[mesh_info.texcoords_buf_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 0]];
        let uv1 = verts_texcoord_array[mesh_info.texcoords_buf_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 1]];
        let uv2 = verts_texcoord_array[mesh_info.texcoords_buf_idx].data[indices_array[mesh_idx].data[tri_idx*3 + 2]];
        let w = 1.0f - uv.x - uv.y;
        let texcoords = uv0*w + uv1*uv.x + uv2*uv.y;

        if mat.color_tex_idx != SENTINEL_IDX {
            color_sample = textureSampleLevel(textures[mat.color_tex_idx], samplers[mat_sampler_idx], texcoords, 0.0f);
            color_sample = vec4f(vec3f_srgb_to_linear(color_sample.rgb), color_sample.a);
        }
    }

    let vert_color = get_vert_color(instance_idx, tri_idx, uv);
    return color_sample.a * mat.color.a * vert_color.a;
}
