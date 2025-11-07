
@group(3) @binding(0) var rt_tlas: acceleration_structure;

fn ray_scene_intersection(ray: Ray)->HitInfo
{
    var rq: ray_query;
    rayQueryInitialize(&rq, rt_tlas, RayDesc(0u, 0xFFu, RAY_EPSILON, F32_MAX, ray.ori, ray.dir));
    rayQueryProceed(&rq);

    let intersection = rayQueryGetCommittedIntersection(&rq);
    if intersection.kind == RAY_QUERY_INTERSECTION_NONE { return HitInfo(); }

    var res = HitInfo();
    res.hit = true;
    res.dst = intersection.t;
    res.uv  = intersection.barycentrics;
    res.instance_idx = intersection.instance_custom_data;
    res.tri_idx = intersection.primitive_index;
    res.hit_backside = !intersection.front_face;
    return res;
}
