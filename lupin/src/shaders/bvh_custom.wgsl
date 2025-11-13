
@group(3) @binding(0) var<storage, read> bvh_nodes_array: binding_array<BvhNodes>;
@group(3) @binding(1) var<storage, read> tlas_nodes: array<TlasNode>;

const _MAX_TLAS_DEPTH: u32 = 50;  // This roughly supports 2^(_MAX_TLAS_DEPTH / 3) objects.

fn ray_scene_intersection(ray: Ray)->HitInfo
{
    var tlas_stack: array<u32, _MAX_TLAS_DEPTH+1>;  // local
    var stack_idx: u32 = 2;
    tlas_stack[0] = 0u;
    tlas_stack[1] = 0u;

    // t, u, v, det
    var min_hit = vec4f(F32_MAX, 0.0f, 0.0f, 0.0f);
    var tri_idx: u32 = 0;
    var instance_idx: u32 = 0;
    while stack_idx > 1
    {
        stack_idx--;
        //if stack_idx >= (_MAX_TLAS_DEPTH+1) { set_debug_output_color(vec3f(1.0f, 0.0f, 0.0f)); }
        let node = tlas_nodes[tlas_stack[stack_idx]];

        if node.left == 0u  // Leaf node
        {
            let instance = instances[node.instance_idx];

            var ray_trans = ray;
            // NOTE: We do vector * matrix because it's transposed.
            ray_trans.ori = (vec4f(ray_trans.ori, 1.0f) * instance.transpose_inverse_transform).xyz;
            // NOTE: We do not normalize because we do want ray.dir's length to change.
            ray_trans.dir = (vec4f(ray_trans.dir, 0.0f) * instance.transpose_inverse_transform).xyz;
            ray_trans.inv_dir = 1.0f / ray_trans.dir;

            let result = _ray_mesh_intersection(ray_trans, min_hit.x, instance.mesh_idx);
            if result.hit.x < min_hit.x
            {
                min_hit      = result.hit;
                tri_idx      = result.tri_idx;
                instance_idx = node.instance_idx;
            }
        }
        else  // Non-leaf node
        {
            let left_child_node  = tlas_nodes[node.left];
            let right_child_node = tlas_nodes[node.right];

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
                    tlas_stack[stack_idx] = node.right;
                    stack_idx++;
                }

                if push_left
                {
                    tlas_stack[stack_idx] = node.left;
                    stack_idx++;
                }
            }
            else
            {
                if push_left
                {
                    tlas_stack[stack_idx] = node.left;
                    stack_idx++;
                }

                if push_right
                {
                    tlas_stack[stack_idx] = node.right;
                    stack_idx++;
                }
            }
        }
    }

    var hit_info = HitInfo();
    if min_hit.x != F32_MAX
    {
        hit_info.hit = true;
        hit_info.dst = min_hit.x;
        hit_info.uv = min_hit.yz;
        hit_info.instance_idx = instance_idx;
        hit_info.tri_idx = tri_idx;
        hit_info.hit_backside = min_hit.w > 0.0f;
    }

    return hit_info;
}

fn compute_instance_lights_pdf(ray: Ray) -> f32
{
    let num_lights = select(arrayLength(&lights), 0u, (constants.flags & FLAG_LIGHTS_EMPTY) != 0);
    let num_envs = select(arrayLength(&environments), 0u, (constants.flags & FLAG_ENVS_EMPTY) != 0);

    let pos = ray.ori;
    let incoming = ray.dir;

    var pdf = 0.0f;
    for(var i = 0u; i < num_lights; i++)
    {
        let light = lights[i];
        let instance_idx = light.instance_idx;

        var light_pdf = 0.0f;
        var next_pos = pos;
        for(var bounce = 0u; bounce < 100; bounce++)
        {
            var ray = Ray(next_pos, incoming, 1.0f / incoming);
            let hit_info = _ray_instance_intersection(ray, F32_MAX, instance_idx);
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

    return pdf;
}

fn ray_skip_alpha_stochastically(start_ray: Ray) -> HitInfo
{
    var hit = HitInfo();
    var ray = start_ray;
    var dst = 0.0f;
    for(var opacity_bounce = 0u; opacity_bounce < MAX_OPACITY_BOUNCES; opacity_bounce++)
    {
        hit = ray_scene_intersection(ray);
        if !hit.hit { break; }

        dst += hit.dst;

        let mat_point = get_material_point(hit);
        if mat_point.opacity < 1.0f && random_f32() >= mat_point.opacity {
            ray.ori = ray.ori + ray.dir * hit.dst;
        } else {
            break;
        }
    }

    // Make sure the distance is accurate to the initial
    // ray, not the last (shorter) one.
    hit.dst = dst;
    return hit;
}

/////////////////////
// Internals
////////////////////

const _MAX_BVH_DEPTH: u32 = 25;

struct RayMeshIntersectionResult
{
    hit: vec4f,  // t, u, v, det
    tri_idx: u32,
}

// cur_min_hit_dst should be F32_MAX if absent.
fn _ray_mesh_intersection(ray: Ray, cur_min_hit_dst: f32, mesh_idx: u32) -> RayMeshIntersectionResult
{
    var bvh_stack: array<u32, _MAX_BVH_DEPTH+1>;  // local
    var stack_idx = 2u;
    bvh_stack[0] = 0u;
    bvh_stack[1] = 0u;

    // t, u, v
    var min_hit = vec4f(cur_min_hit_dst, 0.0f, 0.0f, 0.0f);
    var tri_idx = 0u;
    while stack_idx > 1
    {
        stack_idx--;
        //if stack_idx >= (_MAX_BVH_DEPTH+1) { set_debug_output_color(vec3f(0.0f, 1.0f, 0.0f)); }
        let node = bvh_nodes_array[mesh_idx].data[bvh_stack[stack_idx]];

        if node.tri_count > 0u  // Leaf node
        {
            let tri_begin = node.tri_begin_or_first_child;
            let tri_count = node.tri_count;
            for(var i: u32 = tri_begin; i < tri_begin + tri_count; i++)
            {
                let v0 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[i*3 + 0]];
                let v1 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[i*3 + 1]];
                let v2 = verts_pos_array[mesh_idx].data[indices_array[mesh_idx].data[i*3 + 2]];
                let hit = ray_tri_dst(ray, v0, v1, v2);
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
                    bvh_stack[stack_idx] = right_child;
                    stack_idx++;
                }

                if push_left
                {
                    bvh_stack[stack_idx] = left_child;
                    stack_idx++;
                }
            }
            else
            {
                if push_left
                {
                    bvh_stack[stack_idx] = left_child;
                    stack_idx++;
                }

                if push_right
                {
                    bvh_stack[stack_idx] = right_child;
                    stack_idx++;
                }
            }
        }
    }

    return RayMeshIntersectionResult(min_hit, tri_idx);
}

fn _ray_instance_intersection(ray: Ray, cur_min_hit_dst: f32, instance_idx: u32) -> RayMeshIntersectionResult
{
    let instance = instances[instance_idx];

    let test = transpose(instance.transpose_inverse_transform);
    let trans_mat4 = mat4x4f(vec4f(test[0], 0.0f), vec4f(test[1], 0.0f), vec4f(test[2], 0.0f), vec4f(test[3], 1.0f));

    let ray_trans = transform_ray_without_normalizing_direction(ray, trans_mat4);
    let result = _ray_mesh_intersection(ray_trans, cur_min_hit_dst, instance.mesh_idx);
    return result;
}
