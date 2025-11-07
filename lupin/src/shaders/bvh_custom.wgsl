
@group(3) @binding(0) var<storage, read> bvh_nodes_array: binding_array<BvhNodes>;
@group(3) @binding(1) var<storage, read> tlas_nodes: array<TlasNode>;

fn ray_scene_intersection(ray: Ray)->HitInfo
{
    var tlas_stack: array<u32, MAX_TLAS_DEPTH+1>;  // local
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

            let result = ray_mesh_intersection(ray_trans, min_hit.x, instance.mesh_idx);
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
