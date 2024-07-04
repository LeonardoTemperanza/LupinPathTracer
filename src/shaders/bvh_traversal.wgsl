
#include "base"

struct HitInfo
{
    dst: f32,
    normal: vec3f,
    tex_coords: vec2f
}

struct DebugHitInfo
{
    hit_info: HitInfo,
    num_aabb_tests: u32,
    num_tri_tests: u32
}

fn ray_scene_intersection(ray: Ray)->HitInfo
{
    var stack: array<u32, 40>;  // Max BVH depth is 40
    var stack_idx: i32 = 1;
    stack[0] = 0u;

    var num_boxes_hit: i32 = 0;

    // t, u, v
    var min_hit = vec3f(f32_max, 0.0f, 0.0f);
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
                if right_dst < min_hit.x
                {
                    num_boxes_hit += 1;
                    stack[stack_idx] = right_child;
                    stack_idx++;
                }
                
                if left_dst < min_hit.x
                {
                    num_boxes_hit += 1;
                    stack[stack_idx] = left_child;
                    stack_idx++;
                }
            }
            else
            {
                if left_dst < min_hit.x
                {
                    num_boxes_hit += 1;
                    stack[stack_idx] = left_child;
                    stack_idx++;
                }

                if right_dst < min_hit.x
                {
                    num_boxes_hit += 1;
                    stack[stack_idx] = right_child;
                    stack_idx++;
                }
            }
        }
    }

    var hit_info: HitInfo = HitInfo(min_hit.x, vec3f(0.0f), vec2f(0.0f), num_boxes_hit);
    if hit_info.dst != f32_max
    {
        let vert0: Vertex = verts[indices[tri_idx*3 + 0]];
        let vert1: Vertex = verts[indices[tri_idx*3 + 1]];
        let vert2: Vertex = verts[indices[tri_idx*3 + 2]];
        let u = min_hit.y;
        let v = min_hit.z;
        let w = 1.0 - u - v;

        hit_info.normal = vert0.normal*w + vert1.normal*u + vert2.normal*v;
        hit_info.tex_coords = vert0.tex_coords*w + vert1.tex_coords*u + vert2.tex_coords*v;
    }

    return hit_info;
}

// Used for debugging and profiling
// NOTE: Since this is used for profiling, the traversal
// should match exactly the actual ray_scene_intersection above.
fn debug_ray_scene_intersection(ray: Ray)->DebugHitInfo
{
    var hit_info: HitInfo = HitInfo(min_hit.x, vec3f(0.0f), vec2f(0.0f), num_boxes_hit);
}