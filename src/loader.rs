
use crate::base::*;

use tobj::*;
use std::ptr::NonNull;
use crate::renderer_wgpu::*;

pub fn load_scene_custom_format(path: &str)
{

}

pub fn load_scene_obj(path: &str, renderer: &mut Renderer)->Scene
{
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    assert!(scene.is_ok());

    let (mut models, materials) = scene.expect("Failed to load OBJ file");
    //let materials = materials.expect("Failed to load MTL file");

    println!("Num models: {}", models.len());
    //println!("Num materials: {}", materials.len());

    if models.len() > 0
    {
        let mesh = &mut models[0].mesh;

        // Construct the buffer to send to GPU. Include an extra float
        // for 16-byte padding (which seems to be required in WebGPU).
        let mut verts_pos: Vec<f32> = Vec::new();
        verts_pos.reserve_exact(mesh.positions.len() + mesh.positions.len() / 3);
        for i in (0..mesh.positions.len()).step_by(3)
        {
            verts_pos.push(mesh.positions[i + 0]);
            verts_pos.push(mesh.positions[i + 1]);
            verts_pos.push(mesh.positions[i + 2]);
            verts_pos.push(0.0);
        }

        let bvh = create_bvh(verts_pos.as_slice(), &mut mesh.indices);

        let bvh_buf = renderer.upload_buffer(to_u8_slice(&bvh));
        let verts_pos_buf = renderer.upload_buffer(to_u8_slice(&verts_pos));
        let indices_buf = renderer.upload_buffer(to_u8_slice(&mesh.indices));

        return Scene
        {
            verts: verts_pos_buf,
            indices: indices_buf,
            bvh_nodes: bvh_buf,
        }
    }

    return Scene
    {
        verts: renderer.empty_buffer(),
        indices: renderer.empty_buffer(),
        bvh_nodes: renderer.empty_buffer()
    }
}

pub fn unload_scene(scene: &mut Scene, renderer: &mut Renderer)
{

}

pub fn load_image(path: &str, required_channels: i32)
{
    
}

/////////
// BVH creation

// NOTE: modifies the indices array to change the order of triangles
// based on BVH
pub fn create_bvh(verts: &[f32], indices: &mut[u32])->Vec<BvhNode>
{
    println!("Building bvh...");

    let (aabb_min, aabb_max) = compute_aabb(verts, indices, 0, indices.len() as u32 / 3);
    let bvh_root = BvhNode
    {
        aabb_min,
        aabb_max,
        tri_begin_or_first_child: 0,  // Set tri_begin to 0
        tri_count: indices.len() as u32 / 3
    };

    let mut bvh: Vec<BvhNode> = Vec::new();
    bvh.reserve(1);
    bvh.push(bvh_root);
    // NOTE: Push a fictitious element so that
    // both of a node's children will reside
    // in the same cache line
    //bvh_nodes.push(Default::default());

    const BVH_MAX_DEPTH: u32 = 20;
    bvh_split(&mut bvh, verts, indices, 0, BVH_MAX_DEPTH, 0);

    println!("Done!");
    return bvh;
}

pub fn bvh_split(bvh: &mut Vec<BvhNode>, verts: &[f32], indices: &mut[u32], node: usize, max_depth: u32, depth: u32)
{
    if depth == max_depth { return; }

    let split = choose_split(bvh.as_slice(), verts, indices, node);
    if !split.performed_split { return; }

    let cur_tri_begin = bvh[node].tri_begin_or_first_child;
    let cur_tri_count = bvh[node].tri_count;
    let cur_tri_end   = cur_tri_begin + cur_tri_count;

    // For each child, sort the indices so that they're contiguous
    // and the tris can be referenced with only tri_begin and tri_count

    // Current tri index for left child
    let mut tri_left_idx = cur_tri_begin;

    for tri in cur_tri_begin..cur_tri_end
    {
        let (t0, t1, t2) = get_tri(verts, indices, tri as usize);

        let center = compute_tri_centroid(t0, t1, t2);
        if center[split.axis] <= split.pos
        {
            if tri != tri_left_idx
            {
                // Swap triangle of index tri_left_idx with
                // triangle of index tri_idx
                swap_tris(indices, tri_left_idx, tri);
            }

            tri_left_idx += 1;
        }
    }

    // Set triangle begin and count of children nodes, because they're currently leaves
    let left_begin  = cur_tri_begin;
    let left_count  = tri_left_idx as u32 - cur_tri_begin;
    let right_begin = tri_left_idx as u32;
    let right_count = (cur_tri_count - left_count) as u32;

    // Only proceed if there is a meaningful subdivision
    // (also, if tri_count is 0 it will be interpreted
    // as a non-leaf by the shader)
    if left_count == 0 || right_count == 0 { return; }

    // We've decided to split
    bvh.reserve(bvh.len() + 2);
    bvh.push(Default::default());
    bvh.push(Default::default());
    let left: usize  = bvh.len() - 2;
    let right: usize = bvh.len() - 1;

    bvh[left].tri_begin_or_first_child = left_begin;
    bvh[left].tri_count = left_count;
    bvh[right].tri_begin_or_first_child = right_begin;
    bvh[right].tri_count = right_count;

    (bvh[left].aabb_min, bvh[left].aabb_max) = (split.aabb_left_min, split.aabb_left_max);
    (bvh[right].aabb_min, bvh[right].aabb_max) = (split.aabb_right_min, split.aabb_right_max);

    // The current node is not a leaf node anymore, so set its first child
    bvh[node].tri_begin_or_first_child = left as u32;  // Set first_child
    bvh[node].tri_count = 0;

    //println!("Depth {}", depth);
    /*println!("Performed split, parent: {} {}, left: {} {}, right: {} {}", cur_tri_begin, cur_tri_count,
                                                                          bvh[left].tri_begin_or_first_child, bvh[left].tri_count,
                                                                          bvh[right].tri_begin_or_first_child, bvh[right].tri_count);
    */

    bvh_split(bvh, verts, indices, left, max_depth, depth + 1);
    bvh_split(bvh, verts, indices, right, max_depth, depth + 1);
}

#[derive(Default)]
pub struct BvhSplit
{
    performed_split: bool,
    axis: usize,
    pos: f32,
    cost: f32,

    // Bounding box info
    aabb_left_min: Vec3,
    aabb_left_max: Vec3,
    aabb_right_min: Vec3,
    aabb_right_max: Vec3,

    num_tris_left: u32,
    num_tris_right: u32
}
pub fn choose_split(bvh: &[BvhNode], verts: &[f32], indices: &mut[u32], node: usize)->BvhSplit
{
    const NUM_TESTS_PER_AXIS: u32 = 5;

    let mut res: BvhSplit = Default::default();

    let size = bvh[node].aabb_max - bvh[node].aabb_min;
    let current_cost = node_cost(size, bvh[node].tri_count);

    // Initialize the cost as the cost of the parent
    // node by itself. Only split if the cost is actually lower
    // after the split.
    res.cost = current_cost;
    for axis in 0..3usize
    {
        let bounds_start = bvh[node].aabb_min[axis];
        let bounds_end   = bvh[node].aabb_max[axis];
        for i in 0..NUM_TESTS_PER_AXIS
        {
            // Test a fraction of the bounding box each iteration (1/tests, 2/tests, ... 1)
            let split_t: f32   = (i + 1) as f32 / (NUM_TESTS_PER_AXIS) as f32;
            let split_pos: f32 = lerp_f32(bounds_start, bounds_end, split_t);

            // Compute cost of this split
            {
                let tri_begin = bvh[node].tri_begin_or_first_child as usize;
                let tri_count = bvh[node].tri_count as usize;
                let tri_end   = tri_begin + tri_count as usize;

                let mut aabb_left_min:  Vec3 = Default::default();
                let mut aabb_left_max:  Vec3 = Default::default();
                let mut aabb_right_min: Vec3 = Default::default();
                let mut aabb_right_max: Vec3 = Default::default();
                let mut first_left  = true;
                let mut first_right = true;
                let mut num_tris_left = 0;
                let mut num_tris_right = 0;

                // Count number of triangles and encompassing bounding box size
                for tri in tri_begin..tri_end
                {
                    let (t0, t1, t2) = get_tri(verts, indices, tri);
                    let tri_center = compute_tri_centroid(t0, t1, t2);
                    if tri_center[axis] <= split_pos
                    {
                        if first_left
                        {
                            aabb_left_min = t0;
                            aabb_left_max = t0;
                            first_left = false;
                        }

                        grow_aabb_to_include_tri(&mut aabb_left_min, &mut aabb_left_max, t0, t1, t2);
                        num_tris_left += 1;
                    }
                    else
                    {
                        if first_right
                        {
                            aabb_right_min = t0;
                            aabb_right_max = t0;
                            first_right = false;
                        }

                        grow_aabb_to_include_tri(&mut aabb_right_min, &mut aabb_right_max, t0, t1, t2);
                        num_tris_right += 1;
                    }
                }

                let size_left  = aabb_left_max - aabb_left_min;
                let size_right = aabb_right_max - aabb_right_min;

                let cost = node_cost(size_left, num_tris_left) + node_cost(size_right, num_tris_right);

                if cost < res.cost && num_tris_left > 0 && num_tris_right > 0
                {
                    res.performed_split = true;
                    res.cost = cost;
                    res.axis = axis;
                    res.pos = split_pos;
                    res.aabb_left_min = aabb_left_min;
                    res.aabb_right_min = aabb_right_min;
                    res.aabb_left_max = aabb_left_max;
                    res.aabb_right_max = aabb_right_max;
                    res.num_tris_left = num_tris_left;
                    res.num_tris_right = num_tris_right;
                }
            }
        }
    }

    return res;
}

pub fn node_cost(size: Vec3, num_tris: u32)->f32
{
    // Surface Area Heuristic (SAH)
    // Computing half area instead of full area because
    // it's slightly faster and doesn't change the result
    let half_area: f32 = size.x * (size.y * size.z) + size.y * size.z;
    return half_area * num_tris as f32;
}

pub fn get_tri(verts: &[f32], indices: &[u32], tri_idx: usize)->(Vec3, Vec3, Vec3)
{
    let idx0 = (indices[tri_idx*3+0]*4) as usize;
    let idx1 = (indices[tri_idx*3+1]*4) as usize;
    let idx2 = (indices[tri_idx*3+2]*4) as usize;

    let t0 = Vec3
    {
        x: verts[idx0 + 0],
        y: verts[idx0 + 1],
        z: verts[idx0 + 2],
    };
    let t1 = Vec3
    {
        x: verts[idx1 + 0],
        y: verts[idx1 + 1],
        z: verts[idx1 + 2],
    };
    let t2 = Vec3
    {
        x: verts[idx2 + 0],
        y: verts[idx2 + 1],
        z: verts[idx2 + 2],
    };

    return (t0, t1, t2);
}

pub fn swap_tris(indices: &mut[u32], tri_a: u32, tri_b: u32)
{
    let tri_a = tri_a as usize;
    let tri_b = tri_b as usize;
    let idx0 = indices[tri_a*3+0];
    let idx1 = indices[tri_a*3+1];
    let idx2 = indices[tri_a*3+2];
    indices[tri_a*3 + 0] = indices[tri_b*3 + 0];
    indices[tri_a*3 + 1] = indices[tri_b*3 + 1];
    indices[tri_a*3 + 2] = indices[tri_b*3 + 2];
    indices[tri_b*3 + 0] = idx0 as u32;
    indices[tri_b*3 + 1] = idx1 as u32;
    indices[tri_b*3 + 2] = idx2 as u32;
}

pub fn compute_aabb(verts: &[f32], indices: &[u32], tri_begin: u32, tri_count: u32)->(Vec3, Vec3)
{
    let idx0 = indices[0] as usize;
    let mut aabb_min = Vec3 { x: verts[idx0], y: verts[idx0 + 1], z: verts[idx0 + 2] };
    let mut aabb_max = aabb_min;

    let tri_end = tri_begin + tri_count;
    for tri in tri_begin..tri_end
    {
        let (t0, t1, t2) = get_tri(verts, indices, tri as usize);
        grow_aabb_to_include_tri(&mut aabb_min, &mut aabb_max, t0, t1, t2);
    }

    return (aabb_min, aabb_max);
}
