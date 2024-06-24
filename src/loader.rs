
use crate::base::*;

use tobj::*;
use std::ptr::NonNull;
use crate::renderer::*;

pub fn load_scene_custom_format(path: &str)
{

}

pub fn load_scene_obj(path: &str, renderer: &mut Renderer)->Scene
{
    println!("Loading file from disk...");

    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    assert!(scene.is_ok());

    let (mut models, materials) = scene.expect("Failed to load OBJ file");

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

        let bvh = build_bvh(verts_pos.as_slice(), &mut mesh.indices);

        let bvh_buf = renderer.upload_buffer(to_u8_slice(&bvh));
        let verts_pos_buf = renderer.upload_buffer(to_u8_slice(&verts_pos));
        let indices_buf = renderer.upload_buffer(to_u8_slice(&mesh.indices));

        let mut verts: Vec<Vertex> = Vec::new();
        verts.reserve_exact(mesh.positions.len() / 3);
        for vert_idx in 0..(mesh.positions.len() / 3)
        {
            let mut normal = Vec3::default();
            if mesh.normals.len() > 0
            {
                normal.x = mesh.normals[vert_idx*3+0];
                normal.y = mesh.normals[vert_idx*3+1];
                normal.z = mesh.normals[vert_idx*3+2];
                normal.normalize();
            };

            let mut tex_coords = Vec2::default();
            if mesh.texcoords.len() > 0
            {
                tex_coords.x = mesh.texcoords[vert_idx*2+0];
                tex_coords.y = mesh.texcoords[vert_idx*2+1];
                tex_coords.normalize();
            };

            let vert = Vertex { normal, padding0: 0.0, tex_coords, padding1: 0.0, padding2: 0.0 };

            verts.push(vert);
        }

        let verts_buf = renderer.upload_buffer(to_u8_slice(&verts));

        return Scene
        {
            verts_pos: verts_pos_buf,
            indices: indices_buf,
            bvh_nodes: bvh_buf,
            verts: verts_buf
        }
    }

    return Scene
    {
        verts_pos: renderer.create_empty_buffer(),
        indices: renderer.create_empty_buffer(),
        bvh_nodes: renderer.create_empty_buffer(),
        verts: renderer.create_empty_buffer()
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
// @performance This can be greatly improved, this first implementation
// is just a simple working one, but this could probably be easily SIMD-ized
// and parallelized.
pub fn build_bvh(verts: &[f32], indices: &mut[u32])->Vec<BvhNode>
{
    let num_tris: u32 = indices.len() as u32 / 3;

    // Auxiliary arrays for speed
    let mut centroids: Vec<Vec3> = Vec::default();
    centroids.reserve_exact(num_tris as usize);
    let mut tri_bounds: Vec<Aabb> = Vec::default();
    tri_bounds.reserve_exact(num_tris as usize);

    // Build auxiliary arrays
    for tri in 0..num_tris
    {
        let (t0, t1, t2) = get_tri(verts, indices, tri as usize);

        let centroid = compute_tri_centroid(t0, t1, t2);
        let bounds = compute_tri_bounds(t0, t1, t2);
        centroids.push(centroid);
        tri_bounds.push(bounds);
    }

    let (aabb_min, aabb_max) = compute_aabb(indices, tri_bounds.as_mut_slice(), 0, num_tris);
    let bvh_root = BvhNode
    {
        aabb_min,
        aabb_max,
        tri_begin_or_first_child: 0,  // Set tri_begin to 0
        tri_count: num_tris
    };

    let mut bvh: Vec<BvhNode> = Vec::new();
    bvh.reserve(1);
    bvh.push(bvh_root);
    // NOTE: Push a fictitious element so that
    // both of a node's children will reside
    // in the same cache line
    bvh.push(Default::default());

    const BVH_MAX_DEPTH: i32 = 20;
    bvh_split(&mut bvh, verts, indices, centroids.as_mut_slice(), tri_bounds.as_mut_slice(), 0, BVH_MAX_DEPTH, 0);

    return bvh;
}

// This could be turned into an iterative function with some
// performance gains
pub fn bvh_split(bvh: &mut Vec<BvhNode>, verts: &[f32],
                 indices: &mut[u32],
                 centroids: &mut[Vec3],
                 tri_bounds: &mut[Aabb],
                 node: usize, max_depth: i32, depth: i32)
{
    if depth == max_depth { return; }

    let split = choose_split(bvh.as_slice(), verts, indices, centroids, tri_bounds, node);
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
        let centroid = centroids[tri as usize];
        if centroid[split.axis] <= split.pos
        {
            if tri != tri_left_idx
            {
                // Swap triangle of index tri_left_idx with
                // triangle of index tri_idx
                swap_tris(indices, centroids, tri_bounds, tri_left_idx, tri);
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
    // as a non-leaf by the shader, which would be a problem)
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

    // If we're at a certain depth, instead of recursing like this, start one thread for
    // each split.
    let num_nodes_prev_depth: i32 = if depth == 0 {0} else {2_i32.pow((depth - 1) as u32)};
    let num_nodes_current_depth = 2_i32.pow(depth as u32);
    let num_nodes_next_depth: i32 = 2_i32.pow(depth as u32 + 1);

    //static mut count: i32 = 0;
    //if depth == 3 { println!("count: {}\ndepth: {}", unsafe { count }, depth); unsafe { count = 0; } }
    //if depth > 3 { unsafe { count += 1; } }

    bvh_split(bvh, verts, indices, centroids, tri_bounds, left, max_depth, depth + 1);
    bvh_split(bvh, verts, indices, centroids, tri_bounds, right, max_depth, depth + 1);
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

// Binned BVH building.
// For more info, see: https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
pub struct Bin
{
    bounds: Aabb,
    tri_count: u32
}

impl Default for Bin
{
    fn default()->Bin
    {
        return Bin
        {
            bounds: Aabb::neutral(),
            tri_count: 0,
        }
    }
}

// From: https://jacco.ompf2.com/2022/04/21/how-to-build-a-bvh-part-3-quick-builds/
pub fn choose_split(bvh: &[BvhNode], verts: &[f32],
                    indices: &mut[u32],
                    centroids: &mut[Vec3],
                    tri_bounds: &mut[Aabb],
                    node: usize)->BvhSplit
{
    const NUM_BINS: usize = 5;

    let size = bvh[node].aabb_max - bvh[node].aabb_min;
    let tri_begin: usize = bvh[node].tri_begin_or_first_child as usize;
    let tri_count: usize = bvh[node].tri_count as usize;

    // Initialize the cost as the cost of the parent
    // node by itself. Only split if the cost is actually lower
    // after the split.
    let mut res: BvhSplit = Default::default();
    res.cost = node_cost(size, tri_count as u32);

    for axis in 0..3usize
    {
        // Compute centroid bounds because it slightly
        // improves the quality of the resulting tree
        // with little additional building cost
        let mut centroid_min = f32::MAX;
        let mut centroid_max = f32::MIN;
        for tri in tri_begin..tri_begin+tri_count
        {
            let centroid: Vec3 = centroids[tri];
            centroid_min = centroid_min.min(centroid[axis]);
            centroid_max = centroid_max.max(centroid[axis]);
        }

        // Can't split anything here...
        if centroid_min == centroid_max { continue; }

        // Construct bins, for faster cost computation
        let mut bins: [Bin; NUM_BINS] = Default::default();

        // Populate the bins
        let scale = NUM_BINS as f32 / (centroid_max - centroid_min);  // To avoid repeated division
        for tri in tri_begin..tri_begin+tri_count
        {
            let centroid = centroids[tri];
            let bounds = tri_bounds[tri];
            let bin_idx: usize = (NUM_BINS - 1).min(((centroid[axis] - centroid_min) * scale) as usize);
            grow_aabb_to_include_aabb(&mut bins[bin_idx].bounds, bounds); 
            bins[bin_idx].tri_count += 1;
        }

        // Gather data for the N-1 planes between the N bins
        let mut left_aabbs:  [Aabb; NUM_BINS - 1] = Default::default();
        let mut right_aabbs: [Aabb; NUM_BINS - 1] = Default::default();
        let mut left_count:  [u32; NUM_BINS - 1] = Default::default();
        let mut right_count: [u32; NUM_BINS - 1] = Default::default();
        let mut left_aabb: Aabb = Aabb::neutral();
        let mut right_aabb: Aabb = Aabb::neutral();
        let mut left_sum = 0;
        let mut right_sum = 0;
        for i in 0..NUM_BINS - 1
        {
            left_sum += bins[i].tri_count;
            left_count[i] = left_sum;
            grow_aabb_to_include_aabb(&mut left_aabb, bins[i].bounds);
            left_aabbs[i] = left_aabb;

            right_sum += bins[NUM_BINS - 1 - i].tri_count;
            right_count[NUM_BINS - 2 - i] = right_sum;
            grow_aabb_to_include_aabb(&mut right_aabb, bins[NUM_BINS - 1 - i].bounds);
            right_aabbs[NUM_BINS - 2 - i] = right_aabb;
        }

        // Calculate the SAH cost for the N-1 planes
        let scale: f32 = (centroid_max - centroid_min) / NUM_BINS as f32;
        for i in 0..NUM_BINS - 1
        {
            let left_size = left_aabbs[i].max - left_aabbs[i].min;
            let right_size = right_aabbs[i].max - right_aabbs[i].min;
            let plane_cost = node_cost(left_size, left_count[i]) + node_cost(right_size, right_count[i]);
            if plane_cost < res.cost
            {
                res.performed_split = true;
                res.cost = plane_cost;
                res.axis = axis;
                res.pos = centroid_min + scale * (i + 1) as f32;
                res.aabb_left_min = left_aabbs[i].min;
                res.aabb_right_min = right_aabbs[i].min;
                res.aabb_left_max = left_aabbs[i].max;
                res.aabb_right_max = right_aabbs[i].max;
                res.num_tris_left = left_count[i];
                res.num_tris_right = right_count[i];
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

//pub fn get_split_bounds_and_tri_count(bins: &[Bin], )->

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

pub fn swap_tris(indices: &mut[u32], centroids: &mut[Vec3], tri_bounds: &mut[Aabb], tri_a: u32, tri_b: u32)
{
    let tri_a = tri_a as usize;
    let tri_b = tri_b as usize;
    let idx0 = indices[tri_a*3+0];
    let idx1 = indices[tri_a*3+1];
    let idx2 = indices[tri_a*3+2];

    // Swap indices
    indices[tri_a*3 + 0] = indices[tri_b*3 + 0];
    indices[tri_a*3 + 1] = indices[tri_b*3 + 1];
    indices[tri_a*3 + 2] = indices[tri_b*3 + 2];
    indices[tri_b*3 + 0] = idx0 as u32;
    indices[tri_b*3 + 1] = idx1 as u32;
    indices[tri_b*3 + 2] = idx2 as u32;

    // Swap centroids
    let tmp = centroids[tri_a];
    centroids[tri_a] = centroids[tri_b];
    centroids[tri_b] = tmp;

    // Swap tri bounds
    let tmp = tri_bounds[tri_a];
    tri_bounds[tri_a] = tri_bounds[tri_b];
    tri_bounds[tri_b] = tmp;
}

pub fn compute_aabb(indices: &[u32], tri_bounds: &mut[Aabb], tri_begin: u32, tri_count: u32)->(Vec3, Vec3)
{
    let mut res = Aabb::default();
    let tri_end = tri_begin + tri_count;
    for tri in tri_begin..tri_end
    {
        let bounds = tri_bounds[tri as usize];
        grow_aabb_to_include_aabb(&mut res, bounds);
    }

    return (res.min, res.max);
}
