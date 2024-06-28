
use crate::base::*;

use tobj::*;
use std::ptr::NonNull;
use crate::renderer::*;

#[derive(Default)]
pub struct LoadingTimes
{
    pub parsing: f32,
    pub bvh_build: f32
}

pub fn load_scene_custom_format(path: &str)
{

}

pub fn load_scene_obj(path: &str, renderer: &mut Renderer)->(Scene, LoadingTimes)
{
    let mut loading_times: LoadingTimes = Default::default();

    let timer_start = std::time::Instant::now();
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;

    loading_times.parsing = time;

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

        let timer_start = std::time::Instant::now();
        let bvh = build_bvh(verts_pos.as_slice(), &mut mesh.indices);
        let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;
        loading_times.bvh_build = time;

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
                normal = normalize_vec3(normal);
            };

            let mut tex_coords = Vec2::default();
            if mesh.texcoords.len() > 0
            {
                tex_coords.x = mesh.texcoords[vert_idx*2+0];
                tex_coords.y = mesh.texcoords[vert_idx*2+1];
                tex_coords = normalize_vec2(tex_coords);
            };

            let vert = Vertex { normal, padding0: 0.0, tex_coords, padding1: 0.0, padding2: 0.0 };

            verts.push(vert);
        }

        let verts_buf = renderer.upload_buffer(to_u8_slice(&verts));

        return (Scene
        {
            verts_pos: verts_pos_buf,
            indices: indices_buf,
            bvh_nodes: bvh_buf,
            verts: verts_buf
        }, loading_times);
    }

    return (Scene
    {
        verts_pos: renderer.create_empty_buffer(),
        indices: renderer.create_empty_buffer(),
        bvh_nodes: renderer.create_empty_buffer(),
        verts: renderer.create_empty_buffer()
    }, loading_times);
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

    // Using a heuristic for preallocation size slightly speeds up bvh construction
    let initial_bvh_size: usize = (num_tris as usize).min((2_usize.pow(BVH_MAX_DEPTH as u32 + 1)-1) / 8);
    let mut bvh: Vec<BvhNode> = Vec::with_capacity(initial_bvh_size);
    bvh.push(bvh_root);

    //bvh_split(&mut bvh, verts, indices, centroids.as_mut_slice(), tri_bounds.as_mut_slice(), 0);
    bvh_split_bfs(&mut bvh, verts, indices, centroids.as_mut_slice(), tri_bounds.as_mut_slice(), 0, 1);

    return bvh;
}

// This iteratively splits the root node and its children until
// the maximum allowed depth is reached
pub fn bvh_split(bvh: &mut Vec<BvhNode>, verts: &[f32],
                 indices: &mut[u32],
                 centroids: &mut[Vec3],
                 tri_bounds: &mut[Aabb],
                 node: usize)
{
    #[derive(Default)]
    struct StackInfo
    {
        node: u32,
        depth: u32
    }

    let mut stack: [StackInfo; BVH_MAX_DEPTH as usize] = Default::default();
    let mut stack_idx: usize = 1;
    stack[0] = StackInfo { node: node as u32, depth: 1 };

    while stack_idx > 0
    {
        stack_idx -= 1;
        let node  = stack[stack_idx].node as usize;
        let depth = stack[stack_idx].depth as usize;

        let split = choose_split(bvh.as_slice(), verts, indices, centroids, tri_bounds, node);
        if !split.performed_split { continue; }

        let cur_tri_begin = bvh[node].tri_begin_or_first_child;
        let cur_tri_count = bvh[node].tri_count;
        let cur_tri_end   = cur_tri_begin + cur_tri_count;

        // For each child, sort the indices so that they're contiguous
        // and the tri set can be referenced with only tri_begin and tri_count

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
        if left_count == 0 || right_count == 0 { continue; }

        // We've decided to split
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

        if depth < (BVH_MAX_DEPTH - 1) as usize
        {
            stack[stack_idx + 0] = StackInfo { node: left as u32,  depth: depth as u32 + 1 };
            stack[stack_idx + 1] = StackInfo { node: right as u32, depth: depth as u32 + 1 };
            stack_idx += 2;
        }
    }
}

// This splits all nodes in a breadth-first manner. This
// takes up more memory compared to a depth-first bvh creation,
// but the problem with that is that we get lots of small computations
// which can't easily be simdized. This way the split computation can be
// performed on the entire depth level instead of an individual node
pub fn bvh_split_bfs(bvh: &mut Vec<BvhNode>, verts: &[f32],
                     indices: &mut[u32],
                     centroids: &mut[Vec3],
                     tri_bounds: &mut[Aabb],
                     node: usize,
                     start_depth: i32)
{
    let mut nodes_to_process = Vec::new();
    nodes_to_process.push(node);

    let mut next_nodes = Vec::new();

    let mut depth = start_depth;
    while depth <= BVH_MAX_DEPTH
    {
        // Loop through all nodes to process and then
        // construct the one for the next level
        for i in 0..nodes_to_process.len()
        {
            let node = nodes_to_process[i];

            let split = choose_split(bvh.as_slice(), verts, indices, centroids, tri_bounds, node);
            if !split.performed_split { continue; }

            let cur_tri_begin = bvh[node].tri_begin_or_first_child;
            let cur_tri_count = bvh[node].tri_count;
            let cur_tri_end   = cur_tri_begin + cur_tri_count;

            // For each child, sort the indices so that they're contiguous
            // and the tri set can be referenced with only tri_begin and tri_count

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
            if left_count == 0 || right_count == 0 { continue; }

            // We've reached this point, so we've decided to split

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

            next_nodes.push(left);
            next_nodes.push(right);
        }

        nodes_to_process.clear();        
        std::mem::swap(&mut nodes_to_process, &mut next_nodes);

        depth += 1;
    }
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
            left_count[i] = left_sum + 1;
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

// This implementation uses avx512 intrinsics
pub fn choose_split_simd(bvh: &[BvhNode], verts: &[f32],
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
            // This could be parallelized across iterations of this loop,
            // but also across different axes. The single operations here
            // is a min(a, b) and a max(c, b). So this can be float x 3 (axes)
            // x 5 (iterations) = 16. So this is the equivalent of doing a min in
            // parallel on a vector3, but on many different iterations. I don't know
            // if what i described can actually be done.

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
            // The main operation here is growing the aabb.
            // There is no dependency on the iterations of this
            // loop, so it can be safely parallelized.

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
            // There is a dependency on the iterations here,
            // but these two parts can be done in parallel.
            // Also let's not forget that the axes can all still
            // be calculated in parallel

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
            // No dependency on the iterations here, so we can probably
            // do quite a bit here with avx.

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
