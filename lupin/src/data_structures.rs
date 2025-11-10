
use crate::base::*;
use crate::wgpu_utils::*;
use crate::renderer::*;

#[derive(Default, Debug)]
pub struct EnvMapInfo
{
    pub data: Vec::<Vec4>,
    pub width: u32,
    pub height: u32,
}
pub fn build_lights(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneCPU, envs_info: &[EnvMapInfo]) -> LightsCPU
{
    let environments = &scene.environments;
    let instances = &scene.instances;
    let verts_pos = &scene.verts_pos_array;
    let indices = &scene.indices_array;
    let materials = &scene.materials;
    assert!(environments.len() == envs_info.len(), "Mismatching sizes for environment data!");

    let mut lights = Vec::<Light>::new();
    let mut alias_tables = Vec::<Vec::<AliasBin>>::new();
    let mut env_alias_tables = Vec::<Vec::<AliasBin>>::new();
    for (i, instance) in instances.iter().enumerate()
    {
        let mat = materials[instance.mat_idx as usize];
        let mesh_verts = &verts_pos[instance.mesh_idx as usize];
        let mesh_indices = &indices[instance.mesh_idx as usize];
        if mat.emission.is_zero() { continue; }
        if mesh_indices.is_empty() { continue; }

        let mut total_area: f32 = 0.0;
        let mut weights = Vec::<f32>::new();
        for i in (0..mesh_indices.len()).step_by(3)
        {
            let v0 = mesh_verts[mesh_indices[i+0] as usize];
            let v1 = mesh_verts[mesh_indices[i+1] as usize];
            let v2 = mesh_verts[mesh_indices[i+2] as usize];

            let area = tri_area(v0, v1, v2);
            weights.push(area);
            total_area += area;
        }

        // Guard against the extreme edgecase where all triangles
        // have 0 area.
        if total_area <= 0.0 { continue; }

        let light = Light { instance_idx: i as u32, area: total_area };
        lights.push(light);

        let alias_table = build_alias_table(&weights);
        assert!(alias_table.len() > 0);  // WGPU doesn't like 0 size buffers anyway
        alias_tables.push(alias_table);
    }

    for i in 0..environments.len()
    {
        let mut weights = Vec::<f32>::new();

        let scale = environments[i].emission;
        let env_tex = &envs_info[i].data;
        let env_width = envs_info[i].width;
        let env_height = envs_info[i].height;
        assert!(env_tex.len() == (env_width * env_height) as usize);

        // Environments have the sole purpose of emitting light,
        // so an environment with 0.0 emission should not exist, thus
        // there's a 1:1 mapping of environments and environment lights/alias tables.

        for (i, pixel) in env_tex.iter().enumerate()
        {
            let y = i / env_width as usize;
            let angle = (y as f32 + 0.5) * std::f32::consts::PI / env_height as f32;
            let pixel_emission = f32::max(f32::max(pixel.x * scale.x, pixel.y * scale.y), pixel.z * scale.z);
            let prob = pixel_emission * f32::sin(angle);

            // Use uniform weights in case of 0.0 emission (this
            // shouldn't happen but it's handled correctly regardless).
            if scale.x <= 0.0 && scale.y <= 0.0 && scale.z <= 0.0 {
                weights.push(1.0);
            } else {
                weights.push(prob);
            }
        }

        let alias_table = build_alias_table(&weights);
        assert!(alias_table.len() > 0);  // WGPU doesn't like 0 size buffers anyway
        env_alias_tables.push(alias_table);
    }

    return LightsCPU {
        lights: lights,
        alias_tables,
        env_alias_tables,
    };

    fn tri_area(p0: Vec4, p1: Vec4, p2: Vec4) -> f32
    {
        let p0_v3 = Vec3 { x: p0.x, y: p0.y, z: p0.z };
        let p1_v3 = Vec3 { x: p1.x, y: p1.y, z: p1.z };
        let p2_v3 = Vec3 { x: p2.x, y: p2.y, z: p2.z };
        return length_vec3(cross_vec3(p1_v3 - p0_v3, p2_v3 - p0_v3)) / 2.0;
    }
}

// From: https://www.pbr-book.org/4ed/Sampling_Algorithms/The_Alias_Method#AliasTable
pub fn build_alias_table(weights: &[f32]) -> Vec<AliasBin>
{
    if weights.is_empty() { return vec![]; }

    let mut bins = vec![AliasBin::default(); weights.len()];

    let mut sum: f64 = 0.0;
    for weight in weights {
        sum += *weight as f64;
    }

    if sum == 0.0 { return vec![]; }

    // Normalize weights.
    let normalize_factor = 1.0 / sum;  // * is faster than /
    for (i, weight) in weights.iter().enumerate() {
        bins[i].prob = (*weight as f64 * normalize_factor) as f32;
    }

    // Work lists.
    #[derive(Default, Debug)]
    struct Outcome
    {
        prob_estimate: f32,
        idx: u32,
    }
    let mut under = Vec::<Outcome>::new();
    let mut over  = Vec::<Outcome>::new();
    for (i, bin) in bins.iter().enumerate()
    {
        let prob_estimate = bin.prob * bins.len() as f32;
        let new_outcome = Outcome { prob_estimate, idx: i as u32 };
        if prob_estimate < 1.0 {
            under.push(new_outcome);
        } else {
            over.push(new_outcome);
        }
    }

    while !under.is_empty() && !over.is_empty()
    {
        let under_item = under.pop().unwrap();
        let over_item  = over.pop().unwrap();

        // Initialize state for under_item.
        bins[under_item.idx as usize].alias_threshold = under_item.prob_estimate;
        bins[under_item.idx as usize].alias = over_item.idx;

        // Push excess probability onto work list.
        let prob_excess = under_item.prob_estimate + over_item.prob_estimate - 1.0;
        let new_outcome = Outcome { prob_estimate: prob_excess, idx: over_item.idx };
        if prob_excess < 1.0 {
            under.push(new_outcome);
        } else {
            over.push(new_outcome);
        }
    }

    // Handle remaining work list items.
    // Due to floating point precision, there may be remaining
    // items if their normalized probability is very close to 1.
    while !over.is_empty()
    {
        let over_item = over.pop().unwrap();
        //assert!(f32::abs(over_item.prob_estimate - 1.0) < 0.05);
        bins[over_item.idx as usize].alias_threshold = 1.0;
        bins[over_item.idx as usize].alias = 0;
    }
    while !under.is_empty()
    {
        let under_item = under.pop().unwrap();
        //assert!(f32::abs(under_item.prob_estimate - 1.0) < 0.05);
        bins[under_item.idx as usize].alias_threshold = 1.0;
        bins[under_item.idx as usize].alias = 0;
    }

    return bins;
}

// NOTE: This modifies the indices array to change the order of triangles (indices)
pub fn build_bvh(verts: &[Vec4], indices: &mut[u32]) -> Vec<BvhNode>
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

    bvh_split(&mut bvh, verts, indices, centroids.as_mut_slice(), tri_bounds.as_mut_slice(), 0);
    //bvh_split_bfs(&mut bvh, verts, indices, centroids.as_mut_slice(), tri_bounds.as_mut_slice(), 0, 1);

    return bvh;
}

pub fn bvh_split(bvh: &mut Vec<BvhNode>, verts: &[Vec4],
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

#[derive(Default)]
pub struct BvhSplit
{
    performed_split: bool,
    axis: isize,
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
pub fn choose_split(bvh: &[BvhNode], _verts: &[Vec4],
                    _indices: &mut[u32],
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

    for axis in 0..3isize
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

        // TODO: Do we really need this or do we have a bug in this function?
        const EPS: f32 = 0.001;
        centroid_min -= EPS;
        centroid_max += EPS;

        // Construct bins, for faster cost computation
        let mut bins: [Bin; NUM_BINS] = Default::default();

        // Populate the bins
        let scale = NUM_BINS as f32 / (centroid_max - centroid_min);  // To avoid repeated division
        for tri in tri_begin..tri_begin+tri_count
        {
            let centroid = centroids[tri];
            let bounds = tri_bounds[tri];
            let bin_idx: usize = (((centroid[axis] - centroid_min) * scale).floor() as usize).clamp(0, NUM_BINS-1);
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
    let half_area: f32 = size.x * (size.y + size.z) + size.y * size.z;
    return half_area * num_tris as f32;
}

fn get_tri(verts: &[Vec4], indices: &[u32], tri_idx: usize)->(Vec3, Vec3, Vec3)
{
    let idx0 = (indices[tri_idx*3+0]) as usize;
    let idx1 = (indices[tri_idx*3+1]) as usize;
    let idx2 = (indices[tri_idx*3+2]) as usize;

    let t0 = Vec3 {
        x: verts[idx0].x,
        y: verts[idx0].y,
        z: verts[idx0].z,
    };
    let t1 = Vec3 {
        x: verts[idx1].x,
        y: verts[idx1].y,
        z: verts[idx1].z,
    };
    let t2 = Vec3 {
        x: verts[idx2].x,
        y: verts[idx2].y,
        z: verts[idx2].z,
    };

    return (t0, t1, t2);
}

fn swap_tris(indices: &mut[u32], centroids: &mut[Vec3], tri_bounds: &mut[Aabb], tri_a: u32, tri_b: u32)
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

fn compute_aabb(_indices: &[u32], tri_bounds: &mut[Aabb], tri_begin: u32, tri_count: u32)->(Vec3, Vec3)
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

// The aabbs are in model space, and they're indexed using the instance mesh_idx member.
pub fn build_tlas(instances: &[Instance], model_aabbs: &[Aabb]) -> Vec<TlasNode>
{
    if instances.is_empty() || model_aabbs.is_empty() { return vec![]; }

    let mut node_indices = Vec::<u32>::with_capacity(instances.len());
    let mut tlas = Vec::<TlasNode>::with_capacity(instances.len() * 2 - 1);

    // Assign a leaf node for each instance.
    for i in 0..instances.len() as u32
    {
        let instance = instances[i as usize];
        let model_aabb = model_aabbs[instance.mesh_idx as usize];
        let transform = instance.transpose_inverse_transform.transpose().inverse();
        let aabb_trans = transform_aabb(model_aabb.min, model_aabb.max, transform);

        let tlas_node = TlasNode {
            aabb_min: aabb_trans.min,
            aabb_max: aabb_trans.max,
            instance_idx: i,
            left: 0,  // Makes it a leaf.
            right: 0,
            ..Default::default()
        };
        tlas.push(tlas_node);
        node_indices.push(tlas.len() as u32 - 1);
    }

    // Use agglomerative clustering.
    let mut a: u32 = 0;
    let mut b = tlas_find_best_match(tlas.as_slice(), node_indices.as_slice(), a);
    while node_indices.len() > 1
    {
        let c = tlas_find_best_match(tlas.as_slice(), node_indices.as_slice(), b);
        if a == c
        {
            let node_idx_a = node_indices[a as usize];
            let node_idx_b = node_indices[b as usize];

            let node_a = tlas[node_idx_a as usize];
            let node_b = tlas[node_idx_b as usize];

            let new_node = TlasNode {
                left: node_idx_a,
                right: node_idx_b,
                aabb_min: node_a.aabb_min.min(node_b.aabb_min),
                aabb_max: node_a.aabb_max.max(node_b.aabb_max),
                instance_idx: 0,  // Unused.
                ..Default::default()
            };
            tlas.push(new_node);

            assert!(tlas.len() <= u32::MAX as usize);

            node_indices[a as usize] = tlas.len() as u32 - 1;
            node_indices[b as usize] = node_indices[node_indices.len() - 1];
            node_indices.pop();
            if a >= node_indices.len() as u32 { a = node_indices.len() as u32 - 1; }

            b = tlas_find_best_match(tlas.as_slice(), node_indices.as_slice(), a);
        }
        else
        {
            a = b;
            b = c;
        }
    }

    // Push root
    tlas.push(tlas[node_indices[a as usize] as usize]);

    // Reverse TLAS for better memory layout. This puts
    // the root at index 0.
    let tlas_len = tlas.len();
    for i in 0..tlas_len / 2 + tlas_len % 2
    {
        let tmp = tlas[i];
        tlas[i] = tlas[tlas_len - 1 - i];
        tlas[tlas_len - 1 - i] = tmp;

        if tlas[i].left != 0
        {
            tlas[i].left = tlas_len as u32 - 1 - tlas[i].left;
            tlas[i].right = tlas_len as u32 - 1 - tlas[i].right;
        }

        if tlas[tlas_len - 1 - i].left != 0
        {
            tlas[tlas_len - 1 - i].left = tlas_len as u32 - 1 - tlas[i].left;
            tlas[tlas_len - 1 - i].right = tlas_len as u32 - 1 - tlas[i].right;
        }
    }

    return tlas;
}

pub fn tlas_find_best_match(tlas: &[TlasNode], idx_array: &[u32], node_a: u32) -> u32
{
    let a_idx = idx_array[node_a as usize];

    let mut smallest: f32 = f32::MAX;
    let mut best_b: u32 = u32::MAX;
    for (i, &b_idx) in idx_array.iter().enumerate()
    {
        if node_a == i as u32 { continue; }

        let bmax = tlas[a_idx as usize].aabb_max.max(tlas[b_idx as usize].aabb_max);
        let bmin = tlas[a_idx as usize].aabb_min.min(tlas[b_idx as usize].aabb_min);
        let e = bmax - bmin;
        let area = e.x * e.y + e.y * e.z + e.z * e.x;

        if area < smallest {
            smallest = area;
            best_b = i as u32;
        }
    }

    return best_b;
}

/// Also builds RT acceleration structures, if requested and supported.
pub fn upload_scene_to_gpu(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneCPU, textures: Vec<wgpu::Texture>, samplers: Vec<wgpu::Sampler>, envs_info: &[EnvMapInfo], build_rt_structures: bool) -> Scene
{
    let verts_normal_array: Vec::<wgpu::Buffer> = scene.verts_normal_array.iter().map(|x| upload_storage_buffer(device, queue, to_u8_slice(x))).collect();
    let verts_texcoord_array: Vec::<wgpu::Buffer> = scene.verts_texcoord_array.iter().map(|x| upload_storage_buffer(device, queue, to_u8_slice(x))).collect();
    let verts_color_array: Vec::<wgpu::Buffer> = scene.verts_color_array.iter().map(|x| upload_storage_buffer(device, queue, to_u8_slice(x))).collect();
    let verts_pos_array: Vec::<wgpu::Buffer> = scene.verts_pos_array.iter().map(|x| upload_vertex_pos_buffer(device, queue, to_u8_slice(x))).collect();
    let indices_array: Vec::<wgpu::Buffer> = scene.indices_array.iter().map(|x| upload_indices_buffer(device, queue, to_u8_slice(x))).collect();
    let bvh_nodes_array: Vec::<wgpu::Buffer> = scene.bvh_nodes_array.iter().map(|x| upload_storage_buffer(device, queue, to_u8_slice(x))).collect();

    let mesh_infos = upload_storage_buffer_with_name(device, queue, to_u8_slice(&scene.mesh_infos), "mesh_infos");
    let tlas_nodes = upload_storage_buffer_with_name(device, queue, to_u8_slice(&scene.tlas_nodes), "tlas_nodes");
    let instances = upload_storage_buffer_with_name(device, queue, to_u8_slice(&scene.instances), "instances");
    let materials = upload_storage_buffer_with_name(device, queue, to_u8_slice(&scene.materials), "materials");
    let environments = upload_storage_buffer_with_name(device, queue, to_u8_slice(&scene.environments), "environments");

    // Build auxiliary data structures.
    let lights = build_lights(device, queue, scene, envs_info);

    let (rt_tlas, rt_blases) = if build_rt_structures {
        build_rt_accel_structures(device, queue, scene, &verts_pos_array, &indices_array)
    } else {
        (None, vec![])
    };

    let alias_tables_gpu: Vec::<wgpu::Buffer> = scene.lights.alias_tables.iter().map(|x| upload_storage_buffer(device, queue, to_u8_slice(x))).collect();
    let env_alias_tables_gpu: Vec::<wgpu::Buffer> = scene.lights.env_alias_tables.iter().map(|x| upload_storage_buffer(device, queue, to_u8_slice(x))).collect();

    let lights = Lights {
        lights: upload_storage_buffer_with_name(device, queue, to_u8_slice(&scene.mesh_infos), "lights"),
        alias_tables: alias_tables_gpu,
        env_alias_tables: env_alias_tables_gpu
    };

    return Scene {
        mesh_infos,
        verts_pos_array,
        verts_normal_array,
        verts_texcoord_array,
        verts_color_array,
        indices_array,
        bvh_nodes_array,
        tlas_nodes,
        instances,
        materials,
        textures,
        samplers,
        environments,
        lights,
        rt_tlas,
        rt_blases,
    };
}

/// Used to verify if a scene has been built correctly. It's best to call it
/// right before upload_scene_to_gpu.
pub fn validate_scene(scene: &SceneCPU, num_textures: u32, num_samplers: u32)
{
    // TODO: Put readable messages on all of them.
    assert_eq!(scene.verts_pos_array.len(), scene.bvh_nodes_array.len());
    assert_eq!(scene.verts_pos_array.len(), scene.mesh_infos.len());
    assert_eq!(scene.verts_pos_array.len(), scene.mesh_aabbs.len(), "verts_pos_array.len() doesn't match mesh_aabbs.len()");

    for (i, info) in scene.mesh_infos.iter().enumerate()
    {
        if info.normals_buf_idx != SENTINEL_IDX
        {
            assert!((info.normals_buf_idx as usize) < scene.verts_normal_array.len());
            assert_eq!(scene.verts_normal_array[info.normals_buf_idx as usize].len(), scene.verts_pos_array[i].len());
        }
        if info.texcoords_buf_idx != SENTINEL_IDX
        {
            assert!((info.texcoords_buf_idx as usize) < scene.verts_texcoord_array.len());
            assert_eq!(scene.verts_texcoord_array[info.texcoords_buf_idx as usize].len(), scene.verts_pos_array[i].len());
        }
        if info.colors_buf_idx != SENTINEL_IDX
        {
            assert!((info.colors_buf_idx as usize) < scene.verts_color_array.len());
            assert_eq!(scene.verts_color_array[info.colors_buf_idx as usize].len(), scene.verts_pos_array[i].len());
        }
    }

    for (i, indices) in scene.indices_array.iter().enumerate()
    {
        for idx in indices
        {
            assert!((*idx as usize) < scene.verts_pos_array[i].len());
        }
    }

    for tlas_node in &scene.tlas_nodes {
        assert!((tlas_node.instance_idx as usize) < scene.instances.len());
    }
    for instance in &scene.instances
    {
        assert!((instance.mesh_idx as usize) < scene.mesh_infos.len());
        assert!((instance.mat_idx  as usize) < scene.materials.len());
    }

    // TODO: Fill in other stuff in material
    for mat in &scene.materials
    {
        assert!(mat.color_tex_idx < num_textures      || mat.color_tex_idx == SENTINEL_IDX);
        assert!(mat.emission_tex_idx < num_textures   || mat.emission_tex_idx == SENTINEL_IDX);
        assert!(mat.roughness_tex_idx < num_textures  || mat.roughness_tex_idx == SENTINEL_IDX);
        assert!(mat.scattering_tex_idx < num_textures || mat.scattering_tex_idx == SENTINEL_IDX);
        assert!(mat.normal_tex_idx < num_textures     || mat.normal_tex_idx == SENTINEL_IDX);
    }

    for env in &scene.environments
    {
        assert!(env.emission.x >= 0.0 && env.emission.y >= 0.0 && env.emission.z >= 0.0);
        assert!(env.emission_tex_idx < num_textures || env.emission_tex_idx == SENTINEL_IDX);
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct SceneStats
{
    /// Does not count multiple instances of the same mesh
    pub total_tri_count: u64,
    pub num_instances: u64,
}
pub fn get_scene_stats(scene: &Scene) -> SceneStats
{
    return Default::default();
}

pub fn compute_mesh_aabb(verts_pos: &[Vec4]) -> Aabb
{
    let mut aabb = Aabb::neutral();
    for pos in verts_pos {
        grow_aabb_to_include_vert(&mut aabb, Vec3 { x: pos.x, y: pos.y, z: pos.z });
    }
    return aabb;
}

#[cfg(test)]
mod tests
{
    use rand::Rng;
    use super::*;

    #[test]
    fn test_alias_table()
    {
        let weights = [
            0.2, 0.1, 0.05, 0.8, 1.2, 5.0, 0.1, 0.2, 0.3, 1.0, 1.0, 0.3, 0.35, 0.0,
        ];
        test_alias_table_any(&weights);
    }

    #[test]
    fn test_alias_table_white()
    {
        let weights = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ];
        test_alias_table_any(&weights);
    }

    #[test]
    fn test_alias_table_single()
    {
        let weights = [ 1.0 ];
        test_alias_table_any(&weights);
    }

    fn test_alias_table_any(weights: &[f32])
    {
        let mut weights_sum = 0.0;
        for weight in weights {
            weights_sum += weight;
        }

        let alias_table = build_alias_table(&weights);
        assert!(alias_table.len() == weights.len());

        for (i, bin) in alias_table.iter().enumerate() {
            assert!(f32::abs(bin.prob - weights[i] / weights_sum) < 0.01);
        }

        // Check that the alias probabilities are
        // correct with a lot of random samples.

        #[derive(Default, Clone, Copy)]
        struct Record
        {
            pub idx: u32,
            pub hits: u32,
        }

        let mut hits_per_idx = vec![Record::default(); weights.len()];
        for (i, v) in hits_per_idx.iter_mut().enumerate() {
            v.idx = i as u32;
        }

        let mut rng = rand::thread_rng();
        const NUM_SAMPLES: u32 = 100000;
        for i in 0..NUM_SAMPLES
        {
            let slot_idx: usize = rng.gen_range(0..alias_table.len()); // inclusive range
            let rnd: f32 = rng.gen();
            if rnd >= alias_table[slot_idx].alias_threshold {
                hits_per_idx[alias_table[slot_idx].alias as usize].hits += 1;
            } else {
                hits_per_idx[slot_idx as usize].hits += 1;
            }
        }

        for (i, v) in hits_per_idx.iter().enumerate()
        {
            let ratio = v.hits as f32 / NUM_SAMPLES as f32;
            let prob = weights[i] / weights_sum;
            assert!(f32::abs(ratio - prob) < 0.01);
        }
    }
}

fn build_rt_accel_structures(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneCPU,
                             verts_pos_array: &Vec<wgpu::Buffer>, indices_array: &Vec<wgpu::Buffer>) -> (Option<wgpu::Tlas>, Vec<wgpu::Blas>)
{
    if !supports_rt(device) { return (None, vec![]) }

    let mut blases = Vec::new();

    const RT_MASK_DEFAULT: u8 = 1;
    const RT_MASK_LIGHT: u8 = 2;
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: None,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1000000,
    });

    let mut build_entries = Vec::new();
    let mut size_descs = Vec::new();

    for i in 0..scene.verts_pos_array.len()
    {
        let verts = &scene.verts_pos_array[i];
        let indices = &scene.indices_array[i];

        let size_desc = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: verts.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(indices.len() as u32),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        };
        size_descs.push(size_desc);
    }

    for i in 0..scene.verts_pos_array.len()
    {
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: None,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size_descs[i].clone()],
            },
        );

        blases.push(blas);
    }

    for i in 0..scene.verts_pos_array.len()
    {
        let verts = &scene.verts_pos_array[i];
        let indices = &scene.indices_array[i];

        let triangle_geometry = wgpu::BlasTriangleGeometry {
            size: &size_descs[i],
            vertex_buffer: &verts_pos_array[i],
            first_vertex: 0,
            vertex_stride: std::mem::size_of::<Vec4>() as u64,
            index_buffer: Some(&indices_array[i]),
            first_index: Some(0),
            transform_buffer: None,
            transform_buffer_offset: None,
        };

        build_entries.push(wgpu::BlasBuildEntry {
            blas: &blases[i],
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![triangle_geometry]),
        });
    }

    for (i, instance) in scene.instances.iter().enumerate()
    {
        let mesh_idx = instance.mesh_idx;
        let rt_transform = instance.transpose_inverse_transform.transpose().inverse().transpose();
        let rt_transform_serial = [
            rt_transform.m[0][0], rt_transform.m[0][1], rt_transform.m[0][2], rt_transform.m[0][3],
            rt_transform.m[1][0], rt_transform.m[1][1], rt_transform.m[1][2], rt_transform.m[1][3],
            rt_transform.m[2][0], rt_transform.m[2][1], rt_transform.m[2][2], rt_transform.m[2][3],
        ];

        // TODO: This is "O(n^2)", kinda. Not really a problem right now but fix later
        let mut is_light = false;
        let mut light_idx: u32 = 0;
        for (j, light) in scene.lights.lights.iter().enumerate()
        {
            if light.instance_idx == i as u32
            {
                is_light = true;
                light_idx = j as u32;
            }
        }

        // Using the index_mut trait for wgpu::Tlas.
        tlas[i] = Some(wgpu::TlasInstance::new(
            &blases[mesh_idx as usize],
            rt_transform_serial,
            light_idx,
            if is_light { RT_MASK_LIGHT } else { RT_MASK_DEFAULT },
        ));
    }

    // We might run out of memory due to the size of temporary scratch buffers.
    for chunk in build_entries.chunks(10)
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.build_acceleration_structures(chunk.iter(), std::iter::empty());
        queue.submit(Some(encoder.finish()));
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.build_acceleration_structures(std::iter::empty(), std::iter::once(&tlas));
    queue.submit(Some(encoder.finish()));

    return (Some(tlas), blases);
}
