
use crate::base::*;
use crate::wgpu_utils::*;

pub struct SceneDesc
{
    pub verts_pos: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub bvh_nodes: wgpu::Buffer,
    pub verts: wgpu::Buffer

    // Textures

}

// This doesn't include positions, as that
// is stored in a separate buffer for locality
#[repr(C)]
#[derive(Default)]
pub struct Vertex
{
    pub normal: Vec3,
    pub padding0: f32,
    pub tex_coords: Vec2,
    pub padding1: f32,
    pub padding2: f32,
}

// NOTE: The odd ordering of the fields
// ensures that the struct is 32 bytes wide,
// given that vec3f has 16-byte padding (on the GPU)
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct BvhNode
{
    pub aabb_min: Vec3,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    pub tri_begin_or_first_child: u32,
    pub aabb_max: Vec3,
    pub tri_count: u32
}

// Constants
pub const BVH_MAX_DEPTH: i32 = 25;

// This will need to be used when creating the device
pub fn get_device_spec()->wgpu::DeviceDescriptor<'static>
{
    // We currently need these two features:
    // 1) Arrays of texture bindings, to store textures of arbitrary sizes
    // 2) Texture sampling and buffer non uniform indexing, to access textures
    return wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY |
                           wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
        required_limits: wgpu::Limits::default(),
        memory_hints: Default::default(),
    };
}

// Resources
/*
pub struct PathtracePipelineDesc<'a>
{
    //shader_source: &'a str,
}

impl<'a> Default for PathtracePipelineDesc<'a>
{
    fn default() -> Self
    {
        return Self {

        }
    }
}

pub fn create_pathtrace_pipeline(desc: PathtracePipelineDesc) -> Option<wgpu::ComputePipeline>
{
    return None;
}

pub fn create_vertex_buffer() -> Vec<>
{

}
*/

// Rendering
pub struct PathtraceDesc
{
    camera_transform: Mat4,
    // progressive ?

}

impl Default for PathtraceDesc
{
    fn default() -> Self
    {
        return Self {
            camera_transform: Mat4::IDENTITY,
        }
    }
}



pub fn pathtrace_scene(scene: &SceneDesc,
                       device: &wgpu::Device, queue: &wgpu::Queue,
                       cmd_enc: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline,
                       target: &wgpu::Texture,
                       pathtrace_desc: PathtraceDesc)
{
    use wgpu::*;

    // TODO: Remove the TEXTURE_BINDING check if we're not doing progressive rendering.
    assert!(target.usage().contains(TextureUsages::TEXTURE_BINDING) &&
            target.usage().contains(TextureUsages::STORAGE_BINDING),
            "Pathtrace target texture has invalid usage!");

    let mut pass = cmd_enc.begin_compute_pass(&ComputePassDescriptor {
        label: Some("Lupin Pathtrace Compute Pass"),
        timestamp_writes: None
    });

    pass.set_pipeline(pipeline);

    let bindgroup_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            BindGroupLayoutEntry {  // Target Texture
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageTexture {
                    access: StorageTextureAccess::WriteOnly,
                    format: target.format(),
                    view_dimension: TextureViewDimension::D2,
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {  // Verts pos
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {  // Indices
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None
                },
                count: None,
            },
            BindGroupLayoutEntry {  // BVH Nodes
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            },
            BindGroupLayoutEntry {  // Verts pos
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            },
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            },
        ]
    });

    let target_view = target.create_view(&Default::default());

    #[inline(always)]
    fn buffer_resource(buffer: &Buffer) -> BindingResource
    {
        return BindingResource::Buffer(BufferBinding { buffer: buffer, offset: 0, size: None });
    }

    // TODO: Reuse this buffer
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Lupin Pathtracing Uniform Buffer"),
        contents: unsafe { to_u8_slice(&[pathtrace_desc.camera_transform]) },
        usage: BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: None,
        layout: &bindgroup_layout,
        entries: &[
            BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&target_view) },
            BindGroupEntry { binding: 1, resource: buffer_resource(&scene.verts_pos) },
            BindGroupEntry { binding: 2, resource: buffer_resource(&scene.indices) },
            BindGroupEntry { binding: 3, resource: buffer_resource(&scene.bvh_nodes) },
            BindGroupEntry { binding: 4, resource: buffer_resource(&scene.verts) },
            BindGroupEntry { binding: 5, resource: buffer_resource(&uniform_buffer) }
        ]
    });

    pass.set_bind_group(0, &bind_group, &[]);
    const WORKGROUP_SIZE_X: u32 = 8;
    const WORKGROUP_SIZE_Y: u32 = 8;
    let width  = target.width();
    let height = target.height();
    let num_workers_x = (width + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
    let num_workers_y = (height + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
    pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
}

////////
// BVH creation

pub struct TLAS
{
    buf: wgpu::Buffer,
}

//pub fn build_tlas(device: &wgpu::Device, queue::Queue, instances) -> TLAS;

// Maybe do this on the GPU
//pub fn update_tlas(device: &wgpu::Device, queue::Queue, tlas: &mut TLAS, instances)
//pub fn update_tlas_xform(device: &wgpu::Device, queue::Queue, tlas: &mut TLAS, instances_with_xform)

// NOTE: This modifies the indices array to change the order of triangles
// based on the BVH
pub fn build_bvh(device: &wgpu::Device, queue: &wgpu::Queue, verts: &[f32], indices: &mut[u32])->wgpu::Buffer
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

    // Upload the buffer to GPU
    let res = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&bvh) });
    return res;
}

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
pub fn choose_split(bvh: &[BvhNode], _verts: &[f32],
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

pub fn compute_aabb(_indices: &[u32], tri_bounds: &mut[Aabb], tri_begin: u32, tri_count: u32)->(Vec3, Vec3)
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

// Here we want:
// 1 stuff for drawing the scene. This should allow separating this draw call into multiple draw calls,
// so that the gpu can continue doing other work. Also, if the rendering takes really long, it's good
// user interface to let the user cancel the operation. Also, if the rendering takes too long i think vulkan
// will kill the process. NOTE: multi-queues are not implemented in wgpu as of today,
// but when they are, scene rendering can be moved to a separate queue and then we can synchronize as necessary.
// 2 stuff for executing shaders in general. Use the fancy debugging system.
// 3 constructing the bvh should be here, as it is an integral part of the renderer
