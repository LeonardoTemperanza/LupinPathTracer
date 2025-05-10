
use crate::base::*;
use crate::wgpu_utils::*;

pub static DEFAULT_PATHTRACER_SRC: &str = include_str!("shaders/pathtracer.wgsl");
pub static TONEMAPPING_SRC: &str = include_str!("shaders/tonemapping.wgsl");

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Vec2
{
    pub x: f32,
    pub y: f32
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Vec3
{
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Mat4
{
    pub m: [[f32; 4]; 4]
}

pub struct SceneDesc
{
    // It would be more ergonomic to do a AoS here
    // (which could then just be turned into SoA
    pub verts_pos: Vec<wgpu::Buffer>,
    pub verts: Vec<wgpu::Buffer>,
    pub indices: Vec<wgpu::Buffer>,
    pub bvh_nodes: Vec<wgpu::Buffer>,

    pub tlas_nodes: wgpu::Buffer,
    pub instances: wgpu::Buffer,

    pub textures: Vec<wgpu::Texture>,
    pub samplers: Vec<wgpu::Sampler>
}

pub struct Instance
{
    pub pos: Vec3,
    pub mesh_idx: u32,
    pub texture_idx: u32,
    pub sampler_idx: u32,
    pub padding0: f32,
    pub padding1: f32,
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
pub fn get_required_device_spec()->wgpu::DeviceDescriptor<'static>
{
    // The main feature we need is the possibility to define arrays of buffer,
    // texture and sampler bindings, and to index them with non-uniform values.
    return wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY |
                           wgpu::Features::BUFFER_BINDING_ARRAY  |
                           wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY |
                           wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING |
                           wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY,
        required_limits: wgpu::Limits {
            max_storage_buffers_per_shader_stage: 2048,
            max_samplers_per_shader_stage: 16,
            ..Default::default()
        },
        memory_hints: Default::default(),
    };
}

// Shader params

pub struct PathtraceShaderParams
{
    pub shader: wgpu::ShaderModule,
    pub pipeline: wgpu::ComputePipeline,
}

pub fn build_pathtrace_shader_params(device: &wgpu::Device, with_runtime_checks: bool) -> PathtraceShaderParams
{
    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Pathtracer Shader"),
        source: wgpu::ShaderSource::Wgsl(DEFAULT_PATHTRACER_SRC.into())
    };

    let shader;
    if with_runtime_checks {
        shader = device.create_shader_module(shader_desc);
    } else {
        shader = unsafe { device.create_shader_module_trusted(shader_desc, wgpu::ShaderRuntimeChecks::unchecked()) };
    }

    let scene_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: Some("Lupin Pathtracer Bindgroup Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {  // verts_pos
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(1)
            },
            wgpu::BindGroupLayoutEntry {  // verts
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(1)
            },
            wgpu::BindGroupLayoutEntry {  // indices
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(1)
            },
            wgpu::BindGroupLayoutEntry {  // bvh_nodes
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(1)
            },
            wgpu::BindGroupLayoutEntry {  // tlas_nodes
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // instances
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            /*wgpu::BindGroupLayoutEntry {  // textures
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: std::num::NonZero::new(1)
            },
            wgpu::BindGroupLayoutEntry {  // samplers
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: std::num::NonZero::new(1)
            },
            */
        ]
    });

    let settings_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {  // Camera transform
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            },
        ]
    });

    let render_target_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None
            },
        ]
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lupin Pathtracer Pipeline Layout"),
        bind_group_layouts: &[
            &scene_bind_group_layout,
            &settings_bind_group_layout,
            &render_target_bind_group_layout,
        ],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    return PathtraceShaderParams {
        shader,
        pipeline
    };
}

#[cfg(disable)]
pub fn build_pathtrace_shader_params_custom_shader(device: &wgpu::Device, shader_src: &str) -> ShaderParams
{
    return ShaderParams {

    };
}

// Rendering
// Parameters:
// device
// scene
// render_target
// shader_params (shader itself, and things like vertex layout, bvh depth)
// defaultable - accum_params (accum (bool), Option of previous image, frame_counter, when to stop?) (provide a function to advance the accum params)
// defaultable - region (offset and size to render)
// defaultable - motion_blur_params

pub fn pathtrace_scene(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, render_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, camera_transform: Mat4)
{
    // TODO: Check format and usage of render target params and others

    let render_target_view = render_target.create_view(&Default::default());

    use wgpu::util::DeviceExt;
    let camera_transform_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: to_u8_slice(&[camera_transform]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    fn array_of_buffer_bindings_resource(buffers: &Vec<wgpu::Buffer>) -> Vec<wgpu::BufferBinding>
    {
        let mut bindings: Vec<wgpu::BufferBinding> = Vec::with_capacity(buffers.len());
        for i in 0..buffers.len()
        {
            bindings.push(wgpu::BufferBinding {
                buffer: &buffers[i],
                offset: 0,
                size: None
            });
        }

        return bindings;
    }

    fn array_of_texture_views(textures: &Vec<wgpu::Texture>) -> Vec<wgpu::TextureView>
    {
        let mut bindings: Vec<wgpu::TextureView> = Vec::with_capacity(textures.len());
        for i in 0..textures.len()
        {
            let view = textures[i].create_view(&Default::default());
            bindings.push(view);
        }

        return bindings;
    }

    fn array_of_texture_bindings_resource<'a>(texture_views: &'a Vec<wgpu::TextureView>) -> Vec<&'a wgpu::TextureView>
    {
        let mut bindings: Vec<&'a wgpu::TextureView> = Vec::with_capacity(texture_views.len());
        bindings.push(&texture_views[0]);
        for i in 0..texture_views.len() {
            bindings.push(&texture_views[i]);
        }

        return bindings;
    }

    fn array_of_sampler_bindings_resource<'a>(samplers: &'a Vec<wgpu::Sampler>) -> Vec<&'a wgpu::Sampler>
    {
        let mut bindings: Vec<&'a wgpu::Sampler> = Vec::with_capacity(samplers.len());
        for i in 0..samplers.len() {
            bindings.push(&samplers[i]);
        }

        return bindings;
    }

    let verts_pos_array = array_of_buffer_bindings_resource(&scene.verts_pos);
    let verts_array     = array_of_buffer_bindings_resource(&scene.verts);
    let indices_array   = array_of_buffer_bindings_resource(&scene.indices);
    let bvh_nodes_array = array_of_buffer_bindings_resource(&scene.bvh_nodes);
    //let texture_views   = array_of_texture_views(&scene.textures);
    //let textures_array  = array_of_texture_bindings_resource(&texture_views);
    //let samplers_array  = array_of_sampler_bindings_resource(&scene.samplers);

/*
    println!("{}", verts_pos_array.len);
    println!("{}", verts_array.len);
    println!("{}", indices_array.len);
    println!("{}", bhv_nodes_array.len);
    println!("{}", texture_views.len);
    println!("{}", textures_arr.len);
*/

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::BufferArray(verts_pos_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::BufferArray(verts_array.as_slice())     },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::BufferArray(indices_array.as_slice())   },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::BufferArray(bvh_nodes_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 4, resource: buffer_resource(&scene.tlas_nodes) },
            wgpu::BindGroupEntry { binding: 5, resource: buffer_resource(&scene.instances) },
            //wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureViewArray(textures_array.as_slice()) },
            //wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::SamplerArray(&samplers_array.as_slice()) },
        ]
    });


    let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_resource(&camera_transform_uniform) }
        ]
    });

    let render_target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.pipeline.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&render_target_view) },
        ]
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&shader_params.pipeline);
        compute_pass.set_bind_group(0, &scene_bind_group, &[]);
        compute_pass.set_bind_group(1, &settings_bind_group, &[]);
        compute_pass.set_bind_group(2, &render_target_bind_group, &[]);
        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 8;
        const WORKGROUP_SIZE_Y: u32 = 8;
        let num_workers_x = (render_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (render_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

pub enum DebugVisualizationType
{
    VisualizeNormals,
    VisualizeWireframe,
    VisualizePreview,
    VisualizeBVHBoxTests { threshold: i32 },
    VisualizeBVHTriTests { threshold: i32 }
}

pub struct DebugParams
{
    pub viz: DebugVisualizationType,
    pub display_env_map: bool
}

pub fn pathtrace_scene_debug(device: &wgpu::Device, scene: &SceneDesc, render_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, debug_params: &DebugParams)
{

}

////////
// BVH creation

// NOTE: This modifies the indices array to change the order of triangles (indices)
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
    let res = upload_storage_buffer(&device, &queue, to_u8_slice(&bvh));
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

fn get_tri(verts: &[f32], indices: &[u32], tri_idx: usize)->(Vec3, Vec3, Vec3)
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

////////
// Tonemapping

pub struct TonemapShaderParams
{
    pub aces_pipeline: wgpu::RenderPipeline,
    pub filmic_pipeline: wgpu::RenderPipeline,
}

pub fn build_tonemap_shader_params(device: &wgpu::Device) -> TonemapShaderParams
{
    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Aces Tonemapping Shader"),
        source: wgpu::ShaderSource::Wgsl(TONEMAPPING_SRC.into())
    };

    let tonemap_shader = device.create_shader_module(shader_desc);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            }
        ]
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lupin Tonemapping Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    fn tonemap_pipeline_descriptor<'a>(shader: &'a wgpu::ShaderModule, pipeline_layout: &'a wgpu::PipelineLayout, frag_main: &'static str) -> wgpu::RenderPipelineDescriptor<'a>
    {
        return wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(frag_main),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,  // TODO How to work with other formats?
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        }
    }

    let aces_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "aces_main"));
    let filmic_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "filmic_main"));

    return TonemapShaderParams {
        aces_pipeline,
        filmic_pipeline
    };
}

pub fn aces_tonemapping(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &TonemapShaderParams, hdr_texture: &wgpu::Texture, render_target: &wgpu::Texture, exposure: f32)
{
    let render_target_view = render_target.create_view(&Default::default());
    let hdr_texture_view = hdr_texture.create_view(&Default::default());

    use wgpu::util::DeviceExt;
    let exposure_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: to_u8_slice(&[exposure]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_resource(&exposure_uniform) },
        ]
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None
        });

        pass.set_pipeline(&shader_params.aces_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn filmic_tonemapping(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &TonemapShaderParams, hdr_texture: &wgpu::Texture, render_target: &wgpu::Texture, exposure: f32)
{

}

fn apply_tonemapping(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &TonemapShaderParams, hdr_texture: &wgpu::Texture, render_target: &wgpu::Texture, exposure: f32)
{

}

fn buffer_resource(buffer: &wgpu::Buffer) -> wgpu::BindingResource
{
    return wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buffer, offset: 0, size: None });
}
