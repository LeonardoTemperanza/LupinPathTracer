
use crate::base::*;
use crate::wgpu_utils::*;

use wgpu::util::DeviceExt;  // For some extra device traits.

pub static DEFAULT_PATHTRACER_SRC: &str = include_str!("shaders/pathtracer.wgsl");
pub static TONEMAPPING_SRC: &str = include_str!("shaders/tonemapping.wgsl");

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vec2
{
    pub x: f32,
    pub y: f32
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
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

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Aabb
{
    pub min: Vec3,
    pub max: Vec3
}

pub struct SceneDesc
{
    pub meshes: Vec<Mesh>,
    pub tlas_nodes: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub materials: wgpu::Buffer,

    pub textures: Vec<wgpu::Texture>,
    pub samplers: Vec<wgpu::Sampler>,
    pub env_map: wgpu::Texture,
    pub env_map_sampler: wgpu::Sampler,
}

pub struct Mesh
{
    pub verts_pos: wgpu::Buffer,
    pub verts: wgpu::Buffer,
    pub indices: wgpu::Buffer,
    pub bvh_nodes: wgpu::Buffer,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Instance
{
    pub inv_transform: Mat4,
    pub mesh_idx: u32,
    pub mat_idx: u32,
    pub padding0: f32,
    pub padding1: f32,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(u32)]
pub enum MaterialType
{
    #[default]
    Matte       = 0,
    Glossy      = 1,
    Reflective  = 2,
    Transparent = 3,
    Refractive  = 4,
    Subsurface  = 5,
    Volumetric  = 6,
    GltfPbr     = 7,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Material
{
    pub color: Vec4,
    pub emission: Vec4,
    pub scattering: Vec4,
    pub mat_type: MaterialType,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub sc_anisotropy: f32,
    pub tr_depth: f32,

    pub color_tex_idx:      u32,
    pub emission_tex_idx:   u32,
    pub roughness_tex_idx:  u32,
    pub scattering_tex_idx: u32,
    pub normal_tex_idx:     u32,
    pub padding0: u32,
}

impl Material
{
    pub fn new(mat_type: MaterialType, color: Vec4, emission: Vec4, scattering: Vec4,
               roughness: f32, metallic: f32, ior: f32, sc_anisotropy: f32, tr_depth: f32,
               color_tex_idx: u32, emission_tex_idx: u32, roughness_tex_idx: u32, scattering_tex_idx: u32,
               normal_tex_idx: u32) -> Self
    {
        return Self {
            mat_type, color, emission, scattering, roughness, metallic, ior, sc_anisotropy, tr_depth,
            color_tex_idx, emission_tex_idx, roughness_tex_idx, scattering_tex_idx, normal_tex_idx,
            padding0: 0
        }
    }
}

// This doesn't include positions, as that
// is stored in a separate buffer for locality
#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
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
#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct BvhNode
{
    pub aabb_min: Vec3,
    // If tri_count is 0, this is first_child
    // otherwise this is tri_begin
    pub tri_begin_or_first_child: u32,
    pub aabb_max: Vec3,
    pub tri_count: u32
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct TlasNode
{
    pub aabb_min: Vec3,
    pub left_right: u32,  // 2x16 bits. If it's 0, this node is a leaf
    pub aabb_max: Vec3,
    pub instance_idx: u32
}

// Constants
pub const BVH_MAX_DEPTH: i32 = 25;
pub const MAX_MESHES:    u32 = 15000;
pub const MAX_TEXTURES:  u32 = 15000;
pub const MAX_SAMPLERS:  u32 = 32;
pub const NUM_STORAGE_BUFFERS_PER_MESH: u32 = 4;

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
            max_storage_buffers_per_shader_stage: MAX_MESHES * NUM_STORAGE_BUFFERS_PER_MESH + 64,
            max_sampled_textures_per_shader_stage: MAX_TEXTURES + 64,
            max_samplers_per_shader_stage: MAX_SAMPLERS + 8,
            ..Default::default()
        },
        memory_hints: Default::default(),
    };
}

// Shader params

pub struct PathtraceShaderParams
{
    pub pipeline: wgpu::ComputePipeline,
    pub debug_pipeline: wgpu::ComputePipeline,
    pub gbuffer_albedo_pipeline: wgpu::ComputePipeline,
    pub gbuffer_normals_pipeline: wgpu::ComputePipeline,
    pub dummy_prev_frame_texture: wgpu::Texture,
    pub dummy_albedo_texture: wgpu::Texture,
    pub dummy_normals_texture: wgpu::Texture,
    pub dummy_output_texture: wgpu::Texture,
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
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // verts
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // indices
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // bvh_nodes
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
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
            wgpu::BindGroupLayoutEntry {  // materials
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // textures
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: std::num::NonZero::new(MAX_TEXTURES)
            },
            wgpu::BindGroupLayoutEntry {  // samplers
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: std::num::NonZero::new(MAX_SAMPLERS)
            },
            wgpu::BindGroupLayoutEntry {  // env map texture
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {  // env map sampler
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None
            },
        ]
    });

    assert!(NUM_STORAGE_BUFFERS_PER_MESH == 4);

    let settings_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {  // camera transform
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {  // accum_counter
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {  // prev frame
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
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
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba8Snorm,
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
        entry_point: Some("pathtrace_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let debug_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Debug Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("pathtrace_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let gbuffer_albedo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer GBuffer Albedo Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gbuffer_albedo_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let gbuffer_normals_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer GBuffer Normals Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gbuffer_normals_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };

    // TODO: Make this a 1x1 white texture instead of garbage data.
    let dummy_prev_frame_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Sample Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    // TODO: Make this a 1x1 white texture instead of garbage data.
    let dummy_output_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Storage Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // TODO: Make this a 1x1 white texture instead of garbage data.
    let dummy_albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Storage Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    // TODO: Make this a 1x1 white texture instead of garbage data.
    let dummy_normals_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Storage Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Snorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });

    return PathtraceShaderParams {
        pipeline,
        debug_pipeline,
        gbuffer_albedo_pipeline,
        gbuffer_normals_pipeline,
        dummy_prev_frame_texture,
        dummy_output_texture,
        dummy_albedo_texture,
        dummy_normals_texture,
    };
}

pub struct AccumulationParams<'a>
{
    pub prev_frame: Option<&'a wgpu::Texture>,
    pub accum_counter: u32,
}

pub fn pathtrace_scene(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, render_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, accum_params: &AccumulationParams, camera_transform: Mat4)
{
    // TODO: Check format and usage of render target params and others.

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, shader_params, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, shader_params, Some(accum_params), camera_transform);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, shader_params, Some(render_target), None, None);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&shader_params.pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);
        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 8;
        const WORKGROUP_SIZE_Y: u32 = 8;
        let num_workers_x = (render_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (render_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

pub enum DebugAccelStructureVizType
{
    VizBVHBoxTests { threshold: i32 },
    VizBVHTriTests { threshold: i32 },
    VizBVHAllLevels,
    VizBVHOneLevel { level: i32 },
}

pub enum DebugMeshVizType
{
    VizNormals,
    VizWireframe,
    VizColor,
}

pub struct DebugParams
{
    pub accel_viz: DebugAccelStructureVizType,
    pub mesh_viz: DebugMeshVizType,
    pub display_env_map: bool
}

#[derive(Default)]
#[repr(C)]
struct DebugInput
{
    bvh_viz_type: i32,
    mesh_viz_type: i32,
    threshold: i32,
    level: i32,
}

/*
pub fn pathtrace_scene_debug(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, render_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, camera_transform: Mat4)
{

}
*/

pub fn raycast_gbuffers(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, albedo_target: &wgpu::Texture, normals_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, camera_transform: Mat4)
{
    raycast_albedo(device, queue, scene, albedo_target, shader_params, camera_transform);
    raycast_normals(device, queue, scene, normals_target, shader_params, camera_transform);
}

pub fn raycast_albedo(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, albedo_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, camera_transform: Mat4)
{
    // TODO: Check format and usage of render target params and others.

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, shader_params, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, shader_params, None, camera_transform);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, shader_params, None, Some(albedo_target), None);

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&shader_params.gbuffer_albedo_pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);
        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 8;
        const WORKGROUP_SIZE_Y: u32 = 8;
        let num_workers_x = (albedo_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (albedo_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn raycast_normals(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, normals_target: &wgpu::Texture, shader_params: &PathtraceShaderParams, camera_transform: Mat4)
{
    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, shader_params, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, shader_params, None, camera_transform);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, shader_params, None, None, Some(normals_target));

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&shader_params.gbuffer_normals_pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);
        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 8;
        const WORKGROUP_SIZE_Y: u32 = 8;
        let num_workers_x = (normals_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (normals_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

////////
// Acceleration structures

// The aabbs are in model space, and they're indexed using the instance mesh_idx member.
pub fn build_tlas(device: &wgpu::Device, queue: &wgpu::Queue, instances: &[Instance], model_aabbs: &[Aabb]) -> wgpu::Buffer
{
    let mut node_indices = Vec::<u32>::with_capacity(instances.len());
    let mut tlas = Vec::<TlasNode>::with_capacity(instances.len() * 2 - 1);

    tlas.push(TlasNode::default());  // Reserve slot for root node.

    // Assign a leaf node for each instance.
    for i in 0..instances.len() as u32
    {
        let instance = instances[i as usize];
        let model_aabb = model_aabbs[instance.mesh_idx as usize];
        let transform = mat4_inverse(instance.inv_transform);
        let aabb_trans = transform_aabb(model_aabb.min, model_aabb.max, transform);

        let tlas_node = TlasNode {
            aabb_min: aabb_trans.min,
            aabb_max: aabb_trans.max,
            instance_idx: i,
            left_right: 0,  // Makes it a leaf.
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
                left_right: node_idx_a + (node_idx_b << 16),
                aabb_min: node_a.aabb_min.min(node_b.aabb_min),
                aabb_max: node_a.aabb_max.max(node_b.aabb_max),
                instance_idx: 0,  // Unused.
            };
            tlas.push(new_node);

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

    tlas[0] = tlas[node_indices[a as usize] as usize];
    return upload_storage_buffer(&device, &queue, to_u8_slice(&tlas));
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

// NOTE: This modifies the indices array to change the order of triangles (indices)
// based on the BVH
pub fn build_bvh(device: &wgpu::Device, queue: &wgpu::Queue, verts: &[f32], indices: &mut[u32]) -> wgpu::Buffer
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
    let half_area: f32 = size.x * (size.y + size.z) + size.y * size.z;
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
    pub identity_pipeline: wgpu::RenderPipeline,
    pub aces_pipeline: wgpu::RenderPipeline,
    pub filmic_pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
}

pub fn build_tonemap_shader_params(device: &wgpu::Device) -> TonemapShaderParams
{
    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Tonemapping Shader"),
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

    let identity_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "no_tonemap_main"));
    let aces_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "aces_main"));
    let filmic_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "filmic_main"));

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    return TonemapShaderParams {
        identity_pipeline,
        aces_pipeline,
        filmic_pipeline,
        sampler,
    };
}

#[derive(Clone, Copy, Debug)]
pub enum TonemapOperator
{
    Aces,
    FilmicUC2,
    FilmicUC2Custom { linear_white: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32 },
}

#[derive(Clone, Copy, Debug)]
pub struct TonemapParams
{
    pub operator: TonemapOperator,
    pub exposure: f32,
}

pub fn apply_tonemapping(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &TonemapShaderParams, hdr_texture: &wgpu::Texture, render_target: &wgpu::Texture, tonemap_params: &TonemapParams)
{
    let render_target_view = render_target.create_view(&Default::default());
    let hdr_texture_view = hdr_texture.create_view(&Default::default());

    // NOTE: Coupled to the struct in the shader of the same name.
    #[derive(Default)]
    #[repr(C)]
    struct TonemapShaderParams
    {
        exposure: f32,
        // For filmic tonemapping
        linear_white: f32,
        a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
    }

    let mut params = TonemapShaderParams::default();
    params.exposure = tonemap_params.exposure;

    let pipeline = match tonemap_params.operator
    {
        TonemapOperator::Aces      => &shader_params.aces_pipeline,
        TonemapOperator::FilmicUC2 =>
        {
            params.linear_white = 11.2;
            params.a = 0.22;
            params.b = 0.3;
            params.c = 0.1;
            params.d = 0.2;
            params.e = 0.01;
            params.f = 0.30;

            &shader_params.filmic_pipeline
        }
        TonemapOperator::FilmicUC2Custom { linear_white, a, b, c, d, e, f } =>
        {
            params.linear_white = linear_white;
            params.a = a;
            params.b = b;
            params.c = c;
            params.d = d;
            params.e = e;
            params.f = f;

            &shader_params.filmic_pipeline
        }
    };

    let params_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: to_u8_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shader_params.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_resource(&params_uniform) },
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

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn convert_to_ldr_no_tonemap(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &TonemapShaderParams, hdr_texture: &wgpu::Texture, render_target: &wgpu::Texture)
{
    let render_target_view = render_target.create_view(&Default::default());
    let hdr_texture_view = hdr_texture.create_view(&Default::default());

    // NOTE: Coupled to the struct in the shader of the same name.
    #[derive(Default)]
    #[repr(C)]
    struct TonemapShaderParams
    {
        exposure: f32,
        // For filmic tonemapping
        linear_white: f32,
        a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
    }

    let params = TonemapShaderParams::default();

    let params_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: to_u8_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&shader_params.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_resource(&params_uniform) },
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

        pass.set_pipeline(&shader_params.identity_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

// Wrappers

fn buffer_resource(buffer: &wgpu::Buffer) -> wgpu::BindingResource
{
    return wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buffer, offset: 0, size: None });
}

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

fn array_of_buffer_bindings_ref_resource<'a>(buffers: &Vec<&'a wgpu::Buffer>) -> Vec<wgpu::BufferBinding<'a>>
{
    let mut bindings: Vec<wgpu::BufferBinding> = Vec::with_capacity(buffers.len());
    for i in 0..buffers.len()
    {
        bindings.push(wgpu::BufferBinding {
            buffer: buffers[i],
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

fn create_pathtracer_scene_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &PathtraceShaderParams, scene: &SceneDesc) -> wgpu::BindGroup
{
    // Convert AoS to SoA.
    let mut verts_pos = Vec::<&wgpu::Buffer>::with_capacity(scene.meshes.len());
    let mut verts = Vec::<&wgpu::Buffer>::with_capacity(scene.meshes.len());
    let mut indices = Vec::<&wgpu::Buffer>::with_capacity(scene.meshes.len());
    let mut bvh_nodes = Vec::<&wgpu::Buffer>::with_capacity(scene.meshes.len());
    for mesh in &scene.meshes
    {
        verts_pos.push(&mesh.verts_pos);
        verts.push(&mesh.verts);
        indices.push(&mesh.indices);
        bvh_nodes.push(&mesh.bvh_nodes);
    }

    use crate::wgpu_utils::*;
    let verts_pos_array = array_of_buffer_bindings_ref_resource(&verts_pos);
    let verts_array     = array_of_buffer_bindings_ref_resource(&verts);
    let indices_array   = array_of_buffer_bindings_ref_resource(&indices);
    let bvh_nodes_array = array_of_buffer_bindings_ref_resource(&bvh_nodes);
    let texture_views   = array_of_texture_views(&scene.textures);
    let textures_array  = array_of_texture_bindings_resource(&texture_views);
    let samplers_array  = array_of_sampler_bindings_resource(&scene.samplers);
    let env_map_view    = scene.env_map.create_view(&Default::default());

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0,  resource: wgpu::BindingResource::BufferArray(verts_pos_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::BufferArray(verts_array.as_slice())     },
            wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::BufferArray(indices_array.as_slice())   },
            wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::BufferArray(bvh_nodes_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 4,  resource: buffer_resource(&scene.tlas_nodes) },
            wgpu::BindGroupEntry { binding: 5,  resource: buffer_resource(&scene.instances) },
            wgpu::BindGroupEntry { binding: 6,  resource: buffer_resource(&scene.materials) },
            wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::TextureViewArray(textures_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::SamplerArray(samplers_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::TextureView(&env_map_view) },
            wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&scene.env_map_sampler) },
        ]
    });

    return scene_bind_group;
}

fn create_pathtracer_settings_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &PathtraceShaderParams, accum_params: Option<&AccumulationParams>, camera_transform: Mat4) -> wgpu::BindGroup
{
    let camera_transform_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: to_u8_slice(&[camera_transform]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let accum_counter_uniform;
    let prev_frame_view;
    if let Some(accum_params) = accum_params
    {
        accum_counter_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Frame ID Buffer"),
            contents: to_u8_slice(&[accum_params.accum_counter]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        prev_frame_view = match accum_params.prev_frame
        {
            None      => shader_params.dummy_prev_frame_texture.create_view(&Default::default()),
            Some(tex) => tex.create_view(&Default::default()),
        };
    }
    else
    {
        let accum_counter: u32 = 0;
        accum_counter_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Frame ID Buffer"),
            contents: to_u8_slice(&[accum_counter]),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        prev_frame_view = shader_params.dummy_prev_frame_texture.create_view(&Default::default());
    }

    let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buffer_resource(&camera_transform_uniform) },
            wgpu::BindGroupEntry { binding: 1, resource: buffer_resource(&accum_counter_uniform) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&prev_frame_view) }
        ]
    });

    return settings_bind_group;
}

fn create_pathtracer_output_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, shader_params: &PathtraceShaderParams, main_target: Option<&wgpu::Texture>, albedo: Option<&wgpu::Texture>, normals: Option<&wgpu::Texture>) -> wgpu::BindGroup
{
    let render_target_view  = match main_target
    {
        Some(tex) => tex.create_view(&Default::default()),
        None =>      shader_params.dummy_output_texture.create_view(&Default::default()),
    };
    let gbuffer_albedo_view = match albedo
    {
        Some(tex) => tex.create_view(&Default::default()),
        None =>      shader_params.dummy_albedo_texture.create_view(&Default::default()),
    };
    let gbuffer_normals_view = match normals
    {
        Some(tex) => tex.create_view(&Default::default()),
        None =>      shader_params.dummy_normals_texture.create_view(&Default::default()),
    };

    let render_target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &shader_params.pipeline.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&render_target_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffer_albedo_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer_normals_view) },
        ]
    });

    return render_target_bind_group;
}
