
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

pub struct Scene
{
    // Meshes
    pub verts_pos_array: Vec<wgpu::Buffer>,
    pub verts_array: Vec<wgpu::Buffer>,
    pub indices_array: Vec<wgpu::Buffer>,
    pub bvh_nodes_array: Vec<wgpu::Buffer>,

    pub tlas_nodes: wgpu::Buffer,
    pub instances: wgpu::Buffer,
    pub materials: wgpu::Buffer,

    pub textures: Vec<wgpu::Texture>,
    pub samplers: Vec<wgpu::Sampler>,
    pub environments: wgpu::Buffer,

    // Auxiliary data structures.
    pub lights: Lights,
}

pub struct SceneCPU
{
    pub verts_pos_array: Vec<Vec<VertexPos>>,
    pub verts_array: Vec<Vec<Vertex>>,
    pub indices_array: Vec<Vec<u32>>,
    pub bvh_nodes_array: Vec<Vec<BvhNode>>,
    pub mesh_aabbs: Vec<Aabb>,

    pub tlas_nodes: Vec<TlasNode>,
    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,

    pub environments: Vec<Environment>,
}

pub struct Lights
{
    pub lights: wgpu::Buffer,
    pub alias_tables: Vec::<wgpu::Buffer>,
    pub env_alias_tables: Vec::<wgpu::Buffer>,
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

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Environment
{
    pub emission: Vec3,
    pub emission_tex_idx: u32,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Light
{
    pub instance_idx: u32,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct AliasBin
{
    pub prob: f32,
    pub alias_threshold: f32,  // [0, 1]; should select the alias if rnd_f32 > alias
    pub alias: u32,
}

// This doesn't include positions, as that
// is stored in a separate buffer for locality
#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vertex
{
    pub normal: Vec3,
    pub _padding0: f32,
    pub tex_coords: Vec2,
    pub _padding1: f32,
    pub _padding2: f32,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct VertexPos
{
    pub v: Vec3,
    pub _padding: f32,
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

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct PushConstants
{
    pub camera_transform: Mat4,
    pub id_offset: [u32; 2],
    pub accum_counter: u32,
    pub flags: u32,
    pub heatmap_min: f32,
    pub heatmap_max: f32,
}

// Debug flags
// NOTE: Coupled to shader.
pub const DEBUG_FLAG_TRI_CHECKS:  u32    = 1 << 0;
pub const DEBUG_FLAG_AABB_CHECKS: u32    = 1 << 1;
pub const DEBUG_FLAG_NUM_BOUNCES: u32    = 1 << 2;
pub const DEBUG_FLAG_FIRST_HIT_ONLY: u32 = 1 << 3;

// Constants
pub const BVH_MAX_DEPTH: i32 = 25;
pub const MAX_MESHES:    u32 = 15000;
pub const MAX_ENVS:      u32 = 10;
pub const MAX_TEXTURES:  u32 = 15000;
pub const MAX_SAMPLERS:  u32 = 32;
pub const NUM_STORAGE_BUFFERS_PER_MESH: u32 = 5;

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
                           wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY |
                           wgpu::Features::PUSH_CONSTANTS,
        required_limits: wgpu::Limits {
            max_storage_buffers_per_shader_stage: MAX_MESHES * NUM_STORAGE_BUFFERS_PER_MESH + MAX_ENVS + 64,
            max_sampled_textures_per_shader_stage: MAX_TEXTURES + 64,
            max_samplers_per_shader_stage: MAX_SAMPLERS + 8,
            max_push_constant_size: 128,
            ..Default::default()
        },
        memory_hints: Default::default(),
    };
}

// Shader params

pub struct PathtraceResources
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

pub struct BakedPathtraceParams
{
    pub with_runtime_checks: bool,
    pub max_bounces: u32,
    pub samples_per_pixel: u32,
}

impl Default for BakedPathtraceParams
{
    fn default() -> Self
    {
        return Self {
            with_runtime_checks: true,
            max_bounces: 5,
            samples_per_pixel: 1,
        };
    }
}

pub fn build_pathtrace_resources(device: &wgpu::Device, baked_pathtrace_params: &BakedPathtraceParams) -> PathtraceResources
{
    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Pathtracer Shader"),
        source: wgpu::ShaderSource::Wgsl(DEFAULT_PATHTRACER_SRC.into())
    };

    let shader;
    if baked_pathtrace_params.with_runtime_checks {
        shader = device.create_shader_module(shader_desc);
    } else {
        shader = unsafe { device.create_shader_module_trusted(shader_desc, wgpu::ShaderRuntimeChecks::unchecked()) };
    }

    let scene_bindgroup_layout = create_pathtracer_scene_bindgroup_layout(device);
    let settings_bindgroup_layout = create_pathtracer_settings_bindgroup_layout(device);
    let render_target_bindgroup_layout = create_pathtracer_output_bindgroup_layout(device);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lupin Pathtracer Pipeline Layout"),
        bind_group_layouts: &[
            &scene_bindgroup_layout,
            &settings_bindgroup_layout,
            &render_target_bindgroup_layout,
        ],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32,
        }],
    });

    let constants = std::collections::HashMap::from([
        (std::string::String::from("MAX_BOUNCES"), baked_pathtrace_params.max_bounces as f64),
        (std::string::String::from("SAMPLES_PER_PIXEL"), baked_pathtrace_params.samples_per_pixel as f64),
    ]);

    let debug_constants = std::collections::HashMap::from([
        (std::string::String::from("MAX_BOUNCES"), baked_pathtrace_params.max_bounces as f64),
        (std::string::String::from("SAMPLES_PER_PIXEL"), baked_pathtrace_params.samples_per_pixel as f64),
        (std::string::String::from("DEBUG"), 1 as f64),
    ]);

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("pathtrace_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    let debug_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Debug Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("pathtrace_debug_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &debug_constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    let gbuffer_albedo_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer GBuffer Albedo Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gbuffer_albedo_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    let gbuffer_normals_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer GBuffer Normals Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gbuffer_normals_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: true,
        },
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

    return PathtraceResources {
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

#[derive(Copy, Clone, Debug)]
pub struct TileParams
{
    pub tile_size: u32,  // In pixels.
}

impl Default for TileParams
{
    fn default() -> Self
    {
        return Self {
            tile_size: 256,
        };
    }
}

pub struct PathtraceDesc<'a>
{
    pub scene: &'a Scene,
    pub render_target: &'a wgpu::Texture,
    pub resources: &'a PathtraceResources,
    pub accum_params: &'a AccumulationParams<'a>,
    pub tile_params: &'a TileParams,
    pub camera_transform: Mat4,
}

/// Pathtrace the entire image. Internally the image is still split into tiles,
/// because most operating system have a hard limit on the execution time of shaders,
/// and if that is exceeded a hard driver reset is performed. The tile size can still
/// be adjusted from the tile_params in desc.
pub fn pathtrace_scene(device: &wgpu::Device, queue: &wgpu::Queue, desc: &PathtraceDesc)
{
    let num_tiles_x = (desc.render_target.width().max(1) - 1)  / desc.tile_params.tile_size + 1;
    let num_tiles_y = (desc.render_target.height().max(1) - 1) / desc.tile_params.tile_size + 1;
    let total_tiles = num_tiles_x * num_tiles_y;

    // TODO: Check format and usage of render target params and others.

    let scene = desc.scene;
    let render_target = desc.render_target;
    let resources = desc.resources;
    let accum_params = desc.accum_params;
    let camera_transform = desc.camera_transform;

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, resources, accum_params.prev_frame);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, resources, Some(render_target), None, None);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&resources.pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);

        let push_constants = PushConstants {
            camera_transform: camera_transform,
            id_offset: [0, 0],
            accum_counter: accum_params.accum_counter,
            flags: 0,
            heatmap_min: 0.0,
            heatmap_max: 0.0,
        };
        compute_pass.set_push_constants(0, to_u8_slice(&[push_constants]));

        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 4;
        const WORKGROUP_SIZE_Y: u32 = 4;
        let num_workers_x = (render_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (render_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

/// Partial pathtracing. This makes it possible to break up a single pathtrace
/// call. Useful when the scene is particularly big and you have real-time things
/// running in your application (e.g. a GUI).
pub fn pathtrace_scene_tiles(device: &wgpu::Device, queue: &wgpu::Queue, desc: &PathtraceDesc, tile_counter: &mut u32, tiles_to_render: u32)
{
    let num_tiles_x = (desc.render_target.width().max(1) - 1)  / desc.tile_params.tile_size + 1;
    let num_tiles_y = (desc.render_target.height().max(1) - 1) / desc.tile_params.tile_size + 1;
    let total_tiles = num_tiles_x * num_tiles_y;
    assert!(/* *tile_counter >= 0 && */ *tile_counter < total_tiles, "tile_counter out of range!");

    // TODO: Check format and usage of render target params and others.

    let scene = desc.scene;
    let render_target = desc.render_target;
    let resources = desc.resources;
    let accum_params = desc.accum_params;
    let camera_transform = desc.camera_transform;
    let tile_size = desc.tile_params.tile_size;

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, resources, accum_params.prev_frame);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, resources, Some(render_target), None, None);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&resources.pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);

        for i in *tile_counter..u32::min(*tile_counter + tiles_to_render, total_tiles)
        {
            let push_constants = PushConstants {
                camera_transform: camera_transform,
                id_offset: [
                    (i % num_tiles_x) * tile_size,
                    (i / num_tiles_x) * tile_size,
                ],
                accum_counter: accum_params.accum_counter,
                flags: 0,
                heatmap_min: 0.0,
                heatmap_max: 0.0,
            };
            compute_pass.set_push_constants(0, to_u8_slice(&[push_constants]));

            // NOTE: This is tied to the corresponding value in the shader
            const WORKGROUP_SIZE_X: u32 = 4;
            const WORKGROUP_SIZE_Y: u32 = 4;
            let num_workers_x = (tile_size + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
            let num_workers_y = (tile_size + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
            compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
        }
    }

    queue.submit(Some(encoder.finish()));

    // Advance tile_counter.
    *tile_counter = u32::min(*tile_counter + tiles_to_render, total_tiles) % (total_tiles);
}

pub enum DebugVizType
{
    BVHAABBChecks,
    BVHTriChecks,
    NumBounces,
}

pub struct DebugVizDesc
{
    pub viz_type: DebugVizType,
    pub heatmap_min: f32,
    pub heatmap_max: f32,
    pub first_hit_only: bool,
}

pub fn pathtrace_scene_debug(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, albedo_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4, debug_desc: &DebugVizDesc)
{
    // TODO: Check format and usage of render target params and others.

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, resources, None);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, resources, None, Some(albedo_target), None);

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&resources.debug_pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);

        let mut debug_flags: u32 = 0;
        match debug_desc.viz_type
        {
            DebugVizType::BVHAABBChecks =>
            {
                debug_flags |= DEBUG_FLAG_AABB_CHECKS;
            },
            DebugVizType::BVHTriChecks =>
            {
                debug_flags |= DEBUG_FLAG_TRI_CHECKS;
            },
            DebugVizType::NumBounces =>
            {
                debug_flags |= DEBUG_FLAG_NUM_BOUNCES;
            },
        }

        if debug_desc.first_hit_only {
            debug_flags |= DEBUG_FLAG_FIRST_HIT_ONLY
        }

        let push_constants = PushConstants {
            camera_transform: camera_transform,
            id_offset: [0, 0],
            accum_counter: 0,
            flags: debug_flags,
            heatmap_min: debug_desc.heatmap_min,
            heatmap_max: debug_desc.heatmap_max,
        };
        compute_pass.set_push_constants(0, to_u8_slice(&[push_constants]));

        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 4;
        const WORKGROUP_SIZE_Y: u32 = 4;
        let num_workers_x = (albedo_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (albedo_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn raycast_gbuffers(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, albedo_target: &wgpu::Texture, normals_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4)
{
    raycast_albedo(device, queue, scene, albedo_target, resources, camera_transform);
    raycast_normals(device, queue, scene, normals_target, resources, camera_transform);
}

pub fn raycast_albedo(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, albedo_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4)
{
    // TODO: Check format and usage of render target params and others.

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, resources, None);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, resources, None, Some(albedo_target), None);

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&resources.gbuffer_albedo_pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);

        let push_constants = PushConstants {
            camera_transform: camera_transform,
            id_offset: [0, 0],
            accum_counter: 0,
            flags: 0,
            heatmap_min: 0.0,
            heatmap_max: 0.0,
        };
        compute_pass.set_push_constants(0, to_u8_slice(&[push_constants]));

        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 4;
        const WORKGROUP_SIZE_Y: u32 = 4;
        let num_workers_x = (albedo_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (albedo_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn raycast_normals(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, normals_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4)
{
    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, queue, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, queue, resources, None);
    let output_bindgroup = create_pathtracer_output_bindgroup(device, queue, resources, None, None, Some(normals_target));

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        compute_pass.set_pipeline(&resources.gbuffer_normals_pipeline);
        compute_pass.set_bind_group(0, &scene_bindgroup, &[]);
        compute_pass.set_bind_group(1, &settings_bindgroup, &[]);
        compute_pass.set_bind_group(2, &output_bindgroup, &[]);

        let push_constants = PushConstants {
            camera_transform: camera_transform,
            id_offset: [0, 0],
            accum_counter: 0,
            flags: 0,
            heatmap_min: 0.0,
            heatmap_max: 0.0,
        };
        compute_pass.set_push_constants(0, to_u8_slice(&[push_constants]));

        // NOTE: This is tied to the corresponding value in the shader
        const WORKGROUP_SIZE_X: u32 = 4;
        const WORKGROUP_SIZE_Y: u32 = 4;
        let num_workers_x = (normals_target.width() + WORKGROUP_SIZE_X - 1) / WORKGROUP_SIZE_X;
        let num_workers_y = (normals_target.height() + WORKGROUP_SIZE_Y - 1) / WORKGROUP_SIZE_Y;
        compute_pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
    }

    queue.submit(Some(encoder.finish()));
}

////////
// Tonemapping

pub struct TonemapResources
{
    pub identity_pipeline: wgpu::RenderPipeline,
    pub aces_pipeline: wgpu::RenderPipeline,
    pub filmic_pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
}

pub fn build_tonemap_resources(device: &wgpu::Device) -> TonemapResources
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

    return TonemapResources {
        identity_pipeline,
        aces_pipeline,
        filmic_pipeline,
        sampler,
    };
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub enum TonemapOperator
{
    #[default] Aces,
    FilmicUC2,
    FilmicCustom { linear_white: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32 },
}

#[derive(Default, Clone, Copy, Debug)]
pub struct TonemapParams
{
    pub operator: TonemapOperator,
    pub exposure: f32,
}

pub struct TonemapDesc<'a>
{
    pub resources: &'a TonemapResources,
    pub hdr_texture: &'a wgpu::Texture,
    pub render_target: &'a wgpu::Texture,
    pub tonemap_params: &'a TonemapParams,
}

pub fn apply_tonemapping(device: &wgpu::Device, queue: &wgpu::Queue, desc: &TonemapDesc)
{
    let resources = desc.resources;
    let hdr_texture = desc.hdr_texture;
    let render_target = desc.render_target;
    let tonemap_params = desc.tonemap_params;

    let render_target_view = render_target.create_view(&Default::default());
    let hdr_texture_view = hdr_texture.create_view(&Default::default());

    // NOTE: Coupled to the struct in the shader of the same name.
    #[derive(Default)]
    #[repr(C)]
    struct TonemapResources
    {
        exposure: f32,
        // For filmic tonemapping
        linear_white: f32,
        a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
    }

    let mut params = TonemapResources::default();
    params.exposure = tonemap_params.exposure;

    let pipeline = match tonemap_params.operator
    {
        TonemapOperator::Aces      => &resources.aces_pipeline,
        TonemapOperator::FilmicUC2 =>
        {
            params.linear_white = 11.2;
            params.a = 0.22;
            params.b = 0.3;
            params.c = 0.1;
            params.d = 0.2;
            params.e = 0.01;
            params.f = 0.30;

            &resources.filmic_pipeline
        }
        TonemapOperator::FilmicCustom { linear_white, a, b, c, d, e, f } =>
        {
            params.linear_white = linear_white;
            params.a = a;
            params.b = b;
            params.c = c;
            params.d = d;
            params.e = e;
            params.f = f;

            &resources.filmic_pipeline
        }
    };

    let params_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: to_u8_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &resources.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
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

pub fn convert_to_ldr_no_tonemap(device: &wgpu::Device, queue: &wgpu::Queue, resources: &TonemapResources, hdr_texture: &wgpu::Texture, render_target: &wgpu::Texture)
{
    let render_target_view = render_target.create_view(&Default::default());
    let hdr_texture_view = hdr_texture.create_view(&Default::default());

    #[derive(Default)]
    #[repr(C)]
    struct TonemapResources
    {
        exposure: f32,
        // For filmic tonemapping
        linear_white: f32,
        a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
    }

    let params = TonemapResources::default();

    let params_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: to_u8_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &resources.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
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

        pass.set_pipeline(&resources.identity_pipeline);
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

fn array_of_buffer_bindings_resource<'a>(buffers: &'a Vec<wgpu::Buffer>, empty_buffer: &'a wgpu::Buffer) -> Vec<wgpu::BufferBinding<'a>>
{
    // NOTE: Arrays of bindings can't be empty.
    if buffers.len() <= 0
    {
        return vec![wgpu::BufferBinding {
            buffer: empty_buffer,
            offset: 0,
            size: None,
        }];
    }

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

fn create_pathtracer_scene_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources, scene: &Scene) -> wgpu::BindGroup
{
    let verts_pos = &scene.verts_pos_array;
    let verts = &scene.verts_array;
    let indices = &scene.indices_array;
    let bvh_nodes = &scene.bvh_nodes_array;

    use crate::wgpu_utils::*;
    let empty_buf = create_empty_storage_buffer(device);
    let verts_pos_array = array_of_buffer_bindings_resource(&verts_pos, &empty_buf);
    let verts_array     = array_of_buffer_bindings_resource(&verts, &empty_buf);
    let indices_array   = array_of_buffer_bindings_resource(&indices, &empty_buf);
    let bvh_nodes_array = array_of_buffer_bindings_resource(&bvh_nodes, &empty_buf);
    let texture_views   = array_of_texture_views(&scene.textures);
    let textures_array  = array_of_texture_bindings_resource(&texture_views);
    let samplers_array  = array_of_sampler_bindings_resource(&scene.samplers);
    let alias_table_array = array_of_buffer_bindings_resource(&scene.lights.alias_tables, &empty_buf);
    let env_alias_table_array = array_of_buffer_bindings_resource(&scene.lights.env_alias_tables, &empty_buf);

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Scene bind group"),
        layout: &resources.pipeline.get_bind_group_layout(0),
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
            wgpu::BindGroupEntry { binding: 9,  resource: buffer_resource(&scene.environments) },
            wgpu::BindGroupEntry { binding: 10, resource: buffer_resource(&scene.environments) },  // TODO TODO
            wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::BufferArray(alias_table_array.as_slice()) },
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::BufferArray(env_alias_table_array.as_slice()) },
        ]
    });

    return scene_bind_group;
}

fn create_pathtracer_scene_bindgroup_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout
{
    assert!(NUM_STORAGE_BUFFERS_PER_MESH == 5);

    return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
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
            wgpu::BindGroupLayoutEntry {  // environments
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // lights
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // alias_tables
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES),
            },
            wgpu::BindGroupLayoutEntry {  // env_alias_tables
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_ENVS),
            },
        ]
    });
}

fn create_pathtracer_settings_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources, prev_frame: Option<&wgpu::Texture>) -> wgpu::BindGroup
{
    let prev_frame_view;
    if let Some(prev_frame) = prev_frame
    {
        prev_frame_view = prev_frame.create_view(&Default::default());
    }
    else
    {
        prev_frame_view = resources.dummy_prev_frame_texture.create_view(&Default::default());
    }

    let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Settings bind group"),
        layout: &resources.pipeline.get_bind_group_layout(1),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&prev_frame_view) },
        ]
    });

    return settings_bind_group;
}

fn create_pathtracer_settings_bindgroup_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout
{
    return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: Some("Pathtracer settings bindgroup"),
        entries: &[
            wgpu::BindGroupLayoutEntry {  // prev frame
                binding: 0,
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
}

fn create_pathtracer_output_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources, main_target: Option<&wgpu::Texture>, albedo: Option<&wgpu::Texture>, normals: Option<&wgpu::Texture>) -> wgpu::BindGroup
{
    let render_target_view  = match main_target
    {
        Some(tex) => tex.create_view(&Default::default()),
        None =>      resources.dummy_output_texture.create_view(&Default::default()),
    };
    let gbuffer_albedo_view = match albedo
    {
        Some(tex) => tex.create_view(&Default::default()),
        None =>      resources.dummy_albedo_texture.create_view(&Default::default()),
    };
    let gbuffer_normals_view = match normals
    {
        Some(tex) => tex.create_view(&Default::default()),
        None =>      resources.dummy_normals_texture.create_view(&Default::default()),
    };

    let render_target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pathtracer output bindgroup"),
        layout: &resources.pipeline.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&render_target_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gbuffer_albedo_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer_normals_view) },
        ]
    });

    return render_target_bind_group;
}

fn create_pathtracer_output_bindgroup_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout
{
    return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
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
}
