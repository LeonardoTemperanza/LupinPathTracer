
use crate::base::*;
use crate::wgpu_utils::*;

use rand::Rng;

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
    pub environments: wgpu::Buffer,

    // Auxiliary data structures.
    pub lights: Lights,
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
pub const DEBUG_FLAG_TRI_CHECKS:  u32       = 1 << 0;
pub const DEBUG_FLAG_AABB_CHECKS: u32       = 1 << 1;
pub const DEBUG_FLAG_NUM_BOUNCES: u32       = 1 << 2;
pub const DEBUG_FLAG_FIRST_HIT_ONLY: u32    = 1 << 3;

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
    pub scene: &'a SceneDesc,
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

pub fn pathtrace_scene_debug(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, albedo_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4, debug_desc: &DebugVizDesc)
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

pub fn raycast_gbuffers(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, albedo_target: &wgpu::Texture, normals_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4)
{
    raycast_albedo(device, queue, scene, albedo_target, resources, camera_transform);
    raycast_normals(device, queue, scene, normals_target, resources, camera_transform);
}

pub fn raycast_albedo(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, albedo_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4)
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

pub fn raycast_normals(device: &wgpu::Device, queue: &wgpu::Queue, scene: &SceneDesc, normals_target: &wgpu::Texture, resources: &PathtraceResources, camera_transform: Mat4)
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

fn create_pathtracer_scene_bindgroup(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources, scene: &SceneDesc) -> wgpu::BindGroup
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
    let empty_buf = create_empty_storage_buffer(device);
    let verts_pos_array = array_of_buffer_bindings_ref_resource(&verts_pos);
    let verts_array     = array_of_buffer_bindings_ref_resource(&verts);
    let indices_array   = array_of_buffer_bindings_ref_resource(&indices);
    let bvh_nodes_array = array_of_buffer_bindings_ref_resource(&bvh_nodes);
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

//////////////
// Lights
pub struct EnvMapInfo
{
    pub data: Vec::<Vec4>,
    pub width: u32,
    pub height: u32,
}
pub fn build_lights(device: &wgpu::Device, queue: &wgpu::Queue, instances: &[Instance], environments: &[Environment], envs_info: &[EnvMapInfo]) -> Lights
{
    assert!(environments.len() == envs_info.len(), "Mismatching sizes for environment data!");

    let lights = Vec::<Light>::new();
    let alias_tables = Vec::<wgpu::Buffer>::new();
    let mut env_alias_tables = Vec::<wgpu::Buffer>::new();

    for instance in instances
    {

    }

    for i in 0..environments.len()
    {
        let mut weights = Vec::<f32>::new();

        let scale = environments[i].emission;
        let env_tex = &envs_info[i].data;
        let env_width = envs_info[i].width;
        let env_height = envs_info[i].height;

        if scale.is_zero() { continue; }

        for (i, pixel) in env_tex.iter().enumerate()
        {
            let y = i / env_width as usize;
            let angle = (y as f32 + 0.5) * std::f32::consts::PI / env_height as f32;
            let prob = f32::max(f32::max(pixel.x, pixel.y), pixel.z) * f32::sin(angle);
            weights.push(prob);
        }

        let alias_table = build_alias_table(&weights);
        let alias_table_buf = upload_storage_buffer(device, queue, to_u8_slice(&alias_table));
        env_alias_tables.push(alias_table_buf);
    }

    return Lights {
        lights: upload_storage_buffer(device, queue, to_u8_slice(&lights)),
        alias_tables,
        env_alias_tables,
    };
}

// From: https://www.pbr-book.org/4ed/Sampling_Algorithms/The_Alias_Method#AliasTable
pub fn build_alias_table(weights: &[f32]) -> Vec::<AliasBin>
{
    if weights.is_empty() { return vec![]; }

    let mut bins = vec![AliasBin::default(); weights.len()];

    let mut sum: f64 = 0.0;
    for weight in weights {
        sum += *weight as f64;
    }

    assert!(sum > 0.0);

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
        assert!(f32::abs(over_item.prob_estimate - 1.0) < 0.05);
        bins[over_item.idx as usize].alias_threshold = 1.0;
        bins[over_item.idx as usize].alias = 0;
    }
    while !under.is_empty()
    {
        let under_item = under.pop().unwrap();
        assert!(f32::abs(under_item.prob_estimate - 1.0) < 0.05);
        bins[under_item.idx as usize].alias_threshold = 1.0;
        bins[under_item.idx as usize].alias = 0;
    }

    return bins;
}

// TODO: Turn this into an actual test.
fn test_alias_table(alias_table: &Vec::<AliasBin>, env_width: u32)
{
    println!("test:");
    struct TestValue
    {
        idx: usize,
        hits: u32
    }
    let mut test_values = Vec::<TestValue>::new();
    let mut rng = rand::thread_rng();
    for i in 0..1000000
    {
        let slot_idx: usize = rng.gen_range(0..alias_table.len()); // inclusive range
        let rnd: f32 = rng.gen();
        if rnd >= alias_table[slot_idx].alias_threshold
        {
            add_to_test_value_array(&mut test_values, alias_table[slot_idx].alias as usize)
        }
        else
        {
            add_to_test_value_array(&mut test_values, slot_idx);
        }
    }

    println!("results:");
    test_values.sort_by(|a, b| b.hits.cmp(&a.hits));
    for value in test_values
    {
        println!("x: {}, y: {}, hits: {}, prob: {}", value.idx % env_width as usize, value.idx / env_width as usize, value.hits, alias_table[value.idx].prob);
    }

    fn add_to_test_value_array(test_values: &mut Vec::<TestValue>, idx: usize)
    {
        let mut found = false;
        for value in test_values.iter_mut()
        {
            if value.idx as usize == idx
            {
                value.hits += 1;
                found = true;
                break;
            }
        }

        if !found
        {
            test_values.push(TestValue { idx: idx, hits: 1 });
        }
    }
}

//////////////
// Denoising.

/*
struct DenoiseResources
{

}

fn create_denoise_resources(device: &wgpu::Device, queue: &wgpu::Queue) -> DenoiseResources
{

}
*/

/*
pub fn transfer_to_cpu_and_denoise_image(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture, to_denoise: &wgpu::Texture, albedo: Option<&wgpu::Texture>, normals: Option<&wgpu::Texture>)
{
    // TODO: Check correctness of textures.

    // Transfer images to CPU.

    let cpu_input = copy_texture_to_cpu_sync(device, queue, to_denoise);
    let mut beauty = Vec::<f32>::with_capacity(cpu_input.len() / 4 * 3);
    for i in (0..cpu_input.len()).step_by(4) {
        beauty.push(cpu_input[i+0] as f32 / 255.0); // R
        beauty.push(cpu_input[i+1] as f32 / 255.0); // G
        beauty.push(cpu_input[i+2] as f32 / 255.0); // B
    }

    let mut albedo_buf = Vec::<f32>::new();
    if let Some(albedo_tex) = albedo
    {
        let cpu_albedo = copy_texture_to_cpu_sync(device, queue, albedo_tex);
        albedo_buf.reserve_exact(cpu_albedo.len() / 4 * 3);
        for i in (0..cpu_albedo.len()).step_by(4) {
            albedo_buf.push(cpu_albedo[i+0] as f32 / 255.0); // R
            albedo_buf.push(cpu_albedo[i+1] as f32 / 255.0); // G
            albedo_buf.push(cpu_albedo[i+2] as f32 / 255.0); // B
        }
    }

    let mut normals_buf = Vec::<f32>::new();
    if let Some(normals_tex) = normals
    {
        let cpu_normals = copy_texture_to_cpu_sync(device, queue, normals_tex);
        normals_buf.reserve_exact(cpu_normals.len() / 4 * 3);
        for i in (0..cpu_normals.len()).step_by(4) {
            normals_buf.push(cpu_normals[i+0] as f32 / 255.0); // R
            normals_buf.push(cpu_normals[i+1] as f32 / 255.0); // G
            normals_buf.push(cpu_normals[i+2] as f32 / 255.0); // B
        }
    }

    // Denoise.

    let denoised = beauty;

/*
    let oidn_device = oidn::Device::new();
    let mut filter = oidn::RayTracing::new(&oidn_device)
        .image_dimensions(1920, 1080)
        .clean_aux(true);
        //.filter();
        //.expect("Filter config error.");

    if let Some(albedo_tex) = albedo {
        if let Some(normals_tex) = normals {
            filter = filter.albedo_normal(
        } else {
            filter = filter.albedo
        }
    }

    if let Err(e) = oidn_device.get_error() {
        println!("Error denoising image: {}", e.1);
    }
*/

    // Upload image result back to the GPU.
    upload_rgbf32_texture_to_gpu(device, queue, target, denoised);
}
*/

// Returns an array in the rgbf32 format, interleaved and with no extra padding.
/*
fn copy_texture_to_cpu_sync(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<f32>
{
    let width = texture.size().width;
    let height = texture.size().height;
    let bytes_per_pixel = 8; // TODO: Only supporting rgbaf16 for now.
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = align_to(unpadded_bytes_per_row, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Copy Encoder"),
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    // Block until the GPU is done
    device.poll(wgpu::Maintain::Wait);

    // Map synchronously
    let buffer_slice = output_buffer.slice(..);

    let mapping_result = std::sync::Arc::new(std::sync::Mutex::new(None));
    let c_mapping_result = mapping_result.clone();

    buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
        *c_mapping_result.lock().unwrap() = Some(v);
    });

    device.poll(wgpu::Maintain::Wait); // Wait until the mapping is done

    let map_status = loop {
        if let Some(result) = mapping_result.lock().unwrap().take() {
            break result;
        }
        std::thread::sleep(std::time::Duration::from_millis(1));
        device.poll(wgpu::Maintain::Poll);
    };

    map_status.expect("Buffer mapping failed");

    let data = buffer_slice.get_mapped_range();
    let mut result = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);

    for chunk in data.chunks(padded_bytes_per_row as usize) {
        result.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
    }

    drop(data);
    output_buffer.unmap();

    return result;

    fn align_to(value: u32, alignment: u32) -> u32
    {
        return (value + alignment - 1) / alignment * alignment;
    }
}
*/

/*
fn upload_rgbf32_texture_to_gpu(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture, input: Vec<f32>)
{
    let width  = target.size().width;
    let height = target.size().height;
    assert_eq!(input.len(), (width * height * 3) as usize);

    // Convert RGB to RGBA by adding an alpha channel (typically 1.0)
    let mut rgba_data = Vec::<f32>::with_capacity((width * height * 4) as usize);
    for i in 0..(width * height) as usize {
        rgba_data.push(input[i * 3 + 0]);
        rgba_data.push(input[i * 3 + 1]);
        rgba_data.push(input[i * 3 + 2]);
        rgba_data.push(1.0); // Alpha
    }

    let bytes: &[u8] = to_u8_slice(&rgba_data);

    let bytes_per_row = width * 4 * std::mem::size_of::<f32>() as u32;
    let layout = wgpu::TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(bytes_per_row),
        rows_per_image: Some(width),
    };

    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        layout,
        extent,
    );
}
*/
