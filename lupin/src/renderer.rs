
use crate::base::*;
use crate::wgpu_utils::*;

use wgpu::util::DeviceExt;  // For some extra device traits.

#[allow(unused_macros)]
macro_rules! static_assert {
    ($($tt:tt)*) => {
        const _: () = assert!($($tt)*);
    }
}

pub static DEFAULT_PATHTRACER_SRC: &str = include_str!("shaders/pathtracer.wgsl");
pub static PATHTRACER_BVH_RT_SRC: &str = include_str!("shaders/bvh_rt.wgsl");
pub static PATHTRACER_BVH_CUSTOM_SRC: &str = include_str!("shaders/bvh_custom.wgsl");
pub static SENTINEL_IDX: u32 = u32::MAX;

pub struct Scene
{
    // Meshes
    pub mesh_infos: wgpu::Buffer,
    /// Mandatory for all meshes.
    pub verts_pos_array: Vec<wgpu::Buffer>,
    /// Mandatory for all meshes.
    pub indices_array: Vec<wgpu::Buffer>,
    /// Optional.
    pub verts_normal_array: Vec<wgpu::Buffer>,
    /// Optional.
    pub verts_texcoord_array: Vec<wgpu::Buffer>,
    /// Optional.
    pub verts_color_array: Vec<wgpu::Buffer>,

    pub instances: wgpu::Buffer,
    pub materials: wgpu::Buffer,

    pub textures: Vec<wgpu::Texture>,
    pub samplers: Vec<wgpu::Sampler>,
    pub environments: wgpu::Buffer,

    // Auxiliary data structures.
    pub tlas_nodes: wgpu::Buffer,
    pub bvh_nodes_array: Vec<wgpu::Buffer>,
    pub lights: Lights,
    pub rt_tlas: Option<wgpu::Tlas>,
    pub rt_blases: Vec<wgpu::Blas>,
}

#[derive(Default, Debug)]
pub struct SceneCPU
{
    pub mesh_infos: Vec<MeshInfo>,
    pub verts_pos_array: Vec<Vec<Vec4>>,
    pub verts_normal_array: Vec<Vec<Vec4>>,
    pub verts_texcoord_array: Vec<Vec<Vec2>>,
    pub verts_color_array: Vec<Vec<Vec4>>,
    pub indices_array: Vec<Vec<u32>>,

    pub instances: Vec<Instance>,
    pub materials: Vec<Material>,

    pub environments: Vec<Environment>,
}

#[derive(Default, Debug)]
pub struct LightsCPU
{
    pub lights: Vec<Light>,
    pub alias_tables: Vec<Vec<AliasBin>>,
    pub env_alias_tables: Vec<Vec<AliasBin>>,
}

#[derive(Debug)]
pub struct Lights
{
    pub lights: wgpu::Buffer,
    pub alias_tables: Vec<wgpu::Buffer>,
    pub env_alias_tables: Vec<wgpu::Buffer>,
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct MeshInfo
{
    pub normals_buf_idx: u32,
    pub texcoords_buf_idx: u32,
    pub colors_buf_idx: u32,
}

impl Default for MeshInfo
{
    fn default() -> Self
    {
        return Self {
            normals_buf_idx: SENTINEL_IDX,
            texcoords_buf_idx: SENTINEL_IDX,
            colors_buf_idx: SENTINEL_IDX,
        };
    }
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Instance
{
    pub transpose_inverse_transform: Mat4x3,
    pub mesh_idx: u32,
    pub mat_idx: u32,
    pub _padding0: f32,
    pub _padding1: f32,
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
    //Subsurface  = 5,
    //Volumetric  = 6,
    GltfPbr     = 7,
}

#[derive(Clone, Copy, Debug)]
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

impl Default for Material
{
    fn default() -> Self
    {
        return Self {
            color: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            emission: Vec4::default(),
            scattering: Vec4::default(),
            mat_type: MaterialType::Matte,
            roughness: 0.0,
            metallic: 0.0,
            ior: 1.5,
            sc_anisotropy: 0.0,
            tr_depth: 0.01,
            color_tex_idx: SENTINEL_IDX,
            emission_tex_idx: SENTINEL_IDX,
            roughness_tex_idx: SENTINEL_IDX,
            scattering_tex_idx: SENTINEL_IDX,
            normal_tex_idx: SENTINEL_IDX,
            padding0: 0,
        };
    }
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Environment
{
    pub emission: Vec3,
    pub emission_tex_idx: u32,
    pub transform: Mat4,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Light
{
    pub instance_idx: u32,
    pub area: f32,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct AliasBin
{
    pub prob: f32,
    pub alias_threshold: f32,  // [0, 1]; should select the alias if rnd_f32 > alias
    pub alias: u32,
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
    pub left: u32,  // If it's 0, this node is a leaf
    pub aabb_max: Vec3,
    pub instance_idx: u32,
    pub right: u32,
    pub _padding0: Vec3,
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct PushConstants
{
    pub camera_transform: Mat4,
    pub camera_lens: f32,
    pub camera_film: f32,
    pub camera_aspect: f32,
    pub camera_focus: f32,
    pub camera_aperture: f32,

    pub flags: u32,

    pub id_offset: [u32; 2],
    pub accum_counter: u32,

    // Debug params
    pub heatmap_min: f32,
    pub heatmap_max: f32,

    pub falsecolor_type: u32,
    pub pathtrace_type: u32,

    pub max_radiance: f32,
}

static_assert!(std::mem::size_of::<PushConstants>() < MAX_PUSH_CONSTANTS_SIZE as usize);

// Shader behavior flags
// NOTE: Coupled to shader.
pub const FLAG_CAMERA_ORTHO: u32         = 1 << 0;
pub const FLAG_ENVS_EMPTY: u32           = 1 << 1;
pub const FLAG_LIGHTS_EMPTY: u32         = 1 << 2;
pub const FLAG_DEBUG_TRI_CHECKS:  u32    = 1 << 3;
pub const FLAG_DEBUG_AABB_CHECKS: u32    = 1 << 4;
pub const FLAG_DEBUG_NUM_BOUNCES: u32    = 1 << 5;
pub const FLAG_DEBUG_FIRST_HIT_ONLY: u32 = 1 << 6;

// Constants
pub const MAX_PUSH_CONSTANTS_SIZE: u32 = 128;
// NOTE: Coupled to shader.
pub const BVH_MAX_DEPTH: i32 = 25;
// NOTE: Coupled to shader.
pub const TLAS_MAX_DEPTH: i32 = 50;
pub const MAX_MESHES:    u32 = 15000;
pub const MAX_ENVS:      u32 = 10;
pub const MAX_TEXTURES:  u32 = 15000;
pub const NUM_STORAGE_BUFFERS_PER_MESH: u32 = 7;
// NOTE: Coupled to shader.
pub const WORKGROUP_SIZE: u32 = 4;
const RT_MAX_ACCEL_STRUCTURES: u32 = 2;

/// Requests a WGPU device with the required features for Lupin.
pub fn request_device_for_lupin(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue)
{
    let optional_features = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    let mut supported_optional_features = optional_features.intersection(adapter.features());
    let force_swrt = cfg!(feature = "force-swrt");
    if force_swrt { supported_optional_features.remove(wgpu::Features::EXPERIMENTAL_RAY_QUERY); }

    let supports_rt = supported_optional_features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    let allowed_accel_structures = if supports_rt { RT_MAX_ACCEL_STRUCTURES } else { 0 };
    let max_rt_instances = if supports_rt { 1000000 } else { 0 };
    let max_blas_geometry_count = if supports_rt { 1 } else { 0 };
    let max_blas_primitives = if supports_rt { 10000000 } else { 0 };

    // The main feature we need is the possibility to define arrays of buffer,
    // texture and sampler bindings, and to index them with non-uniform values.
    let desc = wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY |
                           wgpu::Features::BUFFER_BINDING_ARRAY  |
                           wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY |
                           wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING |
                           wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY |
                           wgpu::Features::PUSH_CONSTANTS |
                           supported_optional_features,
        required_limits: wgpu::Limits {
            max_storage_buffers_per_shader_stage: MAX_MESHES * NUM_STORAGE_BUFFERS_PER_MESH + MAX_ENVS + 64,
            max_binding_array_elements_per_shader_stage: MAX_MESHES * NUM_STORAGE_BUFFERS_PER_MESH + MAX_ENVS + MAX_TEXTURES + 64,
            max_binding_array_sampler_elements_per_shader_stage: MAX_TEXTURES,
            max_sampled_textures_per_shader_stage: MAX_TEXTURES + 64,
            max_samplers_per_shader_stage: MAX_TEXTURES + 8,
            max_push_constant_size: MAX_PUSH_CONSTANTS_SIZE,
            max_acceleration_structures_per_shader_stage: allowed_accel_structures,
            max_tlas_instance_count: max_rt_instances,
            max_blas_geometry_count: max_blas_geometry_count,
            max_blas_primitive_count: max_blas_primitives,
            ..Default::default()
        },
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        memory_hints: Default::default(),
        trace: Default::default(),
    };

    let (device, queue) = wait_for(adapter.request_device(&desc)).expect("Failed to get device");
    return (device, queue);
}

#[cfg(feature = "denoising")]
pub enum DenoiseDevice
{
    InteropDevice(oidn_wgpu_interop::Device),
    OidnDevice(oidn::Device),
}

impl DenoiseDevice
{
    pub fn oidn_device(&self) -> &oidn::Device
    {
        return match self {
            DenoiseDevice::InteropDevice(interop) => interop.oidn_device(),
            DenoiseDevice::OidnDevice(device) => &device,
        }
    }
}

/// Requests a WGPU device with the required features for Lupin and for denoising.
#[cfg(feature = "denoising")]
pub fn request_device_for_lupin_with_denoising_capabilities(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue, DenoiseDevice)
{
    let optional_features = wgpu::Features::EXPERIMENTAL_RAY_QUERY;
    let mut supported_optional_features = optional_features.intersection(adapter.features());
    let force_swrt = cfg!(feature = "force-swrt");
    if force_swrt { supported_optional_features.remove(wgpu::Features::EXPERIMENTAL_RAY_QUERY); }

    let supports_rt = supported_optional_features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
    let allowed_accel_structures = if supports_rt { RT_MAX_ACCEL_STRUCTURES } else { 0 };
    let max_rt_instances = if supports_rt { 1000000 } else { 0 };
    let max_blas_geometry_count = if supports_rt { 1 } else { 0 };
    let max_blas_primitives = if supports_rt { 10000000 } else { 0 };

    // The main feature we need is the possibility to define arrays of buffer,
    // texture and sampler bindings, and to index them with non-uniform values.
    let desc = wgpu::DeviceDescriptor {
        label: None,
        required_features: wgpu::Features::TEXTURE_BINDING_ARRAY |
                           wgpu::Features::BUFFER_BINDING_ARRAY  |
                           wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY |
                           wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING |
                           wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY |
                           wgpu::Features::PUSH_CONSTANTS |
                           supported_optional_features,
        required_limits: wgpu::Limits {
            max_storage_buffers_per_shader_stage: MAX_MESHES * NUM_STORAGE_BUFFERS_PER_MESH + MAX_ENVS + 64,
            max_binding_array_elements_per_shader_stage: MAX_MESHES * NUM_STORAGE_BUFFERS_PER_MESH + MAX_ENVS + MAX_TEXTURES + 64,
            max_binding_array_sampler_elements_per_shader_stage: MAX_TEXTURES,
            max_sampled_textures_per_shader_stage: MAX_TEXTURES + 64,
            max_samplers_per_shader_stage: MAX_TEXTURES + 8,
            max_push_constant_size: MAX_PUSH_CONSTANTS_SIZE,
            max_acceleration_structures_per_shader_stage: allowed_accel_structures,
            max_tlas_instance_count: max_rt_instances,
            max_blas_geometry_count: max_blas_geometry_count,
            max_blas_primitive_count: max_blas_primitives,
            ..Default::default()
        },
        experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
        memory_hints: Default::default(),
        trace: Default::default(),
    };

    let disable_shared_device = cfg!(feature = "denoise-force-disable-shared-device");
    let res = wait_for(oidn_wgpu_interop::Device::new(adapter, &desc));
    if matches!(res, Ok(_)) && !disable_shared_device
    {
        let res = res.unwrap();
        let (device, queue) = res;
        return (device.wgpu_device().clone(), queue, DenoiseDevice::InteropDevice(device));
    }
    else
    {
        let (device, queue) = wait_for(adapter.request_device(&desc)).expect("Failed to get device");
        let oidn_device = oidn::Device::new();
        return (device, queue, DenoiseDevice::OidnDevice(oidn_device));
    }
}

// Shader params

pub struct PathtracePipeline
{
    custom: wgpu::ComputePipeline,
    rt: Option<wgpu::ComputePipeline>,
}

pub struct PathtraceResources
{
    pub pipeline: PathtracePipeline,
    pub falsecolor_pipeline: PathtracePipeline,
    pub debug_pipeline: PathtracePipeline,
    pub dummy_prev_frame_texture: wgpu::Texture,

    // NOTE: Hack to guard for 0 size buffers.
    // WGPU doesn't allow 0 size buffers and
    // 0 length arrays of bindings. (Sigh...)
    pub dummy_buf_mesh_infos: wgpu::Buffer,
    pub dummy_buf_vertpos: wgpu::Buffer,
    pub dummy_buf_vertnormal: wgpu::Buffer,
    pub dummy_buf_vertuv: wgpu::Buffer,
    pub dummy_buf_vertcolor: wgpu::Buffer,
    pub dummy_buf_idx: wgpu::Buffer,
    pub dummy_buf_bvh_node: wgpu::Buffer,
    pub dummy_scene_tex: wgpu::Texture,
    pub dummy_scene_sampler: wgpu::Sampler,
    pub dummy_buf_alias_bin: wgpu::Buffer,
    pub dummy_buf_tlas: wgpu::Buffer,
    pub dummy_buf_instance: wgpu::Buffer,
    pub dummy_buf_material: wgpu::Buffer,
    pub dummy_buf_environment: wgpu::Buffer,
    pub dummy_buf_light: wgpu::Buffer,
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
            max_bounces: 8,
            samples_per_pixel: 5,
        };
    }
}

pub fn build_pathtrace_resources(device: &wgpu::Device, baked_pathtrace_params: &BakedPathtraceParams) -> PathtraceResources
{
    let mut shader_src_custom = String::from(DEFAULT_PATHTRACER_SRC);
    shader_src_custom.push_str(PATHTRACER_BVH_CUSTOM_SRC);
    let mut shader_src_rt = String::from(DEFAULT_PATHTRACER_SRC);
    shader_src_rt.push_str(PATHTRACER_BVH_RT_SRC);

    let shader_desc_custom = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Pathtracer Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src_custom.into())
    };
    let shader_desc_rt = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Pathtracer Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_src_rt.into())
    };

    let shader_custom;
    if baked_pathtrace_params.with_runtime_checks {
        shader_custom = device.create_shader_module(shader_desc_custom);
    } else {
        shader_custom = unsafe { device.create_shader_module_trusted(shader_desc_custom, wgpu::ShaderRuntimeChecks::unchecked()) };
    }
    let shader_rt;
    if !supports_rt(device) {
        shader_rt = None;
    } else if baked_pathtrace_params.with_runtime_checks {
        shader_rt = Some(device.create_shader_module(shader_desc_rt));
    } else {
        shader_rt = Some( unsafe { device.create_shader_module_trusted(shader_desc_rt, wgpu::ShaderRuntimeChecks::unchecked()) } );
    }

    let scene_bindgroup_layout = create_pathtracer_scene_bindgroup_layout(device);
    let settings_bindgroup_layout = create_pathtracer_settings_bindgroup_layout(device);
    let render_target_bindgroup_layout = create_pathtracer_output_bindgroup_layout(device);
    let bvh_custom_bindgroup_layout = create_pathtracer_bvh_bindgroup_layout(device, true);

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lupin Pathtracer Pipeline Layout"),
        bind_group_layouts: &[
            &scene_bindgroup_layout,
            &settings_bindgroup_layout,
            &render_target_bindgroup_layout,
            &bvh_custom_bindgroup_layout,
        ],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..std::mem::size_of::<PushConstants>() as u32,
        }],
    });

    let constants = [
        ("MAX_BOUNCES", baked_pathtrace_params.max_bounces as f64),
        ("SAMPLES_PER_PIXEL", baked_pathtrace_params.samples_per_pixel as f64),
    ];

    let debug_constants = [
        ("MAX_BOUNCES", baked_pathtrace_params.max_bounces as f64),
        ("SAMPLES_PER_PIXEL", baked_pathtrace_params.samples_per_pixel as f64),
        ("DEBUG", 1 as f64),
    ];

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_custom,
        entry_point: Some("pathtrace_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    let falsecolor_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Falsecolor Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_custom,
        entry_point: Some("pathtrace_falsecolor_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    let debug_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Lupin Pathtracer Debug Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader_custom,
        entry_point: Some("pathtrace_debug_main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &debug_constants,
            zero_initialize_workgroup_memory: true,
        },
        cache: None,
    });

    // RT Variants
    let mut pipeline_rt = None;
    let mut falsecolor_pipeline_rt = None;
    let mut debug_pipeline_rt = None;
    if let Some(shader_rt) = shader_rt
    {
        let bvh_rt_bindgroup_layout = create_pathtracer_bvh_bindgroup_layout(device, false);
        let pipeline_layout_rt = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Lupin Pathtracer Pipeline Layout (RT)"),
            bind_group_layouts: &[
                &scene_bindgroup_layout,
                &settings_bindgroup_layout,
                &render_target_bindgroup_layout,
                &bvh_rt_bindgroup_layout,
            ],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..std::mem::size_of::<PushConstants>() as u32,
            }],
        });

        pipeline_rt = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Lupin Pathtracer Pipeline"),
            layout: Some(&pipeline_layout_rt),
            module: &shader_rt,
            entry_point: Some("pathtrace_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        }));

        falsecolor_pipeline_rt = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Lupin Pathtracer Falsecolor Pipeline"),
            layout: Some(&pipeline_layout_rt),
            module: &shader_rt,
            entry_point: Some("pathtrace_falsecolor_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        }));

        debug_pipeline_rt = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Lupin Pathtracer Debug Pipeline"),
            layout: Some(&pipeline_layout_rt),
            module: &shader_rt,
            entry_point: Some("pathtrace_debug_main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &debug_constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        }));
    }

    let size = wgpu::Extent3d {
        width: 1,
        height: 1,
        depth_or_array_layers: 1,
    };

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

    let dummy_scene_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Scene Texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Snorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });

    return PathtraceResources {
        pipeline: PathtracePipeline { custom: pipeline, rt: pipeline_rt },
        debug_pipeline: PathtracePipeline { custom: debug_pipeline, rt: debug_pipeline_rt },
        falsecolor_pipeline: PathtracePipeline { custom: falsecolor_pipeline, rt: falsecolor_pipeline_rt },
        dummy_prev_frame_texture,

        // NOTE: Hack to guard for 0 size buffers.
        // WGPU doesn't allow 0 size bindings and
        // 0 length arrays of bindings. (Sigh...)
        dummy_buf_mesh_infos: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<MeshInfo>(), "dummy_buf_meshinfos"),
        dummy_buf_vertpos: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Vec4>(), "dummy_buf_vertpos"),
        dummy_buf_vertnormal: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Vec4>(), "dummy_buf_vertnormal"),
        dummy_buf_vertuv: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Vec2>(), "dummy_buf_vertuv"),
        dummy_buf_vertcolor: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Vec4>(), "dummy_buf_vertcolor"),
        dummy_buf_idx: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<u32>(), "dummy_buf_idx"),
        dummy_buf_bvh_node: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<BvhNode>(), "dummy_buf_bvh_node"),
        dummy_scene_tex,
        dummy_scene_sampler: create_linear_sampler(device),
        dummy_buf_alias_bin: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<AliasBin>(), "dummy_buf_alias_bin"),
        dummy_buf_tlas: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<TlasNode>(), "dummy_buf_tlas"),
        dummy_buf_instance: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Instance>(), "dummy_buf_instance"),
        dummy_buf_material: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Material>(), "dummy_buf_material"),
        dummy_buf_environment: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Environment>(), "dummy_buf_environment"),
        dummy_buf_light: create_storage_buffer_with_size_and_name(device, std::mem::size_of::<Light>(), "dummy_buf_light"),
    };
}

#[derive(Copy, Clone, Debug)]
pub struct AccumulationParams<'a>
{
    pub prev_frame: &'a wgpu::Texture,
    pub accum_counter: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct TileParams
{
    /// In number of GPU workgroups.
    pub tile_size: u32,
    /// Current tile index. Must be
    /// explicitly incremented by the user.
    pub tile_idx: u32,
}

impl Default for TileParams
{
    fn default() -> Self
    {
        return Self {
            tile_size: 100,
            tile_idx: 0,
        };
    }
}

/// Computes the total number of tiles that will be required
/// to complete a single accumulation frame for a given texture size.
/// * `tile_size` - In GPU workgroups, just like in TileParams.
pub fn get_num_tiles(tile_size: u32, width: u32, height: u32) -> u32
{
    let num_tiles_x = (u32::max(1, width) - 1)  / (tile_size * WORKGROUP_SIZE) + 1;
    let num_tiles_y = (u32::max(1, height) - 1) / (tile_size * WORKGROUP_SIZE) + 1;
    let total_tiles = num_tiles_x * num_tiles_y;
    return total_tiles;
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CameraParams
{
    pub is_orthographic: bool,

    pub lens: f32,
    pub film: f32,
    pub aspect: f32,
    pub focus: f32,
    pub aperture: f32,
}

impl Default for CameraParams
{
    fn default() -> Self
    {
        return Self {
            is_orthographic: false,
            lens:     0.050,
            film:     0.036,
            aspect:   1.500,
            focus:    10000.0,
            aperture: 0.0,
        };
    }
}

// NOTE: Coupled to shader.
#[repr(u32)]
#[derive(Default, Copy, Clone, Debug, PartialEq)]
pub enum PathtraceType
{
    /// This is the "poor man's MIS". It's a bit slower than
    /// proper MIS but handles more cases. It can exhibit firefly
    /// artifacts, which is the main reason for clamp_radiance().
    #[default]
    Standard = 0,
    /// Classic MIS. The fastest converging option, but a bit more
    /// prone to artifacts.
    MIS = 1,
    /// BSDF-sampling only. This is only viable in very specific scenes.
    /// Does not exhibit any form of firefly artifacts, but it may take a
    /// very long time to converge to the reference. Generally very slow.
    Naive = 2,
    /// BSDF-sampling coupled with a light ray to check for light occlusion.
    Direct = 3,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct AdvancedParams
{
    pub max_radiance: f32,
}

impl Default for AdvancedParams
{
    fn default() -> Self
    {
        return Self {
            max_radiance: 100.0,
        };
    }
}

#[derive(Default, Clone, Copy)]
pub struct PathtraceDesc<'a>
{
    /// Output texture to store the result. This must be different from
    /// accum_params.prev_frame, due to underlying graphics API limitations.
    pub accum_params: Option<AccumulationParams<'a>>,
    /// For tiled rendering. This makes it possible to break up a single pathtrace
    /// call. Useful when your scene is particularly big and you have real-time things
    /// running in your application (e.g. a GUI). For big scenes this is recommended,
    /// because if the shader takes too long to execute most OSs will perform a driver reset.
    pub tile_params: Option<&'a TileParams>,
    pub camera_params: CameraParams,
    pub camera_transform: Mat3x4,
    pub force_software_bvh: bool,
    pub advanced: AdvancedParams,
}

pub fn pathtrace_scene(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources, scene: &Scene, render_target: &wgpu::Texture, pathtrace_type: PathtraceType, desc: &PathtraceDesc)
{
    assert!(render_target.format() == wgpu::TextureFormat::Rgba16Float);

    let use_sw_rt = !supports_rt(device) || desc.force_software_bvh;

    let sw_bvh_absent = scene.tlas_nodes.size() <= 0 || scene.bvh_nodes_array.len() <= 0;
    if use_sw_rt && sw_bvh_absent {
        panic!("Software Raytracing is required or explicitly enabled, but no software BVH was built for this scene.");
    }

    let target_width = render_target.width();
    let target_height = render_target.height();
    let accum_params = desc.accum_params;

    let pipeline = if use_sw_rt {
        &resources.pipeline.custom
    } else {
        resources.pipeline.rt.as_ref().unwrap()
    };

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, resources, accum_params.map(|params| params.prev_frame));
    let output_bindgroup = create_pathtracer_output_bindgroup(device, resources, render_target);
    let bvh_bindgroup = create_pathtracer_bvh_bindgroup(device, scene, resources, use_sw_rt);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &scene_bindgroup, &[]);
        pass.set_bind_group(1, &settings_bindgroup, &[]);
        pass.set_bind_group(2, &output_bindgroup, &[]);
        pass.set_bind_group(3, &bvh_bindgroup, &[]);

        if let Some(tile_params) = desc.tile_params  // Tiled rendering.
        {
            let tile_idx = tile_params.tile_idx;
            let tile_size = tile_params.tile_size;
            let num_tiles_x = (u32::max(1, target_width) - 1)  / (tile_size * WORKGROUP_SIZE) + 1;
            let num_tiles_y = (u32::max(1, target_height) - 1) / (tile_size * WORKGROUP_SIZE) + 1;
            let total_tiles = num_tiles_x * num_tiles_y;
            assert!(tile_idx < total_tiles, "tile_idx out of range!");

            let tiles_to_render = 1;
            for i in tile_idx..u32::min(tile_idx + tiles_to_render, total_tiles)
            {
                let offset_x = (i % num_tiles_x) * tile_size * WORKGROUP_SIZE;
                let offset_y = (i / num_tiles_x) * tile_size * WORKGROUP_SIZE;

                let push_constants = get_push_constants_tiled(desc, scene, Some(pathtrace_type), None, None, num_tiles_x, tile_size, i);
                pass.set_push_constants(0, to_u8_slice(&[push_constants]));

                let num_workers_x = u32::min(tile_size, (target_width - offset_x) / WORKGROUP_SIZE);
                let num_workers_y = u32::min(tile_size, (target_height - offset_y) / WORKGROUP_SIZE);
                pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
            }
        }
        else  // Full-screen rendering.
        {
            let push_constants = get_push_constants(desc, scene, Some(pathtrace_type), None, None);
            pass.set_push_constants(0, to_u8_slice(&[push_constants]));

            let num_workers_x = (render_target.width() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let num_workers_y = (render_target.height() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
        }
    }

    queue.submit(Some(encoder.finish()));
}

// NOTE: Coupled to shader code.
#[repr(u32)]
#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub enum FalsecolorType
{
    #[default]
    /// Albedo GBuffer, useful for denoising. Stored as linear, not SRGB.
    Albedo = 0,
    /// Normal GBuffer, useful for denoising. Requires rgba8_snorm.
    Normals = 1,
    /// Normal GBuffer, but this op is performed on each pixel: color = normal * 0.5 + 0.5.
    NormalsUnsigned = 2,
    /// 0 (black) if back-facing, 1 (white) if front-facing.
    FrontFacing = 3,
    Emission = 4,
    Roughness = 5,
    Metallic = 6,
    Opacity = 7,
    /// Each material will be given its own (hashed) color.
    MatType = 8,
    /// 1 (white) for delta materials, 0 (black) for everything else.
    IsDelta = 9,
    /// Each instance will be given its own (hashed) color.
    Instance = 10,
    /// Each triangle will be given its own (hashed) color.
    Tri = 11,
}

pub fn pathtrace_scene_falsecolor(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources,
                                  scene: &Scene, render_target: &wgpu::Texture, falsecolor_type: FalsecolorType,
                                  desc: &PathtraceDesc)
{
    assert!(render_target.format() == wgpu::TextureFormat::Rgba16Float);

    let use_sw_rt = !supports_rt(device) || desc.force_software_bvh;

    let sw_bvh_absent = scene.tlas_nodes.size() <= 0 || scene.bvh_nodes_array.len() <= 0;
    if use_sw_rt && sw_bvh_absent {
        panic!("Software Raytracing is required or explicitly enabled, but no software BVH was built for this scene.");
    }

    let target_width = render_target.width();
    let target_height = render_target.height();
    let accum_params = desc.accum_params;

    let pipeline = if use_sw_rt {
        &resources.falsecolor_pipeline.custom
    } else {
        resources.falsecolor_pipeline.rt.as_ref().unwrap()
    };

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, resources, accum_params.map(|params| params.prev_frame));
    let output_bindgroup = create_pathtracer_output_bindgroup(device, resources, render_target);
    let bvh_bindgroup = create_pathtracer_bvh_bindgroup(device, scene, resources, use_sw_rt);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &scene_bindgroup, &[]);
        pass.set_bind_group(1, &settings_bindgroup, &[]);
        pass.set_bind_group(2, &output_bindgroup, &[]);
        pass.set_bind_group(3, &bvh_bindgroup, &[]);

        if let Some(tile_params) = desc.tile_params  // Tiled rendering.
        {
            let tile_idx = tile_params.tile_idx;
            let tile_size = tile_params.tile_size;
            let num_tiles_x = (u32::max(1, target_width) - 1)  / (tile_size * WORKGROUP_SIZE) + 1;
            let num_tiles_y = (u32::max(1, target_height) - 1) / (tile_size * WORKGROUP_SIZE) + 1;
            let total_tiles = num_tiles_x * num_tiles_y;
            assert!(tile_idx < total_tiles, "tile_idx out of range!");

            let tiles_to_render = 1;
            for i in tile_idx..u32::min(tile_idx + tiles_to_render, total_tiles)
            {
                let offset_x = (i % num_tiles_x) * tile_size * WORKGROUP_SIZE;
                let offset_y = (i / num_tiles_x) * tile_size * WORKGROUP_SIZE;

                let push_constants = get_push_constants_tiled(desc, scene, None, Some(falsecolor_type), None, num_tiles_x, tile_size, i);
                pass.set_push_constants(0, to_u8_slice(&[push_constants]));

                let num_workers_x = u32::min(tile_size, (target_width - offset_x) / WORKGROUP_SIZE);
                let num_workers_y = u32::min(tile_size, (target_height - offset_y) / WORKGROUP_SIZE);
                pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
            }
        }
        else  // Full-screen rendering.
        {
            let push_constants = get_push_constants(desc, scene, None, Some(falsecolor_type), None);
            pass.set_push_constants(0, to_u8_slice(&[push_constants]));

            let num_workers_x = (render_target.width() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let num_workers_y = (render_target.height() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
        }
    }

    queue.submit(Some(encoder.finish()));
}

#[derive(Copy, Clone, Debug, PartialEq)]
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

pub fn pathtrace_scene_debug(device: &wgpu::Device, queue: &wgpu::Queue, resources: &PathtraceResources, scene: &Scene, render_target: &wgpu::Texture, debug_desc: &DebugVizDesc, desc: &PathtraceDesc)
{
    assert!(render_target.format() == wgpu::TextureFormat::Rgba16Float);

    let use_sw_rt = !supports_rt(device) || desc.force_software_bvh;

    let sw_bvh_absent = scene.tlas_nodes.size() <= 0 || scene.bvh_nodes_array.len() <= 0;
    if use_sw_rt && sw_bvh_absent {
        panic!("Software Raytracing is required or explicitly enabled, but no software BVH was built for this scene.");
    }

    let target_width = render_target.width();
    let target_height = render_target.height();
    let accum_params = desc.accum_params;

    let pipeline = if use_sw_rt {
        &resources.debug_pipeline.custom
    } else {
        resources.debug_pipeline.rt.as_ref().unwrap()
    };

    let scene_bindgroup = create_pathtracer_scene_bindgroup(device, resources, scene);
    let settings_bindgroup = create_pathtracer_settings_bindgroup(device, resources, accum_params.map(|params| params.prev_frame));
    let output_bindgroup = create_pathtracer_output_bindgroup(device, resources, render_target);
    let bvh_bindgroup = create_pathtracer_bvh_bindgroup(device, scene, resources, use_sw_rt);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &scene_bindgroup, &[]);
        pass.set_bind_group(1, &settings_bindgroup, &[]);
        pass.set_bind_group(2, &output_bindgroup, &[]);
        pass.set_bind_group(3, &bvh_bindgroup, &[]);

        if let Some(tile_params) = desc.tile_params  // Tiled rendering.
        {
            let tile_idx = tile_params.tile_idx;
            let tile_size = tile_params.tile_size;
            let num_tiles_x = (u32::max(1, target_width) - 1)  / (tile_size * WORKGROUP_SIZE) + 1;
            let num_tiles_y = (u32::max(1, target_height) - 1) / (tile_size * WORKGROUP_SIZE) + 1;
            let total_tiles = num_tiles_x * num_tiles_y;
            assert!(tile_idx < total_tiles, "tile_idx out of range!");

            let tiles_to_render = 1;
            for i in tile_idx..u32::min(tile_idx + tiles_to_render, total_tiles)
            {
                let offset_x = (i % num_tiles_x) * tile_size * WORKGROUP_SIZE;
                let offset_y = (i / num_tiles_x) * tile_size * WORKGROUP_SIZE;

                let push_constants = get_push_constants_tiled(desc, scene, None, None, Some(debug_desc), num_tiles_x, tile_size, i);
                pass.set_push_constants(0, to_u8_slice(&[push_constants]));

                let num_workers_x = u32::min(tile_size, (target_width - offset_x) / WORKGROUP_SIZE);
                let num_workers_y = u32::min(tile_size, (target_height - offset_y) / WORKGROUP_SIZE);
                pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
            }
        }
        else  // Full-screen rendering.
        {
            let push_constants = get_push_constants(desc, scene, None, None, Some(debug_desc));
            pass.set_push_constants(0, to_u8_slice(&[push_constants]));

            let num_workers_x = (render_target.width() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            let num_workers_y = (render_target.height() + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
            pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
        }
    }

    queue.submit(Some(encoder.finish()));
}

// Wrappers

fn buffer_resource_nocheck(buffer: &wgpu::Buffer) -> wgpu::BindingResource
{
    return wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buffer, offset: 0, size: None });
}

fn buffer_resource<'a>(buffer: &'a wgpu::Buffer, dummy: &'a wgpu::Buffer) -> wgpu::BindingResource<'a>
{
    // NOTE: Arrays of bindings can't be empty.
    if buffer.size() <= 0 {
        return wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: dummy, offset: 0, size: None });
    }

    return wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buffer, offset: 0, size: None });
}

fn array_of_buffer_bindings_resource<'a, 'b>(buffers: &'a Vec<wgpu::Buffer>, dummy: &'a wgpu::Buffer) -> Vec<wgpu::BufferBinding<'a>>
{
    // NOTE: Arrays of bindings can't be empty.
    if buffers.is_empty()
    {
        // NOTE: Buffer bindings can't be 0 size either.
        let res = vec![wgpu::BufferBinding {
            buffer: &dummy,
            offset: 0,
            size: None,
        }];
        return res;
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

fn array_of_texture_bindings_resource<'a>(texture_views: &'a Vec<wgpu::TextureView>, dummy: &'a wgpu::TextureView) -> Vec<&'a wgpu::TextureView>
{
    // NOTE: Arrays of bindings can't be empty.
    if texture_views.is_empty() {
        return vec![dummy];
    }

    let mut bindings: Vec<&'a wgpu::TextureView> = Vec::with_capacity(texture_views.len());
    for i in 0..texture_views.len() {
        bindings.push(&texture_views[i]);
    }

    return bindings;
}

fn array_of_sampler_bindings_resource<'a>(samplers: &'a Vec<wgpu::Sampler>, dummy: &'a wgpu::Sampler) -> Vec<&'a wgpu::Sampler>
{
    // NOTE: Arrays of bindings can't be empty.
    if samplers.is_empty() {
        return vec![dummy];
    }

    let mut bindings: Vec<&'a wgpu::Sampler> = Vec::with_capacity(samplers.len());
    for i in 0..samplers.len() {
        bindings.push(&samplers[i]);
    }

    return bindings;
}

fn create_pathtracer_scene_bindgroup(device: &wgpu::Device, resources: &PathtraceResources, scene: &Scene) -> wgpu::BindGroup
{
    let verts_pos = &scene.verts_pos_array;
    let verts_normal = &scene.verts_normal_array;
    let verts_texcoord = &scene.verts_texcoord_array;
    let verts_color = &scene.verts_color_array;
    let indices = &scene.indices_array;

    // NOTE: Need to do a whole lot of work here to guard from 0 size buffers.
    // WGPU doesn't like 0 size bindings and 0 length binding arrays, but for
    // ergonomics of this library we'd like for them to just work.

    let texture_views = array_of_texture_views(&scene.textures);
    let dummy_tex_view = resources.dummy_scene_tex.create_view(&Default::default());

    use crate::wgpu_utils::*;
    let verts_pos_array = array_of_buffer_bindings_resource(&verts_pos, &resources.dummy_buf_vertpos);
    let verts_normal_array   = array_of_buffer_bindings_resource(&verts_normal, &resources.dummy_buf_vertnormal);
    let verts_texcoord_array = array_of_buffer_bindings_resource(&verts_texcoord, &resources.dummy_buf_vertuv);
    let verts_color_array    = array_of_buffer_bindings_resource(&verts_color, &resources.dummy_buf_vertcolor);
    let indices_array   = array_of_buffer_bindings_resource(&indices, &resources.dummy_buf_idx);
    let textures_array  = array_of_texture_bindings_resource(&texture_views, &dummy_tex_view);
    let samplers_array  = array_of_sampler_bindings_resource(&scene.samplers, &resources.dummy_scene_sampler);
    let alias_table_array     = array_of_buffer_bindings_resource(&scene.lights.alias_tables, &resources.dummy_buf_alias_bin);
    let env_alias_table_array = array_of_buffer_bindings_resource(&scene.lights.env_alias_tables, &resources.dummy_buf_alias_bin);

    let mesh_infos_buf = buffer_resource(&scene.mesh_infos, &resources.dummy_buf_mesh_infos);
    let instances_buf = buffer_resource(&scene.instances, &resources.dummy_buf_instance);
    let materials_buf = buffer_resource(&scene.materials, &resources.dummy_buf_material);
    let environments_buf = buffer_resource(&scene.environments, &resources.dummy_buf_environment);
    let lights_buf = buffer_resource(&&scene.lights.lights, &resources.dummy_buf_light);

    let scene_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Scene bind group"),
        layout: &resources.pipeline.custom.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0,  resource: mesh_infos_buf },
            wgpu::BindGroupEntry { binding: 1,  resource: wgpu::BindingResource::BufferArray(verts_pos_array.as_slice())       },
            wgpu::BindGroupEntry { binding: 2,  resource: wgpu::BindingResource::BufferArray(verts_normal_array.as_slice())    },
            wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::BufferArray(verts_texcoord_array.as_slice())  },
            wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::BufferArray(verts_color_array.as_slice())     },
            wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::BufferArray(indices_array.as_slice())         },
            wgpu::BindGroupEntry { binding: 6,  resource: instances_buf },
            wgpu::BindGroupEntry { binding: 7,  resource: materials_buf },
            wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureViewArray(textures_array.as_slice())   },
            wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::SamplerArray(samplers_array.as_slice())       },
            wgpu::BindGroupEntry { binding: 10, resource: environments_buf },
            wgpu::BindGroupEntry { binding: 11, resource: lights_buf },
            wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::BufferArray(alias_table_array.as_slice())     },
            wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::BufferArray(env_alias_table_array.as_slice()) },
        ]
    });

    return scene_bind_group;
}

fn create_pathtracer_scene_bindgroup_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout
{
    assert!(NUM_STORAGE_BUFFERS_PER_MESH == 7);

    return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: Some("Lupin Pathtracer Bindgroup Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {  // mesh_infos
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // verts_pos
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // verts_normal
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // verts_texcoord
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // verts_color
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // indices
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES)
            },
            wgpu::BindGroupLayoutEntry {  // instances
                binding: 6,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // materials
                binding: 7,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // textures
                binding: 8,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: std::num::NonZero::new(MAX_TEXTURES)
            },
            wgpu::BindGroupLayoutEntry {  // samplers
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: std::num::NonZero::new(MAX_TEXTURES)
            },
            wgpu::BindGroupLayoutEntry {  // environments
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // lights
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {  // alias_tables
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: std::num::NonZero::new(MAX_MESHES),
            },
            wgpu::BindGroupLayoutEntry {  // env_alias_tables
                binding: 13,
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

fn create_pathtracer_settings_bindgroup(device: &wgpu::Device, resources: &PathtraceResources, prev_frame: Option<&wgpu::Texture>) -> wgpu::BindGroup
{
    let prev_frame_view;
    if let Some(prev_frame) = prev_frame {
        prev_frame_view = prev_frame.create_view(&Default::default());
    } else {
        prev_frame_view = resources.dummy_prev_frame_texture.create_view(&Default::default());
    }

    let settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Settings bind group"),
        layout: &resources.pipeline.custom.get_bind_group_layout(1),
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

fn create_pathtracer_output_bindgroup(device: &wgpu::Device, resources: &PathtraceResources, output: &wgpu::Texture) -> wgpu::BindGroup
{
    let view = output.create_view(&Default::default());

    let render_target_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Pathtracer output bindgroup"),
        layout: &resources.pipeline.custom.get_bind_group_layout(2),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
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
            }
        ]
    });
}

fn create_pathtracer_bvh_bindgroup(device: &wgpu::Device, scene: &Scene, resources: &PathtraceResources, use_software_bvh: bool) -> wgpu::BindGroup
{
    if use_software_bvh
    {
        let bvh_nodes = &scene.bvh_nodes_array;
        let bvh_nodes_array = array_of_buffer_bindings_resource(&bvh_nodes, &resources.dummy_buf_bvh_node);
        let tlas_buf = buffer_resource(&scene.tlas_nodes, &resources.dummy_buf_tlas);

        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pathtracer BVH bindgroup (software)"),
            layout: &resources.pipeline.custom.get_bind_group_layout(3),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::BufferArray(bvh_nodes_array.as_slice()) },
                wgpu::BindGroupEntry { binding: 1, resource: tlas_buf },
            ]
        });
    }
    else
    {
        assert!(supports_rt(device));

        let layout = resources.pipeline.rt.as_ref().unwrap().get_bind_group_layout(3);
        let tlas = scene.rt_tlas.as_ref().unwrap();

        return device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pathtracer BVH bindgroup (hardware)"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tlas.as_binding() },
            ]
        });
    }
}

fn create_pathtracer_bvh_bindgroup_layout(device: &wgpu::Device, use_software_bvh: bool) -> wgpu::BindGroupLayout
{
    if use_software_bvh
    {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {  // bvh_nodes_array
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: std::num::NonZero::new(MAX_MESHES)
                },
                wgpu::BindGroupLayoutEntry {  // tlas_nodes
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ]
        });
    }
    else
    {
        return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {  // rt_tlas
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::AccelerationStructure { vertex_return: false },
                    count: None,
                },
            ]
        });
    }
}

fn get_push_constants(desc: &PathtraceDesc, scene: &Scene, pathtrace_type: Option<PathtraceType>, falsecolor_type: Option<FalsecolorType>, debug_desc: Option<&DebugVizDesc>) -> PushConstants
{
    let mut push_constants = PushConstants::default();

    if let Some(debug_desc) = debug_desc
    {
        match debug_desc.viz_type
        {
            DebugVizType::BVHAABBChecks =>
            {
                push_constants.flags |= FLAG_DEBUG_AABB_CHECKS;
            },
            DebugVizType::BVHTriChecks =>
            {
                push_constants.flags |= FLAG_DEBUG_TRI_CHECKS;
            },
            DebugVizType::NumBounces =>
            {
                push_constants.flags |= FLAG_DEBUG_NUM_BOUNCES;
            },
        }

        if debug_desc.first_hit_only {
            push_constants.flags |= FLAG_DEBUG_FIRST_HIT_ONLY
        }

        push_constants.heatmap_min = debug_desc.heatmap_min;
        push_constants.heatmap_max = debug_desc.heatmap_max;
    }

    if desc.camera_params.is_orthographic { push_constants.flags |= FLAG_CAMERA_ORTHO; }

    push_constants.camera_transform = desc.camera_transform.to_mat4();
    push_constants.camera_lens = desc.camera_params.lens;
    push_constants.camera_film = desc.camera_params.film;
    push_constants.camera_aspect = desc.camera_params.aspect;
    push_constants.camera_focus = desc.camera_params.focus;
    push_constants.camera_aperture = desc.camera_params.aperture;

    if scene.environments.size() <= 0 {
        push_constants.flags |= FLAG_ENVS_EMPTY;
    }
    if scene.lights.lights.size() <= 0 {
        push_constants.flags |= FLAG_LIGHTS_EMPTY;
    }

    if let Some(pathtrace_type) = pathtrace_type {
        push_constants.pathtrace_type = pathtrace_type as u32;
    }
    if let Some(falsecolor_type) = falsecolor_type {
        push_constants.falsecolor_type = falsecolor_type as u32;
    }

    if let Some(accum_params) = desc.accum_params {
        push_constants.accum_counter = accum_params.accum_counter;
    } else {
        push_constants.accum_counter = 0;
    }

    push_constants.max_radiance = desc.advanced.max_radiance;

    return push_constants;
}

fn get_push_constants_tiled(desc: &PathtraceDesc, scene: &Scene, pathtrace_type: Option<PathtraceType>, falsecolor_type: Option<FalsecolorType>, debug_desc: Option<&DebugVizDesc>, num_tiles_x: u32, tile_size: u32, i: u32) -> PushConstants
{
    let mut push_constants = get_push_constants(desc, scene, pathtrace_type, falsecolor_type, debug_desc);

    let offset_x = (i % num_tiles_x) * tile_size * WORKGROUP_SIZE;
    let offset_y = (i / num_tiles_x) * tile_size * WORKGROUP_SIZE;

    push_constants.id_offset = [ offset_x, offset_y ];

    return push_constants;
}
