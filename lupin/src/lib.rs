
#![allow(unexpected_cfgs)]

//! # Lupin
//! Lupin is a cross-platform, data-oriented library for fast photorealistic
//! rendering on the GPU with WGPU. It's meant to be simple and C-like, and supports hardware raytracing.
//! For scene serialization, lupin_loader could be used, or a custom loader which uses lupin's scene-building API.

//! # Library Usage
//! Here's a simple example of loading a scene and producing a path traced image:
//! ```no_run
//! use lupin as lp;
//! use lupin_loader as lpl;  // Optional
//! use lupin::wgpu as wgpu;
//!
//! fn main()
//! {
//!     // Initialize WGPU
//!     let (device, queue, adapter) = lp::init_default_wgpu_context_no_window();
//!
//!     // Initialize lupin resources (all desc-type structs have reasonable defaults)
//!     let tonemap_res = lp::build_tonemap_resources(&device);
//!     let pathtrace_res = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
//!         with_runtime_checks: false,
//!         max_bounces: 8,
//!         samples_per_pixel: 5,
//!     });
//!
//!     // Load/create the scene.
//!     let (scene, cameras) = lp::build_scene_cornell_box(&device, &queue, false);
//!     // let (scene, cameras) = lpl::load_scene_yoctogl_v24("scene_path", &device, &queue, false).unwrap();
//!
//!     // Set up double buffered output texture for accumulation
//!     let output = lp::DoubleBufferedTexture::create(&device, &wgpu::TextureDescriptor {
//!         label: None,
//!         size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
//!         mip_level_count: 1,
//!         sample_count: 1,
//!         dimension: wgpu::TextureDimension::D2,
//!         format: wgpu::TextureFormat::Rgba16Float,
//!         usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
//!                wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
//!                wgpu::TextureUsages::RENDER_ATTACHMENT,
//!         view_formats: &[]
//!     });
//!
//!     // Accumulation loop. This is highly recommended as opposed to increasing the sample
//!     // count in lp::BakedPathtraceParams, because shader invocations that run for too long
//!     // will cause most current OSs to issue a complete driver reset. Accumulation is useful
//!     // as a way to break-up the GPU work into multiple invocations.
//!     let num_accums = 200;
//!     for accum_idx in 0..num_accums
//!     {
//!         lp::pathtrace_scene(&device, &queue, &pathtrace_res, Default::default(), &lp::PathtraceDesc {
//!             accum_params: Some(lp::AccumulationParams {
//!                 prev_frame: output.back(),
//!                 accum_counter: accum_idx,
//!             }),
//!             tile_params: None,
//!             camera_params: Default::default(),
//!             camera_transform: Default::default(),
//!             force_software_bvh: false,
//!             advanced: Default::default(),
//!         });
//!
//!         output.flip();
//!     }
//!
//!     output.flip();
//!     lpl::save_texture(&device, &queue, "output.hdr", output.front());
//! }
//!
//! ```
//!
//! # Scene building API
//! The typical workflow is as follows:
//! ```no_run
//! use lupin as lp;
//!
//! fn main()
//! {
//!     let mut textures = Vec::<wgpu::Texture>::new();
//!     let mut samplers = Vec::<wgpu::Sampler>::new();
//!     let mut environment_infos = Vec::<lp::EnvMapInfo>::with_capacity(scene.environments.len());
//!     let scene_cpu = lp::SceneCPU {
//!         mesh_infos: /* ... */,
//!         verts_pos_array: /* ... */,
//!         verts_normal_array: /* ... */,
//!         verts_texcoord_array: /* ... */,
//!         verts_color_array: /* ... */,
//!         indices_array: /* ... */,
//!         instances: /* ... */,
//!         materials: /* ... */,
//!         environments: /* ... */,
//!     }
//!
//!     // Load textures and samplers (no CPU copy for those, except for environments).
//!     // Each texture needs to have a corresponding sampler.
//!     // textures.push(...);
//!     // samplers.push(...);
//!
//!     // Load environments (with CPU copy)
//!     // environment_infos.push(...);
//!
//!     lp::validate_scene(&scene_cpu, textures.len() as u32, samplers.len() as u32);
//!     let scene_gpu = lp::build_accel_structures_and_upload(device, queue, &scene, textures, samplers, &environment_infos, true);
//!
//!     // Some stats can be obtained:
//!     let scene_stats = get_scene_stats(&scene_gpu);
//! }
//!
//! ```
//!
//! But, if your project has specific requirements - e.g. big scenes and/or strict memory constraints -
//! you can build the acceleration structures directly, through functions like [`build_bvh`], [`build_tlas`], [`build_rt_accel_structures`], [`build_lights`].
//!
//! # Materials
//! The material system is the same as the one used in the [Yocto/GL](https://github.com/xelatihy/yocto-gl/) library.
//!
//! The material type is a 'megastruct' where its fields can be used in different ways depending on the material type:
//! - Matte: Diffuse BSDF,
//! - Glossy: Diffuse + Microfacet BSDF
//! - Reflective: Metallic appearance achieved using a delta/microfacet BSDF
//! - Transparent: Thin glass-like appearance achieved using a delta/microfacet BSDF
//! - Refractive: Glass-like appearance achieved using a delta/microfacet BSDF
//! - GLTF-PBR: Implements compatibility with Khronos' glTF format
//!
//! # Denoising
//! This library supports machine-learned denoising with the use of [Intel Open Image Denoise (OIDN)](https://www.openimagedenoise.org/).
//! Since this costs two extra dependencies, it is an optional feature which can be enabled like this:
//! ```toml
//! lupin = { features = [ "denoising" ] }
//! ```
//! Denoising is straightforward:
//! ```no_run
//! use lupin as lp;
//!
//! fn main()
//! {
//!     // Initialize WGPU
//!     let (device, queue, adapter, denoise_device) = lp::init_default_wgpu_context_with_denoising_capabilities();
//!
//!     let denoise_res = lp::build_denoise_resources(&device, &denoise_device, /* width */, /* height */);
//!
//!     // Render a scene. Optionally, for better denoising quality,
//!     // albedo and normals gbuffers can be provided.
//!     let output = /* ... */;
//!     let albedo = /* ... */;
//!     let normals = /* ... */;
//!
//!     lp::denoise(&device, &queue, &denoise_device, &denoise_res, &lp::DenoiseDesc {
//!         pathtrace_output: &output,
//!         albedo: Some(&albedo),
//!         normals: Some(&normals),
//!         denoise_output: Some(&output),  // Can be the same as pathtrace_output!
//!         quality: Default::default(),
//!     });
//! }
//!
//! ```

mod base;
mod renderer;
mod wgpu_utils;
mod data_structures;
mod tonemapping;
#[cfg(feature = "denoising")]
mod denoising;

#[doc(no_inline)]
pub use base::*;
pub use renderer::*;
pub use wgpu_utils::*;
pub use data_structures::*;
pub use tonemapping::*;
#[cfg(feature = "denoising")]
pub use denoising::*;
pub use wgpu;  // No need to have an extra cargo dependency on wgpu.
