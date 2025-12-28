
#![allow(dead_code)]
#![allow(unused_labels)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

//! # Lupin
//! lupin is a cross-platform, data-oriented library for fast photorealistic
//! rendering on the GPU. It's meant to be simple and C-like.
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
