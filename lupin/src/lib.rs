
#![allow(dead_code)]
#![allow(unused_labels)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

/// # Lupin
/// lupin is a cross-platform, data-oriented library for fast photorealistic
/// rendering on the GPU. It's meant to be simple and C-like.
/// For scene serialization, lupin_loader could be used.

/// # Library Usage
/// Here's a simple example of loading a scene and producing a path traced image:
/// ```
/// use lupin as lp;
/// use lupin::wgpu as wgpu;
///
/// fn main()
/// {
///     // Initialize WGPU
///     let (device, queue, adapter) = lp::init_default_wgpu_context_no_window();
///
///     // Initialize lupin resources (all desc-type structs have reasonable defaults)
///     let tonemap_resources = lp::build_tonemap_resources(&device);
///     let pathtrace_resources = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
///         with_runtime_checks: false,
///         max_bounces: 8,
///         samples_per_pixel: 5,
///     });
///
///     // Load/create the scene. Alternatively, lupin_loader could be used or a custom loader using
///     // lupin's scene building API.
///     let (scene, cameras) = lp::build_scene_cornell_box(&device, &queue, false);
///     // let (scene, cameras) = lpl::load_scene_yoctogl_v24("scene_path", &device, &queue, false).unwrap();
///
///     // Accumulation loop. This is highly recommended as opposed to increasing the sample
///     // count in lp::BakedPathtraceParams, because shader invocations that run for too long
///     // will cause all current operating to issue a complete driver reset. Accumulation is useful
///     // as a way to break-up the GPU work into multiple invocations.
///     let num_accums = 200;
///     for accum_idx in 0..num_accums
///     {
///     }
///     lp::pathtrace_scene
/// }
///
/// ```

mod base;
mod renderer;
mod wgpu_utils;
mod data_structures;
mod tonemapping;
#[cfg(feature = "denoising")]
mod denoising;

pub use base::*;
pub use renderer::*;
pub use wgpu_utils::*;
pub use data_structures::*;
pub use tonemapping::*;
#[cfg(feature = "denoising")]
pub use denoising::*;
pub use wgpu;  // No need to have an extra cargo dependency on wgpu.
