# Lupin: A WGPU Path Tracing Library
**Lupin** is a data-oriented library for fast photorealistic rendering on the GPU with WGPU. It's meant to be simple and C-like, and while it supports hardware raytracing, it still provides a software implementation for compatibility with older devices. It is designed for research, testing, or integration into graphics pipelines.

- Physically based path tracing with multiple importance sampling (MIS).
- Naive path tracing and other algorithms are also supported.
- Russian Roulette path termination.
- Progressive and tiled rendering.
- Emissive mesh lights and HDRIs.
- Materials: Matte, glossy, reflective, transparent, refractive, subsurface, volumetric, with GLTF-PBR compatibility.
- Support for emission-maps, metallic and roughness maps and normal-maps.
- Support for different aperture, aspect ratio, focal length, and orthographic cameras.
- Optional GPU Denoising with [OIDN](https://www.openimagedenoise.org/).

During the development of this project I've used [Yocto/GL](https://github.com/xelatihy/yocto-gl) as a reference for the theoretical foundations of material models and light sampling. This project also uses its scene format and scenes (with permission by its author, who I know personally).

Scene serialization could be implemented using **lupin_loader**, or a custom loader which can use **Lupin**'s scene building API.

## Showcase:
![lonemonk](readme_images/lonemonk.png)
Scene by Carlo Bergonzini.

![classroom](readme_images/classroom.png)
Scene by Christophe Seux.

![bistroexterior](readme_images/bistroexterior.png)
Scene by Amazon Lumberyard.

![landscape](readme_images/landscape1.png)
Scene by Jan-Walter Schliep, Burak Kahraman, Timm Dapper.

https://www.youtube.com/watch?v=EcDY_xUkNxs

Depending on the hardware and on the complexity of the scene, the user can likely move the camera and visualize the path traced scene in a pseudo-real-time fashion. Try it yourself by downloading the latest release, which includes **Lupin Viewer** and a few test scenes!

## API Usage:
Here's a simple example of loading a scene and producing a path traced image:
```rust
use lupin as lp;
use lupin_loader as lpl;  // Optional
use lupin::wgpu as wgpu;
fn main()
{
    // Initialize WGPU
    let (device, queue, adapter) = lp::init_default_wgpu_context_no_window();
    // Initialize lupin resources (all desc-type structs have reasonable defaults)
    let tonemap_res = lp::build_tonemap_resources(&device);
    let pathtrace_res = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
        with_runtime_checks: false,
        max_bounces: 8,
        samples_per_pixel: 5,
    });
    // Load/create the scene.
    let (scene, cameras) = lpl::build_scene_cornell_box(&device, &queue, false);
    // let (scene, cameras) = lpl::load_scene_yoctogl_v24("scene_path", &device, &queue, false).unwrap();
    // Set up double buffered output texture for accumulation
    let output = lp::DoubleBufferedTexture::create(&device, &wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
               wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
               wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[]
    });
    // Accumulation loop. This is highly recommended as opposed to increasing the sample
    // count in lp::BakedPathtraceParams, because shader invocations that run for too long
    // will cause most current OSs to issue a complete driver reset. Accumulation is useful
    // as a way to break-up the GPU work into multiple invocations.
    let num_accums = 200;
    for accum_idx in 0..num_accums
    {
        lp::pathtrace_scene(&device, &queue, &pathtrace_res, Default::default(), &lp::PathtraceDesc {
            accum_params: Some(lp::AccumulationParams {
                prev_frame: output.back(),
                accum_counter: accum_idx,
            }),
            tile_params: None,
            camera_params: Default::default(),
            camera_transform: Default::default(),
            force_software_bvh: false,
            advanced: Default::default(),
        });
        output.flip();
    }
    output.flip();
    lpl::save_texture(&device, &queue, "output.hdr", output.front());
}
```

For more info see: MISSING.

## Support Matrix:
|         | DX12 | Vulkan | Metal |
|---------|------|--------|-------|
| Windows |  ⏳* |  ✅    |   /   |
| Linux   |   /  | ✅    |   /   |
| Mac     |   /  |    /   |  ⏳**  |

⏳: Automatically supported once [WGPU](https://wgpu.rs/) implements certain features.

*: This can run, but only with software raytracing.

**: Currently does not support GPU denoising. Will fall back to CPU denoising. Other than that, needs upstream WGPU work to be able to run at all.

## Build:

Just add the following line to your `Cargo.toml`:
```toml
[dependencies]
lupin_pt = "*"
```

This library optionally supports denoising using [OIDN](https://www.openimagedenoise.org/). To enable denoising, make sure to add `features = [ "denoising" ]` to your `Cargo.toml`. **OIDN** has to be installed separately (binaries can be found [here](https://github.com/RenderKit/oidn/releases))

Other features are `"denoise-force-disable-shared-device"`, which forces the CPU denoising fallback, and `"force-swrt"` which acts like the device doesn't support hardware raytracing, even if it does.
