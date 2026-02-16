# Lupin: A WGPU Path Tracing Library
**Lupin** is a data-oriented library for fast photorealistic rendering on the GPU with WGPU. It's meant to be simple and C-like, and while it supports hardware raytracing, it still provides a software implementation for compatibility with older devices. It is designed for research, testing, or integration into graphics pipelines.

- Physically based path tracing with multiple importance sampling (MIS).

- Russian Roulette path termination.
- Progressive and tiled rendering.
- Emissive mesh lights and HDRIs.
- Matte, glossy, reflective, transparent, refractive, subsurface, volumetric, materials, with GLTF-PBR compatibility.
- Support for emission-maps, metallic and roughness maps and normal-maps.
- Support for different aperture, aspect ratio, focal length, and orthographic cameras.
- Optional GPU Denoising with [OIDN](https://www.openimagedenoise.org/).

During the development of this project I've used [Yocto/GL](https://github.com/xelatihy/yocto-gl) as a reference for the theoretical foundations of material models and light sampling. This project also uses its scene format and scenes (with permission by its author, who I know personally).

Scene serialization could be implemented using **lupin_loader**, or a custom loader which can use **Lupin**'s scene building API.

## Showcase:
![lonemonk](readme_images/lonemonk.png)

![classroom](readme_images/classroom.png)

![bistroexterior](readme_images/bistroexterior.png)

![landscape](readme_images/landscape1.png)

Depending on the hardware and on the complexity of the scene, the user can likely move the camera and visualize the path traced scene in a pseudo-real-time fashion. Try it yourself by downloading the latest release, which includes **Lupin Viewer** and a few test scenes!

## API Usage:


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
lupin = "*"
```

This library optionally supports denoising using [OIDN](https://www.openimagedenoise.org/). To enable denoising, make sure to add `features = [ "denoising" ]` to your `Cargo.toml`. **OIDN** has to be installed separately (binaries can be found [here]())

Other features are `"denoise-force-disable-shared-device"`, which forces the CPU denoising fallback, and `"force-swrt"` which acts like the device doesn't support hardware raytracing, even if it does.
