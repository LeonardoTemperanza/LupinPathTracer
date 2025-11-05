
use crate::base::*;
use crate::wgpu_utils::*;
use crate::renderer::*;

/// Used for denoising an image of a given format and dimensions.
/// Can be reused for multiple images/renders given that they have the same
/// format and dimensions.
pub struct DenoiseResources
{
    pub buf: oidn_wgpu_interop::SharedBuffer,
    pub width: u32,
    pub height: u32,
    pub pixel_byte_size: u8,
}

pub fn build_denoise_resources(device: wgpu::Device, denoise_device: DenoiseDevice, pixel_byte_size: u8, width: u32, heigth: u32)
{
    match denoise_device
    {
        DenoiseDevice::InteropDevice(interop_device) =>
        {

        }
        DenoiseDevice::OidnDevice(oidn_device) =>
        {

        }
    }
}

/// Description of the
pub struct DenoiseDesc<'a>
{
    pub pathtrace_output: &'a wgpu::Texture,
    pub albedo: Option<&'a wgpu::Texture>,
    pub normals: Option<&'a wgpu::Texture>,
    pub denoise_output: &'a wgpu::Texture,
}

pub fn denoise(device: wgpu::Device, denoise_device: DenoiseDevice, denoise_resources: &DenoiseResources, denoise_desc: &DenoiseDesc)
{
    // I guess we should check the format of the textures.
}
