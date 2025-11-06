
use crate::base::*;
use crate::wgpu_utils::*;
use crate::renderer::*;

/// Used for denoising an image of a given format and dimensions.
/// Can be reused for multiple images/renders given that they have the same
/// format and dimensions.
pub struct DenoiseResources
{
    /// Assumed to be rgbaf16 linear.
    pub beauty: oidn_wgpu_interop::SharedBuffer,
    pub albedo: oidn_wgpu_interop::SharedBuffer,
    pub normals: oidn_wgpu_interop::SharedBuffer,
    pub filter: oidn::sys::OIDNFilter,
    pub width: u32,
    pub height: u32,
    pub beauty_pixel_byte_size: u8,
}

impl Drop for DenoiseResources
{
    fn drop(&mut self)
    {
        unsafe
        {
            use oidn::sys::*;
            if !self.filter.is_null() {
                oidnReleaseFilter(self.filter);
            }
        }
    }
}

pub fn build_denoise_resources(device: &wgpu::Device, denoise_device: &DenoiseDevice,
                               beauty_pixel_byte_size: u8, width: u32, height: u32) -> DenoiseResources
{
    let beauty: oidn_wgpu_interop::SharedBuffer;
    let albedo: oidn_wgpu_interop::SharedBuffer;
    let normals: oidn_wgpu_interop::SharedBuffer;
    let denoise_output: oidn_wgpu_interop::SharedBuffer;

    let filter: oidn::sys::OIDNFilter;

    let beauty_size = beauty_pixel_byte_size as u32 * width * height;
    let gbuffer_size = 3 * 4 * width * height;

    match denoise_device
    {
        DenoiseDevice::InteropDevice(interop_device) =>
        {
            beauty  = interop_device.allocate_shared_buffers(beauty_size  as u64).unwrap();
            albedo  = interop_device.allocate_shared_buffers(gbuffer_size as u64).unwrap();
            normals = interop_device.allocate_shared_buffers(gbuffer_size as u64).unwrap();

            unsafe
            {
                use oidn::sys::*;
                let device_raw = interop_device.oidn_device().raw();
                let shared_beauty_raw  = beauty.oidn_buffer().raw();
                let shared_albedo_raw  = beauty.oidn_buffer().raw();
                let shared_normals_raw = beauty.oidn_buffer().raw();

                filter = oidnNewFilter(device_raw, c"RT".as_ptr());
                if filter.is_null() { panic!("Failed to create OIDN filter"); }

                oidnSetFilterBool(filter, c"hdr".as_ptr(), true);
                oidnSetFilterBool(filter, c"srgb".as_ptr(), false);
                oidnSetFilterBool(filter, c"clean_aux".as_ptr(), false);
                oidnSetFilterImage(filter, c"color".as_ptr(), shared_beauty_raw,
                                   OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, 0);
                oidnSetFilterImage(filter, c"output".as_ptr(), shared_beauty_raw,
                                   OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, 0);
                //oidnSetFilterImage(filter, c"albedo".as_ptr(), shared_albedo_raw,
                //                   OIDNFormat_OIDN_FORMAT_UINT3, width as usize, height as usize, 0, 4, 0);
                //oidnSetFilterImage(filter, c"normals".as_ptr(), shared_normals_raw,
                //                   OIDNFormat_OIDN_FORMAT_UINT3, width as usize, height as usize, 0, 4, 0);
                oidnCommitFilter(filter);
                oidn_check(device_raw);
            }
        }
        DenoiseDevice::OidnDevice(oidn_device) =>
        {
            panic!("unsupported");
        }
    }

    return DenoiseResources {
        beauty,
        albedo,
        normals,
        filter,
        width,
        height,
        beauty_pixel_byte_size,
    };
}

/// Denoising params.
pub struct DenoiseDesc<'a>
{
    /// Assumed to be rgbaf16 linear.
    pub pathtrace_output: &'a wgpu::Texture,
    /// Assumed to be rgba8_unorm.
    //pub albedo: &'a wgpu::Texture,
    /// Assumed to be rgba8_snorm, in the [-1, 1] range.
    //pub normals: &'a wgpu::Texture,
    /// Output of denoise operation. This can be the same texture as pathtrace_output,
    /// in which case the denoising will cleanly be performed in place.
    pub denoise_output: &'a wgpu::Texture,

    pub quality: DenoiseQuality,
}

#[derive(Default)]
pub enum DenoiseQuality
{
    /// Typically used for interactive preview rendering.
    Low,
    /// Typically used for interactive rendering.
    Medium,
    /// Typically used for final-frame rendering.
    #[default]
    High,
}

/// Keep in mind this will trigger a CPU stall waiting for currently scheduled GPU commands to finish.
pub fn denoise(device: &wgpu::Device, queue: &wgpu::Queue,
               denoise_device: &DenoiseDevice, resources: &DenoiseResources,
               desc: &DenoiseDesc)
{
    assert!(desc.pathtrace_output.format() == wgpu::TextureFormat::Rgba16Float);
    assert!(desc.pathtrace_output.width() == resources.width && desc.pathtrace_output.height() == resources.height);
    assert!(desc.denoise_output.width() == resources.width && desc.denoise_output.height() == resources.height);

    //let has_albedo = desc.albedo.is_some();
    //let has_normals = desc.albedo.is_some();
    let width = resources.width;
    let height = resources.height;

    // Copy pathtrace_output to shared buffer
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: desc.pathtrace_output,
            mip_level: 0,
            origin: Default::default(),
            aspect: Default::default()
        },
        wgpu::TexelCopyBufferInfo {
            buffer: resources.beauty.wgpu_buffer(),
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * width),
                rows_per_image: Some(height)
            }
        },
        desc.pathtrace_output.size()
    );
    queue.submit(Some(encoder.finish()));

    let filter_quality = match desc.quality
    {
        DenoiseQuality::Low => { oidn::Quality::Fast }
        DenoiseQuality::Medium => { oidn::Quality::Balanced }
        DenoiseQuality::High => { oidn::Quality::High }
    };

    // Denoise on shared buffers
    match denoise_device
    {
        DenoiseDevice::InteropDevice(interop_device) =>
        {
            // Must wait for wgpu to finish before we can start the oidn workload.
            // Unfortunately wgpu-oidn synchronization is currently not possible.
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

            unsafe
            {
                use oidn::sys::*;
                let device_raw = interop_device.oidn_device().raw();

                let filter = resources.filter;
                oidnSetFilterInt(filter, c"quality".as_ptr(), filter_quality as i32);
                oidnCommitFilter(filter);

                // This will stall the CPU until the operation is done.
                oidnExecuteFilter(filter);
                oidn_check(device_raw);
            }
        }
        DenoiseDevice::OidnDevice(oidn_device) => { panic!("unsupported"); }
    };

    // Copy output shared buffer to output texture
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: resources.beauty.wgpu_buffer(),
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * width),
                rows_per_image: Some(height)
            }
        },
        wgpu::TexelCopyTextureInfo {
            texture: desc.denoise_output,
            mip_level: 0,
            origin: Default::default(),
            aspect: Default::default()
        },
        desc.pathtrace_output.size()
    );
    queue.submit(Some(encoder.finish()));
}

unsafe fn oidn_check(oidn_device: oidn::sys::OIDNDevice)
{
    unsafe
    {
        use oidn::sys::*;

        let mut error_msg: *const i8 = std::ptr::null();
        if oidnGetDeviceError(oidn_device, &mut error_msg) != OIDNError_OIDN_ERROR_NONE
        {
            let msg = if !error_msg.is_null() {
                std::ffi::CStr::from_ptr(error_msg).to_string_lossy().into_owned()
            } else {
                "<no message>".to_string()
            };

            panic!("OIDN Error: {}", msg);
        }
    }
}
