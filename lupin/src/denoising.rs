
use crate::base::*;
use crate::wgpu_utils::*;
use crate::renderer::*;

pub enum DenoiseBuffer
{
    Shared(oidn_wgpu_interop::SharedBuffer),
    Private(oidn::Buffer)
}

impl DenoiseBuffer
{
    pub fn oidn_buffer(&self) -> &oidn::Buffer
    {
        return match self {
            DenoiseBuffer::Shared(shared) => shared.oidn_buffer(),
            DenoiseBuffer::Private(private) => &private,
        }
    }
}

/// Used for denoising an image of a given dimension.
/// Can be reused for multiple images/renders given that they have the same
/// dimensions. Formats are all assumed to be rgbaf16 linear.
pub struct DenoiseResources
{
    pub beauty: DenoiseBuffer,
    pub albedo: DenoiseBuffer,
    pub normals: DenoiseBuffer,
    pub filter: oidn::sys::OIDNFilter,
    pub width: u32,
    pub height: u32,

    // Current state, to minimize the number of commit() calls
    // on the filter
    pub cur_has_albedo: bool,
    pub cur_has_normals: bool,
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
                               width: u32, height: u32) -> DenoiseResources
{
    let beauty: DenoiseBuffer;
    let albedo: DenoiseBuffer;
    let normals: DenoiseBuffer;

    let filter: oidn::sys::OIDNFilter;

    let row_size = align_up(4 * 2 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = row_size * height;

    match denoise_device
    {
        // Supports shared device.
        DenoiseDevice::InteropDevice(interop_device) =>
        {
            let beauty_shared  = interop_device.allocate_shared_buffers(buffer_size as u64).unwrap();
            let albedo_shared  = interop_device.allocate_shared_buffers(buffer_size as u64).unwrap();
            let normals_shared = interop_device.allocate_shared_buffers(buffer_size as u64).unwrap();

            unsafe
            {
                use oidn::sys::*;
                let device_raw = interop_device.oidn_device().raw();
                let shared_beauty_raw  = beauty_shared.oidn_buffer().raw();

                filter = oidnNewFilter(device_raw, c"RT".as_ptr());
                if filter.is_null() { panic!("Failed to create OIDN filter"); }

                oidnSetFilterBool(filter, c"hdr".as_ptr(), true);
                oidnSetFilterBool(filter, c"srgb".as_ptr(), false);
                oidnSetFilterBool(filter, c"clean_aux".as_ptr(), false);
                oidnSetFilterImage(filter, c"color".as_ptr(), shared_beauty_raw,
                                   OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, row_size as usize);
                oidnSetFilterImage(filter, c"output".as_ptr(), shared_beauty_raw,
                                   OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, row_size as usize);
                oidnCommitFilter(filter);
                oidn_check(device_raw);
            }

            beauty = DenoiseBuffer::Shared(beauty_shared);
            albedo = DenoiseBuffer::Shared(albedo_shared);
            normals = DenoiseBuffer::Shared(normals_shared);
        }
        // Doesn't support shared device.
        DenoiseDevice::OidnDevice(oidn_device) =>
        {
            panic!("Not implemented!")
            //buffers = DenoiseBuffers::Shared(DenoiseSharedBuffers { beauty, albedo, normals });
        }
    }

    return DenoiseResources {
        beauty,
        albedo,
        normals,
        filter,
        width,
        height,

        cur_has_albedo: false,
        cur_has_normals: false,
    };
}

/// Denoising params.
pub struct DenoiseDesc<'a>
{
    /// Assumed to be rgbaf16 linear.
    pub pathtrace_output: &'a wgpu::Texture,
    /// Assumed to be rgba8_unorm.
    pub albedo: Option<&'a wgpu::Texture>,
    /// Assumed to be rgba8_snorm, in the [-1, 1] range.
    pub normals: Option<&'a wgpu::Texture>,
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
               denoise_device: &DenoiseDevice, resources: &mut DenoiseResources,
               desc: &DenoiseDesc)
{
    assert!(desc.pathtrace_output.format() == wgpu::TextureFormat::Rgba16Float);
    assert!(desc.pathtrace_output.width() == resources.width && desc.pathtrace_output.height() == resources.height);
    assert!(desc.denoise_output.width() == resources.width && desc.denoise_output.height() == resources.height);

    if let Some(albedo) = desc.albedo {
        assert!(albedo.format() == wgpu::TextureFormat::Rgba16Float);
        assert!(albedo.width() == resources.width && albedo.height() == resources.height);
    }
    if let Some(normals) = desc.normals {
        assert!(normals.format() == wgpu::TextureFormat::Rgba16Float);
        assert!(normals.width() == resources.width && normals.height() == resources.height);
    }

    let width = resources.width;
    let height = resources.height;

    // Copy local buffers to shared buffers
    /*
    {
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
                    bytes_per_row: Some(align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
                    rows_per_image: Some(height)
                }
            },
            desc.pathtrace_output.size()
        );

        if let Some(albedo) = desc.albedo
        {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: albedo,
                    mip_level: 0,
                    origin: Default::default(),
                    aspect: Default::default()
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: resources.albedo.wgpu_buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
                        rows_per_image: Some(height)
                    }
                },
                desc.pathtrace_output.size()
            );
        }

        if let Some(normals) = desc.normals
        {
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: normals,
                    mip_level: 0,
                    origin: Default::default(),
                    aspect: Default::default()
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: resources.normals.wgpu_buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
                        rows_per_image: Some(height)
                    }
                },
                desc.pathtrace_output.size()
            );
        }

        queue.submit(Some(encoder.finish()));
    }
    */

    copy_texture_to_oidn_buf(device, queue, desc.pathtrace_output, &resources.beauty);
    if let Some(albedo) = desc.albedo {
        copy_texture_to_oidn_buf(device, queue, albedo, &resources.albedo);
    }
    if let Some(normals) = desc.normals {
        copy_texture_to_oidn_buf(device, queue, normals, &resources.normals);
    }

    let filter_quality = match desc.quality
    {
        DenoiseQuality::Low => { oidn::Quality::Fast }
        DenoiseQuality::Medium => { oidn::Quality::Balanced }
        DenoiseQuality::High => { oidn::Quality::High }
    };

    let albedo_oidn = resources.albedo.oidn_buffer();
    let normals_oidn = resources.normals.oidn_buffer();
    let device_oidn = denoise_device.oidn_device();

    // Must wait for wgpu to finish before we can start the OIDN workload.
    // Unfortunately WGPU-OIDN synchronization is currently not possible.
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // Run OIDN
    unsafe
    {
        use oidn::sys::*;
        let device_raw = device_oidn.raw();

        let shared_albedo_raw  = resources.albedo.oidn_buffer().raw();
        let shared_normals_raw = resources.normals.oidn_buffer().raw();

        let filter = resources.filter;
        oidnSetFilterInt(filter, c"quality".as_ptr(), filter_quality as i32);

        let row_size = align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

        // Update aux images if necessary
        if !resources.cur_has_albedo && desc.albedo.is_some()
        {
            oidnSetFilterImage(filter, c"albedo".as_ptr(), shared_albedo_raw,
                               OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, row_size as usize);
            resources.cur_has_albedo = true;
        }
        else if resources.cur_has_albedo && !desc.albedo.is_some()
        {
            oidnUnsetFilterImage(filter, c"albedo".as_ptr());
            resources.cur_has_albedo = false;
        }
        if !resources.cur_has_normals && desc.normals.is_some()
        {
            oidnSetFilterImage(filter, c"normals".as_ptr(), shared_normals_raw,
                               OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, row_size as usize);
            resources.cur_has_normals = true;
        }
        else if resources.cur_has_normals && !desc.normals.is_some()
        {
            oidnUnsetFilterImage(filter, c"normals".as_ptr());
            resources.cur_has_normals = false;
        }
        oidnCommitFilter(filter);

        // This will stall the CPU until the operation is done.
        oidnExecuteFilter(filter);
        oidn_check(device_raw);
    }

    copy_oidn_buf_to_texture(device, queue, &resources.beauty, desc.denoise_output);
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

fn align_up(value: u32, align: u32) -> u32
{
    assert!(0 == (align & (align - 1)), "Must align to a power of two");
    return (value + (align - 1)) & !(align - 1);
}

fn copy_texture_to_oidn_buf(device: &wgpu::Device, queue: &wgpu::Queue,
                            input: &wgpu::Texture, output: &DenoiseBuffer)
{
    let width = input.width();
    let height = input.height();

    match output
    {
        DenoiseBuffer::Shared(shared) =>
        {
            let mut encoder = device.create_command_encoder(&Default::default());

            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: input,
                    mip_level: 0,
                    origin: Default::default(),
                    aspect: Default::default()
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: shared.wgpu_buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
                        rows_per_image: Some(height)
                    }
                },
                input.size()
            );
            queue.submit(Some(encoder.finish()));
        }
        DenoiseBuffer::Private(private) =>
        {
            panic!("Not implemented!")
        }
    }
}

fn copy_oidn_buf_to_texture(device: &wgpu::Device, queue: &wgpu::Queue,
                            input: &DenoiseBuffer, output: &wgpu::Texture)
{
    let width = output.width();
    let height = output.height();

    match input
    {
        DenoiseBuffer::Shared(shared) =>
        {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: shared.wgpu_buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
                        rows_per_image: Some(height)
                    }
                },
                wgpu::TexelCopyTextureInfo {
                    texture: output,
                    mip_level: 0,
                    origin: Default::default(),
                    aspect: Default::default()
                },
                output.size()
            );
            queue.submit(Some(encoder.finish()));
        }
        DenoiseBuffer::Private(private) =>
        {
            panic!("Not implemented!")
        }
    }
}
