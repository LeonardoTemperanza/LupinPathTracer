
use crate::renderer::*;

pub struct DenoiseBuffersPrivate
{
    pub output: oidn::Buffer,
    pub albedo: oidn::Buffer,
    pub normals: oidn::Buffer,
    pub cpu_to_gpu: wgpu::Buffer,
    pub gpu_to_cpu: wgpu::Buffer,
}

pub struct DenoiseBuffersShared
{
    pub output: oidn_wgpu_interop::SharedBuffer,
    pub albedo: oidn_wgpu_interop::SharedBuffer,
    pub normals: oidn_wgpu_interop::SharedBuffer,
}

pub enum DenoiseBuffers
{
    Shared(DenoiseBuffersShared),
    Private(DenoiseBuffersPrivate),
}

impl DenoiseBuffers
{
    pub fn oidn_buffer_output(&self) -> &oidn::Buffer
    {
        return match self {
            DenoiseBuffers::Shared(shared) => shared.output.oidn_buffer(),
            DenoiseBuffers::Private(private) => &private.output,
        }
    }

    pub fn oidn_buffer_albedo(&self) -> &oidn::Buffer
    {
        return match self {
            DenoiseBuffers::Shared(shared) => shared.albedo.oidn_buffer(),
            DenoiseBuffers::Private(private) => &private.albedo,
        }
    }

    pub fn oidn_buffer_normals(&self) -> &oidn::Buffer
    {
        return match self {
            DenoiseBuffers::Shared(shared) => shared.normals.oidn_buffer(),
            DenoiseBuffers::Private(private) => &private.normals,
        }
    }
}

/// Used for denoising an image of a given dimension.
/// Can be reused for multiple images/renders given that they have the same
/// dimensions. Formats are all assumed to be rgbaf16 linear.
pub struct DenoiseResources
{
    pub buffers: DenoiseBuffers,
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
    let buffers: DenoiseBuffers;

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

            buffers = DenoiseBuffers::Shared(DenoiseBuffersShared {
                output: beauty_shared,
                albedo: albedo_shared,
                normals: normals_shared,
            });
        }
        // Doesn't support shared device.
        DenoiseDevice::OidnDevice(oidn_device) =>
        {
            let cpu_to_gpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: buffer_size as u64,  // Only ever need to transfer output.
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let gpu_to_cpu = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: buffer_size as u64 * 3,  // Enough space to fit all buffers.
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            unsafe
            {
                use oidn::sys::*;
                let device_raw = oidn_device.raw();

                let beauty_private_raw = oidnNewBuffer(device_raw, buffer_size as usize);
                let albedo_private_raw = oidnNewBuffer(device_raw, buffer_size as usize);
                let normals_private_raw = oidnNewBuffer(device_raw, buffer_size as usize);
                assert!(!beauty_private_raw.is_null());
                assert!(!albedo_private_raw.is_null());
                assert!(!normals_private_raw.is_null());

                filter = oidnNewFilter(device_raw, c"RT".as_ptr());
                if filter.is_null() { panic!("Failed to create OIDN filter"); }

                oidnSetFilterBool(filter, c"hdr".as_ptr(), true);
                oidnSetFilterBool(filter, c"srgb".as_ptr(), false);
                oidnSetFilterBool(filter, c"clean_aux".as_ptr(), false);
                oidnSetFilterImage(filter, c"color".as_ptr(), beauty_private_raw,
                                   OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, row_size as usize);
                oidnSetFilterImage(filter, c"output".as_ptr(), beauty_private_raw,
                                   OIDNFormat_OIDN_FORMAT_HALF3, width as usize, height as usize, 0, 4 * 2, row_size as usize);
                oidnCommitFilter(filter);
                oidn_check(device_raw);

                buffers = DenoiseBuffers::Private(DenoiseBuffersPrivate {
                    output: oidn_device.create_buffer_from_raw(beauty_private_raw),
                    albedo: oidn_device.create_buffer_from_raw(albedo_private_raw),
                    normals: oidn_device.create_buffer_from_raw(normals_private_raw),
                    cpu_to_gpu,
                    gpu_to_cpu
                });
            }
        }
    }

    return DenoiseResources {
        buffers,
        filter,
        width,
        height,

        cur_has_albedo: false,
        cur_has_normals: false,
    };
}

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
/// It will also copy to the CPU and back if the device doesn't support WGPU-OIDN compatibility.
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

    // Copy input buffers
    copy_textures_to_oidn_bufs(device, queue, desc, &resources.buffers);

    let filter_quality = match desc.quality
    {
        DenoiseQuality::Low => { oidn::Quality::Fast }
        DenoiseQuality::Medium => { oidn::Quality::Balanced }
        DenoiseQuality::High => { oidn::Quality::High }
    };

    let albedo_oidn = resources.buffers.oidn_buffer_albedo();
    let normals_oidn = resources.buffers.oidn_buffer_normals();
    let device_oidn = denoise_device.oidn_device();

    // Must wait for WGPU to finish before we can start the OIDN workload.
    // Unfortunately WGPU-OIDN synchronization is currently not possible.
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    // Run OIDN
    unsafe
    {
        use oidn::sys::*;
        let device_raw = device_oidn.raw();

        let albedo_raw  = albedo_oidn.raw();
        let normals_raw = normals_oidn.raw();

        let filter = resources.filter;
        oidnSetFilterInt(filter, c"quality".as_ptr(), filter_quality as i32);

        let row_size = align_up(8 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

        // Update aux images if necessary
        if !resources.cur_has_albedo && desc.albedo.is_some()
        {
            oidnSetFilterImage(filter, c"albedo".as_ptr(), albedo_raw,
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
            oidnSetFilterImage(filter, c"normals".as_ptr(), normals_raw,
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

    // Copy to output texture.
    copy_oidn_output_to_texture(device, queue, &resources.buffers, desc.denoise_output);
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

fn copy_textures_to_oidn_bufs(device: &wgpu::Device, queue: &wgpu::Queue,
                              desc: &DenoiseDesc, outputs: &DenoiseBuffers)
{
    let width = desc.pathtrace_output.width();
    let height = desc.pathtrace_output.height();
    let row_size = align_up(4 * 2 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = (row_size * height) as usize;

    match outputs
    {
        DenoiseBuffers::Shared(shared) =>
        {
            let mut encoder = device.create_command_encoder(&Default::default());
            copy_tex_to_buf(&mut encoder, desc.pathtrace_output, shared.output.wgpu_buffer(), 0);
            if let Some(albedo) = desc.albedo {
                copy_tex_to_buf(&mut encoder, albedo, shared.albedo.wgpu_buffer(), 0);
            }
            if let Some(normals) = desc.normals {
                copy_tex_to_buf(&mut encoder, normals, shared.normals.wgpu_buffer(), 0);
            }
            queue.submit(Some(encoder.finish()));
        }
        DenoiseBuffers::Private(private) =>
        {
            // Copy texture to transfer buffer first, then back to OIDN buffer (CPU/GPU)

            let mut encoder = device.create_command_encoder(&Default::default());
            copy_tex_to_buf(&mut encoder, desc.pathtrace_output, &private.gpu_to_cpu, 0);
            if let Some(albedo) = desc.albedo {
                copy_tex_to_buf(&mut encoder, albedo, &private.gpu_to_cpu, buffer_size);
            }
            if let Some(normals) = desc.normals {
                copy_tex_to_buf(&mut encoder, normals, &private.gpu_to_cpu, buffer_size * 2);
            }
            queue.submit(Some(encoder.finish()));

            {
                let slice = private.gpu_to_cpu.slice(..);

                let _ = slice.map_async(wgpu::MapMode::Read, | _ | {} );
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

                let data = slice.get_mapped_range();
                unsafe
                {
                    use oidn::sys::*;
                    let output_raw = private.output.raw();
                    let albedo_raw = private.albedo.raw();
                    let normals_raw = private.normals.raw();
                    let output = &data[0..buffer_size as usize];
                    let albedo = &data[buffer_size as usize..buffer_size*2];
                    let normals = &data[buffer_size*2..buffer_size*3];
                    oidnWriteBuffer(output_raw, 0, output.len(), output.as_ptr() as *const std::ffi::c_void);
                    if desc.albedo.is_some() {
                        oidnWriteBuffer(albedo_raw, 0, albedo.len(), albedo.as_ptr() as *const std::ffi::c_void);
                    }
                    if desc.normals.is_some() {
                        oidnWriteBuffer(normals_raw, 0, normals.len(), normals.as_ptr() as *const std::ffi::c_void);
                    }
                }
            }

            private.gpu_to_cpu.unmap();
        }
    }
}

fn copy_oidn_output_to_texture(device: &wgpu::Device, queue: &wgpu::Queue,
                               input: &DenoiseBuffers, output: &wgpu::Texture)
{
    let width = output.width();
    let height = output.height();
    let row_size = align_up(4 * 2 * width, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    match input
    {
        DenoiseBuffers::Shared(shared) =>
        {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: shared.output.wgpu_buffer(),
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_size),
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
        DenoiseBuffers::Private(private) =>
        {
            // Copy OIDN buffer to transfer buffer first, then back to GPU texture

            {
                let slice = private.cpu_to_gpu.slice(..);
                let _mapping = slice.map_async(wgpu::MapMode::Write, | _ | {} );
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

                let data = slice.get_mapped_range_mut();

                unsafe
                {
                    use oidn::sys::*;
                    let private_raw = private.output.raw();
                    oidnReadBuffer(private_raw, 0, data.len(), data.as_ptr() as *mut std::ffi::c_void);
                }
            }

            private.cpu_to_gpu.unmap();

            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &private.cpu_to_gpu,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row_size),
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

            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
    }
}

fn copy_tex_to_buf(cmd_buf: &mut wgpu::CommandEncoder, output: &wgpu::Texture, buf: &wgpu::Buffer, offset: usize)
{
    let row_size = align_up(4 * 2 * output.width(), wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);

    cmd_buf.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: output,
            mip_level: 0,
            origin: Default::default(),
            aspect: Default::default()
        },
        wgpu::TexelCopyBufferInfo {
            buffer: buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: offset as u64,
                bytes_per_row: Some(row_size),
                rows_per_image: Some(output.height())
            }
        },
        output.size()
    );
}

fn align_up(value: u32, align: u32) -> u32
{
    assert!(0 == (align & (align - 1)), "Must align to a power of two");
    return (value + (align - 1)) & !(align - 1);
}
