
use lupin_pt as lp;
use lupin_loader as lpl;
use lupin_pt::wgpu as wgpu;

fn main()
{
    // Initialize WGPU
    let (device, queue, _) = lp::init_default_wgpu_context_no_window();
    // Initialize lupin resources (all desc-type structs have reasonable defaults)
    let pathtrace_res = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
        with_runtime_checks: false,  // This greatly affects render time!
        max_bounces: 8,
        samples_per_pixel: 5,
    });

    // Load/create the scene.
    let (scene, cameras) = lpl::build_scene_cornell_box(&device, &queue, false);
    // let (scene, cameras) = lpl::load_scene_yoctogl_v24("scene_path", &device, &queue, false).unwrap();
    // Set up double buffered output texture for accumulation
    let mut output = lp::DoubleBufferedTexture::create(&device, &wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 1000, height: 1000, depth_or_array_layers: 1 },
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
        lp::pathtrace_scene(&device, &queue, &pathtrace_res, &scene, output.front(), Default::default(), &lp::PathtraceDesc {
            accum_params: Some(lp::AccumulationParams {
                prev_frame: output.back(),
                accum_counter: accum_idx,
            }),
            tile_params: None,
            camera_params: cameras[0].params,
            camera_transform: cameras[0].transform,
            force_software_bvh: false,
            advanced: Default::default(),
        });
        output.flip();
    }
    output.flip();

    lpl::save_texture(&device, &queue, std::path::Path::new("output.hdr"), output.front()).unwrap();
}
