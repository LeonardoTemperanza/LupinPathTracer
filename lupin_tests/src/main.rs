
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

use std::time::Instant;

pub use winit::
{
    window::*,
    event::*,
    dpi::*,
    event_loop::*
};

//use ::egui::FontDefinitions;

pub use lupin as lp;

fn main()
{
    // Initialize window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Lupin Raytracer")
                                     .with_inner_size(PhysicalSize::new(1920.0, 1080.0))
                                     .with_visible(false)
                                     .with_resizable(false)
                                     .build(&event_loop)
                                     .unwrap();

    let win_size = window.inner_size();
    let (width, height) = (win_size.width as i32, win_size.height as i32);
    let device_spec = lp::get_required_device_spec();

    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: width as u32,
        height: height as u32,
        present_mode: wgpu::PresentMode::Mailbox,
        desired_maximum_frame_latency: 0,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    };

    let (device, queue, surface, adapter) = lp::init_default_wgpu_context(device_spec, &surface_config, &window, width, height);
    //log_backend(&adapter);

    // Init rendering resources
    let shader_params = lp::build_pathtrace_shader_params(&device, false);
    let tonemap_shader_params = lp::build_tonemap_shader_params(&device);

    // let scene = build_scene(&device, &queue);

    let test_state = TestState::default();

    let output_textures = [
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING |
                   wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC |
                   wgpu::TextureUsages::COPY_DST,
            view_formats: &[]
        }),
        device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING |
                   wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC |
                   wgpu::TextureUsages::COPY_DST,
            view_formats: &[]
        })
    ];

    let mut output_tex_front = 1;
    let mut output_tex_back  = 0;

    let mut accum_counter: u32 = 0;

    window.set_visible(true);

    event_loop.run(|event, target|
    {
        if let Event::WindowEvent { window_id: _, event } = event
        {
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    assert!(false);  // Should not happen, window is not resizable.
                },
                WindowEvent::CloseRequested =>
                {
                    target.exit();
                },
                WindowEvent::RedrawRequested =>
                {
                    let frame = surface.get_current_texture().unwrap();

                    const MAX_ACCUMS: u32 = 1000;
                    if accum_counter < MAX_ACCUMS
                    {
                        let accum_params = lp::AccumulationParams {
                            prev_frame: Some(&output_textures[output_tex_back]),
                            accum_counter: accum_counter,
                        };
                        //lp::pathtrace_scene(&device, &queue, &scene, &output_textures[output_tex_front],
                        //                    &shader_params, &accum_params, camera_transform.into());
                    }

                    // Draw rendered image.
                    {
                        let tonemap_params = lp::TonemapParams {
                            operator: lp::TonemapOperator::Aces,
                            exposure: 0.0
                        };
                        lp::apply_tonemapping(&device, &queue, &tonemap_shader_params,
                                              &output_textures[output_tex_front], &frame.texture, &tonemap_params);
                    }

                    // Draw ground truth.
                    {

                    }

                    frame.present();

                    // Swap output textures.
                    if accum_counter < MAX_ACCUMS
                    {
                        let tmp = output_tex_back;
                        output_tex_back  = output_tex_front;
                        output_tex_front = tmp;
                    }

                    // Finished rendering, check correctness.
                    if accum_counter >= MAX_ACCUMS
                    {
                        let eps = 0.01;
                        let result = check_correctness(&test_state, eps);
                        if result.correct
                        {
                            println!(" - OK")
                        }
                        else
                        {
                            result
                        }
                    }

                    accum_counter = (accum_counter + 1).min(MAX_ACCUMS);

                    // Continuously request drawing messages to let the main loop continue
                    window.request_redraw();
                },
                _ => {},
            }
        }
    }).unwrap();
}

pub struct TestState
{
    pub out_textures: [wgpu::Texture; 2],
    pub scene: lp::SceneDesc,
}

pub fn load_next_test(test_state: &mut TestState, test_idx: u32)
{

}

pub struct CheckCorrectnessResult
{
    pub correct: bool,
    pub max_err: f32,
    pub error_texture: wgpu::Texture,
}
pub fn check_correctness(test_state: &TestState, epsilon: f32) -> CheckCorrectnessResult
{
    return CheckCorrectnessResult {
        correct: true,
        max_err: 0,
        error_texture: Default::default()
    }
}
