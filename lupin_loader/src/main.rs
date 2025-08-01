
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

mod loader;
mod input;
mod base;
mod ui;
pub use loader::*;
pub use input::*;
pub use base::*;
pub use ui::*;

fn main()
{
    let mut compile_shaders_only = false;

    let args: Vec<String> = std::env::args().collect();
    for arg in args
    {
        if arg == "-compile_shaders_only" { compile_shaders_only = true; }
    }

    // Initialize window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Lupin Raytracer")
                                     .with_inner_size(PhysicalSize::new(1920.0, 1080.0))
                                     .with_visible(false)
                                     .build(&event_loop)
                                     .unwrap();

    let win_size = window.inner_size();
    let (width, height) = (win_size.width as i32, win_size.height as i32);
    let device_spec = lp::get_required_device_spec();

    let mut surface_config = wgpu::SurfaceConfiguration {
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

    if compile_shaders_only { return; }

    let scene = build_scene(&device, &queue);
    let mut output_textures = [
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

    let mut albedo_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING |
               wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[]
    });
    let mut normals_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Snorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING |
               wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[]
    });
    //

    // let mut egui_ctx = egui::Context::default();
    // let viewport_id = egui_ctx.viewport_id();
    // let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None);

    let mut input = Input::default();

    let min_delta_time: f32 = 1.0/10.0;
    let mut delta_time: f32 = 1.0/60.0;
    let mut time_begin = Instant::now();

    let mut cam_pos = Vec3 { x: 0.0, y: 1.0, z: -3.0 };
    let mut cam_rot = Quat::IDENTITY;

    let mut accum_counter: u32 = 0;
    let mut prev_cam_transform = Mat4::IDENTITY;

    window.set_visible(true);

    event_loop.run(|event, target|
    {
        process_input_event(&mut input, &event);

        if let Event::WindowEvent { window_id: _, event } = event
        {
            // Collect inputs
            // let _ = egui_state.on_window_event(&window, &event);

            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    // Resize screen dependent resources
                    resize_texture(&device, &mut output_textures[0], new_size.width, new_size.height);
                    resize_texture(&device, &mut output_textures[1], new_size.width, new_size.height);
                    resize_texture(&device, &mut albedo_texture, new_size.width, new_size.height);
                    resize_texture(&device, &mut normals_texture, new_size.width, new_size.height);
                    accum_counter = 0;

                    // Resize surface
                    surface_config.width  = new_size.width;
                    surface_config.height = new_size.height;
                    surface.configure(&device, &surface_config);

                    window.request_redraw();
                },
                WindowEvent::CloseRequested =>
                {
                    target.exit();
                },
                WindowEvent::RedrawRequested =>
                {
                    delta_time = time_begin.elapsed().as_secs_f32();
                    delta_time = delta_time.min(min_delta_time);
                    time_begin = Instant::now();

                    update_camera(&mut cam_pos, &mut cam_rot, &input, delta_time);

                    let camera_transform = xform_to_matrix(cam_pos, cam_rot, Vec3 { x: 1.0, y: 1.0, z: 1.0 });
                    if camera_transform.m != prev_cam_transform.m
                    {
                        accum_counter = 0;
                        performed_denoise = false;
                    }
                    prev_cam_transform = camera_transform;

                    let frame = surface.get_current_texture().unwrap();

                    const MAX_ACCUMS: u32 = 1000;
                    if accum_counter < MAX_ACCUMS
                    {
                        let accum_params = lp::AccumulationParams {
                            prev_frame: Some(&output_textures[output_tex_back]),
                            accum_counter: accum_counter,
                        };
                        lp::pathtrace_scene(&device, &queue, &scene, &output_textures[output_tex_front],
                                            &shader_params, &accum_params, camera_transform.into());
                    }
                    else if !performed_denoise
                    {
                        println!("Done!");
                        //lp::transfer_to_cpu_and_denoise_image(&device, &queue, &output_textures[output_tex_front],
                        //                                      &output_textures[output_tex_back], None, None);
                        performed_denoise = true;
                    }

                    let tonemap_params = lp::TonemapParams {
                        operator: lp::TonemapOperator::Aces,
                        exposure: 0.0
                    };
                    lp::apply_tonemapping(&device, &queue, &tonemap_shader_params,
                                          &output_textures[output_tex_front], &frame.texture, &tonemap_params);

                    frame.present();

                    begin_input_events(&mut input);

                    // Swap output textures
                    if accum_counter < MAX_ACCUMS
                    {
                        let tmp = output_tex_back;
                        output_tex_back  = output_tex_front;
                        output_tex_front = tmp;
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
