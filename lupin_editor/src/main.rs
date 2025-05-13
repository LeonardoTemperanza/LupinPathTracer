
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

// Don't spawn a terminal window on windows
//#![windows_subsystem = "windows"]

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
    // Initialize window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Lupin Raytracer")
                                     .with_inner_size(LogicalSize::new(1080.0, 720.0))
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
    log_backend(&adapter);

    // Init rendering resources
    let scene = load_scene_obj(&device, &queue, "stanford-bunny.obj");
    let shader_params = lp::build_pathtrace_shader_params(&device, true);
    let tonemap_shader_params = lp::build_tonemap_shader_params(&device);
    let mut hdr_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
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
                    resize_texture(&device, &mut hdr_texture, new_size.width, new_size.height);

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

                    let frame = surface.get_current_texture().unwrap();
                    lp::pathtrace_scene(&device, &queue, &scene, &hdr_texture,
                                        &shader_params, camera_transform.into());

                    let tonemap_params = lp::TonemapParams {
                        operator: lp::TonemapOperator::Aces,
                        exposure: 0.0
                    };
                    lp::apply_tonemapping(&device, &queue, &tonemap_shader_params,
                                         &hdr_texture, &frame.texture, &tonemap_params);


                    //lp::convert_to_ldr_no_tonemap(&device, &queue, &tonemap_shader_params,
                    //                              &hdr_texture, &frame.texture);

                    frame.present();

                    begin_input_events(&mut input);

                    // Continuously request drawing messages to let the main loop continue
                    window.request_redraw();
                },
                _ => {},
            }
        }
    }).unwrap();
}

fn update_camera(cam_pos: &mut Vec3, cam_rot: &mut Quat, input: &Input, delta_time: f32)
{
    fn deg_to_rad(degrees: f32) -> f32 {
        return degrees * std::f32::consts::PI / 180.0;
    }

    let mouse_sensitivity = deg_to_rad(0.2);
    static mut ANGLE: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    let mut mouse = Vec2::default();
    if input.rmouse.pressing
    {
        mouse.x = input.mouse_dx * mouse_sensitivity;
        mouse.y = input.mouse_dy * mouse_sensitivity;
    }

    unsafe
    {
        ANGLE += mouse;
        // Wrap ANGLE.x
        while ANGLE.x < 0.0                        { ANGLE.x += 2.0 * std::f32::consts::PI; }
        while ANGLE.x > 2.0 * std::f32::consts::PI { ANGLE.x -= 2.0 * std::f32::consts::PI; }

        ANGLE.y = ANGLE.y.clamp(deg_to_rad(-90.0), deg_to_rad(90.0));
        let y_rot = angle_axis(Vec3 { x: -1.0, y: 0.0, z: 0.0 }, ANGLE.y);
        let x_rot = angle_axis(Vec3 { x:  0.0, y: 1.0, z: 0.0 }, ANGLE.x);
        *cam_rot = x_rot * y_rot
    }

    // Movement
    static mut CUR_VEL: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };
    let move_speed: f32 = 6.0;
    let move_speed_fast: f32 = 15.0;
    let move_accel: f32 = 100.0;

    let cur_move_speed = if input.keys[Key::LSHIFT].pressing { move_speed_fast } else { move_speed };

    let mut keyboard_dir_xz = Vec3::default();
    let mut keyboard_dir_y: f32 = 0.0;
    if input.rmouse.pressing
    {
        keyboard_dir_xz.x = ((input.keys[Key::D].pressing) as i32 - (input.keys[Key::A].pressing) as i32) as f32;
        keyboard_dir_xz.z = ((input.keys[Key::W].pressing) as i32 - (input.keys[Key::S].pressing) as i32) as f32;
        keyboard_dir_y    = ((input.keys[Key::E].pressing) as i32 - (input.keys[Key::Q].pressing) as i32) as f32;

        // It's a "direction" input so its length
        // should be no more than 1
        if dot_vec3(keyboard_dir_xz, keyboard_dir_xz) > 1.0 {
            keyboard_dir_xz = normalize_vec3(keyboard_dir_xz)
        }

        if keyboard_dir_y.abs() > 1.0 {
            keyboard_dir_y = keyboard_dir_y.signum();
        }
    }

    let mut target_vel = keyboard_dir_xz * cur_move_speed;
    target_vel = vec3_quat_mul(*cam_rot, target_vel);
    target_vel.y += keyboard_dir_y * cur_move_speed;

    unsafe {
        CUR_VEL = approach_linear(CUR_VEL, target_vel, move_accel * delta_time);
        *cam_pos += CUR_VEL * delta_time;
    }

    fn approach_linear(cur: Vec3, target: Vec3, delta: f32) -> Vec3
    {
        let diff = target - cur;
        let dist = magnitude_vec3(diff);

        if dist <= delta { return target; }
        return cur + diff / dist * delta;
    }
}

fn resize_texture(device: &wgpu::Device, texture: &mut wgpu::Texture, new_width: u32, new_height: u32)
{
    let desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: new_width, height: new_height, depth_or_array_layers: 1 },
        mip_level_count: texture.mip_level_count(),
        sample_count: texture.sample_count(),
        dimension: texture.dimension(),
        format: texture.format(),
        usage: texture.usage(),
        view_formats: &[]
    };

    texture.destroy();
    *texture = device.create_texture(&desc);
}

pub fn log_backend(adapter: &wgpu::Adapter)
{
    print!("WGPU backend: ");
    let backend = adapter.get_info().backend;
    match backend
    {
        wgpu::Backend::Empty  => { println!("Empty"); }
        wgpu::Backend::Vulkan => { println!("Vulkan"); }
        wgpu::Backend::Metal  => { println!("Metal"); }
        wgpu::Backend::Dx12   => { println!("D3D12"); }
        wgpu::Backend::Gl     => { println!("OpenGL"); }
        wgpu::Backend::BrowserWebGpu => { println!("WebGPU"); }
    }
}
