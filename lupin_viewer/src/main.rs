
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

use ::egui::FontDefinitions;

pub use lupin as lp;

mod loader;
mod input;
mod ui;
pub use loader::*;
pub use input::*;
pub use ui::*;

fn main()
{
    let event_loop = EventLoop::builder().build().unwrap();
    let window_attributes = WindowAttributes::default()
        .with_title("Lupin Pathtracer")
        .with_inner_size(PhysicalSize::new(1920.0, 1080.0))
        .with_visible(false);

    let window = event_loop.create_window(window_attributes).unwrap();

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

    let mut app_state = AppState::new(&device, &queue, &window);

    let egui_ctx = egui::Context::default();
    let viewport_id = egui_ctx.viewport_id();
    let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None, None);
    let mut egui_renderer = egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, None, 1, true);

    let mut input = Input::default();

    let min_delta_time: f32 = 1.0/10.0;
    let mut delta_time: f32 = 1.0/60.0;
    let mut time_begin = Instant::now();

    window.set_visible(true);

    event_loop.run(|event, target|
    {
        process_input_event(&mut input, &event);

        if let Event::WindowEvent { window_id: _, event } = event
        {
            // Collect inputs
            let _ = egui_state.on_window_event(&window, &event);

            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    app_state.resize_callback(new_size.width, new_size.height);

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

                    let frame = surface.get_current_texture().unwrap();

                    app_state.update_and_render(&egui_ctx, &mut egui_state, &mut egui_renderer, &frame.texture, &input, delta_time);

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

#[derive(Default, Debug, PartialEq, Clone, Copy)]
pub enum RenderType
{
    #[default]
    Albedo,
    Normals,
    Pathtrace,
    TriChecks,
    AABBChecks,
    NumBounces,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum TonemapKind
{
    Aces,
    FilmicUC2,
    FilmicCustom,
}

fn to_tonemap_kind(op: lp::TonemapOperator) -> TonemapKind
{
    return match op
    {
        lp::TonemapOperator::Aces => TonemapKind::Aces,
        lp::TonemapOperator::FilmicUC2 => TonemapKind::FilmicUC2,
        lp::TonemapOperator::FilmicCustom {..} => TonemapKind::FilmicCustom,
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct DebugVizInfo
{
    pub multibounce: bool,
    pub heatmap_min: f32,
    pub heatmap_max: f32,
}

pub struct AppState<'a>
{
    pub device: &'a wgpu::Device,
    pub queue: &'a wgpu::Queue,
    pub window: &'a winit::window::Window,

    // UI
    pub render_type: RenderType,
    pub max_accums: u32,
    pub max_bounces: u32,
    pub samples_per_pixel: u32,
    pub tonemap_params: lp::TonemapParams,
    pub debug_viz: DebugVizInfo,
    pub show_normals_when_moving: bool,
    pub should_rebuild_pathtrace_resources: bool,
    pub controlling_camera: bool,

    // Camera
    pub cam_pos: lp::Vec3,
    pub cam_rot: lp::Quat,

    // Lupin resources,
    pub pathtrace_resources: lp::PathtraceResources,
    pub tonemap_resources: lp::TonemapResources,
    pub scene: lp::Scene,

    // Textures
    pub output_textures: [wgpu::Texture; 2],
    pub output_rgba8_unorm: wgpu::Texture,
    pub output_rgba8_snorm: wgpu::Texture,

    // Saved state for accumulation
    pub prev_cam_transform: lp::Mat4,
    pub accum_counter: u32,
    pub tile_idx: u32,
    pub output_tex_front: usize,
    pub output_tex_back: usize,
}

impl<'a> AppState<'a>
{
    pub fn new(device: &'a wgpu::Device, queue: &'a wgpu::Queue, window: &'a winit::window::Window) -> Self
    {
        const DEFAULT_SAMPLES_PER_PIXEL: u32 = 1;
        const DEFAULT_MAX_BOUNCES: u32 = 5;

        let pathtrace_resources = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
            with_runtime_checks: true,
            samples_per_pixel: DEFAULT_SAMPLES_PER_PIXEL,
            max_bounces: DEFAULT_MAX_BOUNCES,
        });
        let tonemap_resources = lp::build_tonemap_resources(&device);

        let width = window.inner_size().width;
        let height = window.inner_size().height;

        let scene = build_scene(&device, &queue);

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

        let output_rgba8_unorm = device.create_texture(&wgpu::TextureDescriptor {
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
        let output_rgba8_snorm = device.create_texture(&wgpu::TextureDescriptor {
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

        return Self {
            device: device,
            queue: queue,
            window: window,

            // UI
            render_type: Default::default(),
            samples_per_pixel: DEFAULT_SAMPLES_PER_PIXEL,
            max_bounces: DEFAULT_MAX_BOUNCES,
            max_accums: 200,
            tonemap_params: Default::default(),
            debug_viz: Default::default(),
            show_normals_when_moving: true,
            should_rebuild_pathtrace_resources: false,
            controlling_camera: false,

            // Camera
            cam_pos: lp::Vec3 { x: 0.0, y: 1.0, z: -3.0 },
            cam_rot: Default::default(),

            // Lupin resources
            pathtrace_resources,
            tonemap_resources,
            scene,

            // Textures
            output_textures,
            output_rgba8_unorm,
            output_rgba8_snorm,

            // Saved state for accumulation
            prev_cam_transform: lp::Mat4::zeros(),
            accum_counter: 0,
            tile_idx: 0,
            output_tex_front: 1,
            output_tex_back: 0,
        };
    }

    pub fn update_and_render(&mut self, egui_ctx: &egui::Context, egui_state: &mut egui_winit::State, egui_renderer: &mut egui_wgpu::Renderer, swapchain: &wgpu::Texture, input: &Input, delta_time: f32)
    {
        // Update
        self.update_camera(input, delta_time);

        self.controlling_camera = input.rmouse.pressing;
        if input.rmouse.pressing {
            self.reset_accumulation();
        }

        // Consume the accumulated egui inputs
        let egui_input = egui_state.take_egui_input(&self.window);

        // Update UI
        let egui_output = egui_ctx.run(egui_input, |ui|
        {
            self.update_ui(egui_ctx);
        });

        if self.should_rebuild_pathtrace_resources
        {
            self.pathtrace_resources = lp::build_pathtrace_resources(&self.device, &lp::BakedPathtraceParams {
                with_runtime_checks: true,
                samples_per_pixel: self.samples_per_pixel,
                max_bounces: self.max_bounces,
            });
            self.reset_accumulation();
            self.should_rebuild_pathtrace_resources = false;
        }

        // Render scene
        self.render_scene(swapchain);

        self.render_egui(egui_ctx, egui_state, egui_renderer, swapchain, egui_output);
    }

    fn render_egui(&mut self, egui_ctx: &egui::Context, egui_state: &mut egui_winit::State, egui_renderer: &mut egui_wgpu::Renderer, swapchain: &wgpu::Texture, egui_output: egui::FullOutput)
    {
        let swapchain_view = swapchain.create_view(&Default::default());

        egui_state.handle_platform_output(&self.window, egui_output.platform_output);
        let tris = egui_ctx.tessellate(egui_output.shapes, egui_output.pixels_per_point);

        let win_width = self.window.inner_size().width.max(1) as u32;
        let win_height = self.window.inner_size().height.max(1) as u32;

        for (id, image_delta) in &egui_output.textures_delta.set
        {
            egui_renderer.update_texture(&self.device, &self.queue, *id, &image_delta);
        }

        {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            let screen_descriptor = egui_wgpu::ScreenDescriptor
            {
                size_in_pixels: [win_width, win_height],
                pixels_per_point: self.window.scale_factor() as f32,
            };
            egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &tris, &screen_descriptor);

            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor
            {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment
                {
                    view: &swapchain_view,
                    resolve_target: None,
                    ops: wgpu::Operations
                    {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                label: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Apparently the egui_wgpu needs to get around the borrow-checker like this...
            let mut pass_static = pass.forget_lifetime();
            egui_renderer.render(&mut pass_static, &tris, &screen_descriptor);
            drop(pass_static);

            let cmd_buf = encoder.finish();
            self.queue.submit(Some(cmd_buf));
        }

        for x in &egui_output.textures_delta.free
        {
            egui_renderer.free_texture(x)
        }
    }

    fn render_scene(&mut self, swapchain: &wgpu::Texture)
    {
        let camera_transform = lp::xform_to_matrix(self.cam_pos, self.cam_rot, lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 });
        if camera_transform.m != self.prev_cam_transform.m
        {
            self.reset_accumulation();
        }
        self.prev_cam_transform = camera_transform;

        match self.render_type
        {
            RenderType::Albedo =>
            {
                lp::raycast_albedo(&self.device, &self.queue, &self.scene, &self.output_rgba8_unorm,
                                   &self.pathtrace_resources, camera_transform.into());
                lp::convert_to_ldr_no_tonemap(&self.device, &self.queue, &self.tonemap_resources,
                                              &self.output_rgba8_unorm, &swapchain);
            }
            RenderType::Normals =>
            {
                lp::raycast_normals(&self.device, &self.queue, &self.scene, &self.output_rgba8_snorm,
                                    &self.pathtrace_resources, camera_transform.into());
                lp::convert_to_ldr_no_tonemap(&self.device, &self.queue, &self.tonemap_resources,
                                              &self.output_rgba8_snorm, &swapchain);
            }
            RenderType::Pathtrace =>
            {

                if self.controlling_camera
                {
                    if self.show_normals_when_moving
                    {
                        lp::raycast_normals(&self.device, &self.queue, &self.scene, &self.output_rgba8_snorm,
                                            &self.pathtrace_resources, camera_transform.into());
                        lp::convert_to_ldr_no_tonemap(&self.device, &self.queue, &self.tonemap_resources,
                                                      &self.output_rgba8_snorm, &swapchain);
                    }
                    else
                    {
                        if self.accum_counter < self.max_accums
                        {
                            lp::pathtrace_scene(&self.device, &self.queue, &lp::PathtraceDesc {
                                scene: &self.scene,
                                render_target: &self.output_textures[self.output_tex_front],
                                resources: &self.pathtrace_resources,
                                accum_params: &lp::AccumulationParams {
                                    prev_frame: Some(&self.output_textures[self.output_tex_back]),
                                    accum_counter: self.accum_counter,
                                },
                                tile_params: &Default::default(),
                                camera_params: &Default::default(),
                                camera_transform: camera_transform,
                            });
                        }

                        lp::apply_tonemapping(&self.device, &self.queue, &lp::TonemapDesc {
                            resources: &self.tonemap_resources,
                            hdr_texture: &self.output_textures[self.output_tex_front],
                            render_target: &swapchain,
                            tonemap_params: &self.tonemap_params,
                        });
                    }
                }
                else
                {
                    if self.accum_counter < self.max_accums
                    {
                        lp::pathtrace_scene_tiles(&self.device, &self.queue, &lp::PathtraceDesc {
                            scene: &self.scene,
                            render_target: &self.output_textures[self.output_tex_front],
                            resources: &self.pathtrace_resources,
                            accum_params: &lp::AccumulationParams {
                                prev_frame: Some(&self.output_textures[self.output_tex_back]),
                                accum_counter: self.accum_counter,
                            },
                            tile_params: &Default::default(),
                            camera_params: &Default::default(),
                            camera_transform: camera_transform,
                        }, &mut self.tile_idx, 3);
                    }

                    lp::apply_tonemapping(&self.device, &self.queue, &lp::TonemapDesc {
                        resources: &self.tonemap_resources,
                        hdr_texture: &self.output_textures[self.output_tex_front],
                        render_target: &swapchain,
                        tonemap_params: &self.tonemap_params,
                    });
                }

                // Swap output textures
                if self.accum_counter < self.max_accums
                {
                    if self.tile_idx == 0
                    {
                        let tmp = self.output_tex_back;
                        self.output_tex_back  = self.output_tex_front;
                        self.output_tex_front = tmp;
                        self.accum_counter = (self.accum_counter + 1).min(self.max_accums);
                    }
                }
            }
            RenderType::TriChecks =>
            {
                let debug_desc = lp::DebugVizDesc {
                    viz_type: lp::DebugVizType::BVHTriChecks,
                    heatmap_min: self.debug_viz.heatmap_min,
                    heatmap_max: self.debug_viz.heatmap_max,
                    first_hit_only: !self.debug_viz.multibounce,
                };

                lp::pathtrace_scene_debug(&self.device, &self.queue, &self.scene, &self.output_rgba8_unorm,
                                          &self.pathtrace_resources, camera_transform.into(), &debug_desc);
                lp::convert_to_ldr_no_tonemap(&self.device, &self.queue, &self.tonemap_resources,
                                              &self.output_rgba8_unorm, &swapchain);
            }
            RenderType::AABBChecks =>
            {
                let debug_desc = lp::DebugVizDesc {
                    viz_type: lp::DebugVizType::BVHAABBChecks,
                    heatmap_min: self.debug_viz.heatmap_min,
                    heatmap_max: self.debug_viz.heatmap_max,
                    first_hit_only: !self.debug_viz.multibounce,
                };

                lp::pathtrace_scene_debug(&self.device, &self.queue, &self.scene, &self.output_rgba8_unorm,
                                          &self.pathtrace_resources, camera_transform.into(), &debug_desc);
                lp::convert_to_ldr_no_tonemap(&self.device, &self.queue, &self.tonemap_resources,
                                              &self.output_rgba8_unorm, &swapchain);
            }
            RenderType::NumBounces =>
            {
                let debug_desc = lp::DebugVizDesc {
                    viz_type: lp::DebugVizType::NumBounces,
                    heatmap_min: self.debug_viz.heatmap_min,
                    heatmap_max: self.debug_viz.heatmap_max,
                    first_hit_only: !self.debug_viz.multibounce,
                };

                lp::pathtrace_scene_debug(&self.device, &self.queue, &self.scene, &self.output_rgba8_unorm,
                                          &self.pathtrace_resources, camera_transform.into(), &debug_desc);
                lp::convert_to_ldr_no_tonemap(&self.device, &self.queue, &self.tonemap_resources,
                                              &self.output_rgba8_unorm, &swapchain);
            }
        }
    }

    fn update_camera(&mut self, input: &Input, delta_time: f32)
    {
        fn deg_to_rad(degrees: f32) -> f32 {
            return degrees * std::f32::consts::PI / 180.0;
        }

        let mouse_sensitivity = deg_to_rad(0.2);
        static mut ANGLE: lp::Vec2 = lp::Vec2 { x: 0.0, y: 0.0 };
        let mut mouse = lp::Vec2::default();
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
            let y_rot = lp::angle_axis(lp::Vec3 { x: -1.0, y: 0.0, z: 0.0 }, ANGLE.y);
            let x_rot = lp::angle_axis(lp::Vec3 { x:  0.0, y: 1.0, z: 0.0 }, ANGLE.x);
            self.cam_rot = x_rot * y_rot;
        }

        // Movement
        static mut CUR_VEL: lp::Vec3 = lp::Vec3 { x: 0.0, y: 0.0, z: 0.0 };
        let move_speed: f32 = 6.0;
        let move_speed_fast: f32 = 15.0;
        let move_speed_slow: f32 = 0.5;
        let move_accel: f32 = 100.0;

        let cur_move_speed = if input.keys[Key::LSHIFT].pressing {
            move_speed_fast
        } else if input.keys[Key::LCTRL].pressing {
            move_speed_slow
        } else {
            move_speed
        };

        let mut keyboard_dir_xz = lp::Vec3::default();
        let mut keyboard_dir_y: f32 = 0.0;
        if input.rmouse.pressing
        {
            keyboard_dir_xz.x = ((input.keys[Key::D].pressing) as i32 - (input.keys[Key::A].pressing) as i32) as f32;
            keyboard_dir_xz.z = ((input.keys[Key::W].pressing) as i32 - (input.keys[Key::S].pressing) as i32) as f32;
            keyboard_dir_y    = ((input.keys[Key::E].pressing) as i32 - (input.keys[Key::Q].pressing) as i32) as f32;

            // It's a "direction" input so its length
            // should be no more than 1
            if lp::dot_vec3(keyboard_dir_xz, keyboard_dir_xz) > 1.0 {
                keyboard_dir_xz = lp::normalize_vec3(keyboard_dir_xz)
            }

            if keyboard_dir_y.abs() > 1.0 {
                keyboard_dir_y = keyboard_dir_y.signum();
            }
        }

        let mut target_vel = keyboard_dir_xz * cur_move_speed;
        target_vel = lp::vec3_quat_mul(self.cam_rot, target_vel);
        target_vel.y += keyboard_dir_y * cur_move_speed;

        unsafe {
            CUR_VEL = approach_linear(CUR_VEL, target_vel, move_accel * delta_time);
            self.cam_pos += CUR_VEL * delta_time;
        }

        fn approach_linear(cur: lp::Vec3, target: lp::Vec3, delta: f32) -> lp::Vec3
        {
            let diff = target - cur;
            let dist = lp::magnitude_vec3(diff);

            if dist <= delta { return target; }
            return cur + diff / dist * delta;
        }
    }

    fn resize_callback(&mut self, new_width: u32, new_height: u32)
    {
        self.reset_accumulation();
        resize_texture(&self.device, &mut self.output_textures[0], new_width, new_height);
        resize_texture(&self.device, &mut self.output_textures[1], new_width, new_height);
        resize_texture(&self.device, &mut self.output_rgba8_unorm, new_width, new_height);
        resize_texture(&self.device, &mut self.output_rgba8_snorm, new_width, new_height);
    }

    fn update_ui(&mut self, egui_ctx: &egui::Context)
    {
        egui::SidePanel::left("backend_panel").resizable(false).show(egui_ctx, |ui| {
            ui.add_space(4.0);
            ui.vertical_centered(|ui| {
                ui.heading("Settings");
            });

            ui.separator();
            ui.add_space(12.0);

            ui.heading("Visualization:");
            ui.add_space(4.0);
            {
                let changed = self.ui_render_type(ui);

                match self.render_type
                {
                    RenderType::Albedo => {}
                    RenderType::Normals => {}
                    RenderType::Pathtrace =>
                    {
                        ui.horizontal(|ui| {
                            let response = ui.add(egui::DragValue::new(&mut self.samples_per_pixel).range(1..=200));
                            if response.changed() { self.should_rebuild_pathtrace_resources = true; }
                            ui.label("Samples per pixel");
                        });

                        ui.horizontal(|ui| {
                            let response = ui.add(egui::DragValue::new(&mut self.max_bounces).range(1..=200));
                            if response.changed() { self.should_rebuild_pathtrace_resources = true; }
                            ui.label("Max bounces");
                        });

                        ui.horizontal(|ui| {
                            let response = ui.add(egui::DragValue::new(&mut self.max_accums).range(1..=10000));
                            ui.label("Max accumulations");
                        });

                        ui.checkbox(&mut self.show_normals_when_moving, "Show normals when moving");

                        egui::CollapsingHeader::new("Stats").show(ui, |ui| {
                            ui.label(format!("Iteration: {:?}", self.accum_counter));
                            ui.label(format!("Tile: {:?}", self.tile_idx));
                        });
                    }
                    RenderType::TriChecks =>
                    {
                        ui.checkbox(&mut self.debug_viz.multibounce, "Multiple bounces");

                        if changed {
                            (self.debug_viz.heatmap_min, self.debug_viz.heatmap_max) = default_tri_check_heatmap_params();
                        }

                        self.ui_heatmap_params(ui);
                    }
                    RenderType::AABBChecks =>
                    {
                        ui.checkbox(&mut self.debug_viz.multibounce, "Multiple bounces");

                        if changed {
                            (self.debug_viz.heatmap_min, self.debug_viz.heatmap_max) = default_aabb_check_heatmap_params();
                        }

                        self.ui_heatmap_params(ui);
                    }
                    RenderType::NumBounces =>
                    {
                        if changed {
                            (self.debug_viz.heatmap_min, self.debug_viz.heatmap_max) = default_num_bounces_heatmap_params();
                        }

                        self.ui_heatmap_params(ui);
                    }
                }
            }

            ui.add_space(12.0);
            ui.separator();
            ui.add_space(12.0);

            ui.heading("Tonemapping:");
            ui.add_space(4.0);
            {
                self.ui_tonemap_operator(ui);

                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut self.tonemap_params.exposure).speed(0.05));
                    ui.label("Exposure");
                });
            }
        });
    }

    fn ui_render_type(&mut self, ui: &mut egui::Ui) -> bool
    {
        let old_render_type = self.render_type;
        egui::ComboBox::from_label("Visualization")
            .selected_text(format!("{:?}", self.render_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.render_type, RenderType::Albedo, "Albedo");
                ui.selectable_value(&mut self.render_type, RenderType::Normals, "Normals");
                ui.selectable_value(&mut self.render_type, RenderType::Pathtrace, "Pathtrace");
                ui.selectable_value(&mut self.render_type, RenderType::TriChecks, "Triangle checks (Debug)");
                ui.selectable_value(&mut self.render_type, RenderType::AABBChecks, "Box checks (Debug)");
                ui.selectable_value(&mut self.render_type, RenderType::NumBounces, "Number of bounces (Debug)");
            });
        return self.render_type != old_render_type;
    }

    fn ui_heatmap_params(&mut self, ui: &mut egui::Ui)
    {
        ui_min_max(ui, "Heatmap:", &mut self.debug_viz.heatmap_min, &mut self.debug_viz.heatmap_max, 0.0..=200.0);
    }

    fn ui_tonemap_operator(&mut self, ui: &mut egui::Ui)
    {
        let mut tonemap_kind = to_tonemap_kind(self.tonemap_params.operator);

        egui::ComboBox::from_label("Operator")
            .selected_text(format!("{:?}", tonemap_kind))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut tonemap_kind, TonemapKind::Aces, "Aces");
                ui.selectable_value(&mut tonemap_kind, TonemapKind::FilmicUC2, "Filmic (Uncharted 2)");
                ui.selectable_value(&mut tonemap_kind, TonemapKind::FilmicCustom, "Filmic (Custom)");
            });

        match tonemap_kind
        {
            TonemapKind::Aces =>
            {
                self.tonemap_params.operator = lp::TonemapOperator::Aces;
            },
            TonemapKind::FilmicUC2 =>
            {
                self.tonemap_params.operator = lp::TonemapOperator::FilmicUC2;
            },
            TonemapKind::FilmicCustom =>
            {
                let mut linear_white_ = 0.0;
                let mut a_ = 0.0;
                let mut b_ = 0.0;
                let mut c_ = 0.0;
                let mut d_ = 0.0;
                let mut e_ = 0.0;
                let mut f_ = 0.0;

                match self.tonemap_params.operator
                {
                    lp::TonemapOperator::FilmicCustom {linear_white, a, b, c, d, e, f} =>
                    {
                        linear_white_ = linear_white;
                        a_ = a;
                        b_ = b;
                        c_ = c;
                        d_ = d;
                        e_ = e;
                        f_ = f;
                    },
                    _ => {}
                }

                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut linear_white_).speed(0.05));
                    ui.label("Linear white");
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut a_).speed(0.05));
                    ui.label("A");
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut b_).speed(0.05));
                    ui.label("B");
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut c_).speed(0.05));
                    ui.label("C");
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut d_).speed(0.05));
                    ui.label("D");
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut e_).speed(0.05));
                    ui.label("E");
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut f_).speed(0.05));
                    ui.label("F");
                });

                self.tonemap_params.operator = lp::TonemapOperator::FilmicCustom {
                    linear_white: linear_white_,
                    a: a_,
                    b: b_,
                    c: c_,
                    d: d_,
                    e: e_,
                    f: f_,
                };
            },
        }
    }

    fn reset_accumulation(&mut self)
    {
        self.accum_counter = 0;
        self.tile_idx = 0;
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

fn default_tri_check_heatmap_params() -> (f32, f32)
{
    return (0.0, 10.0);
}

fn default_aabb_check_heatmap_params() -> (f32, f32)
{
    return (0.0, 1000.0);
}

fn default_num_bounces_heatmap_params() -> (f32, f32)
{
    return (0.0, 5.0);
}

fn ui_min_max(ui: &mut egui::Ui, string: &str, min: &mut f32, max: &mut f32, range: std::ops::RangeInclusive<f32>)
{
    let (range_start, range_end) = (*range.start(), *range.end());

    ui.horizontal(|ui| {
        ui.label(string);
        ui.label("Min:");
        ui.add(
            egui::DragValue::new(min)
                .range(range_start..=*max)
                .speed(0.1),
        );
        ui.label("Max:");
        ui.add(
            egui::DragValue::new(max)
                .range(*min..=range_end)
                .speed(0.1),
        );
    });
}
