
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

// Don't spawn a terminal window on windows
//#![windows_subsystem = "windows"]

const FORCE_SW_RT: bool = false;

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
use lupin::wgpu as wgpu;

mod input;
mod ui;
pub use input::*;
pub use ui::*;

pub use lupin_loader as lpl;

fn main()
{
    let event_loop = EventLoop::new().unwrap();
    let window_attributes = WindowAttributes::default()
        .with_title("Lupin Pathtracer")
        .with_inner_size(PhysicalSize::new(1920.0, 1080.0))
        .with_visible(false);

    let window = event_loop.create_window(window_attributes).unwrap();

    let win_size = window.inner_size();
    let (width, height) = (win_size.width as i32, win_size.height as i32);

    let mut surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: wgpu::TextureFormat::Rgba8Unorm,
        width: width as u32,
        height: height as u32,
        present_mode: wgpu::PresentMode::AutoNoVsync,  // Disabling vsync makes it so tiled rendering isn't slowed down.
        desired_maximum_frame_latency: 0,
        alpha_mode: wgpu::CompositeAlphaMode::Auto,
        view_formats: vec![],
    };

    let (device, queue, surface, adapter, denoise_device) = lp::init_default_wgpu_context_with_denoising_capabilities(&surface_config, &window, width, height);

    let mut app_state = AppState::new(&device, &denoise_device, &queue, &window);

    let egui_ctx = egui::Context::default();
    let viewport_id = egui_ctx.viewport_id();
    let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None, None);
    let mut egui_renderer = egui_wgpu::Renderer::new(&device, wgpu::TextureFormat::Rgba8Unorm, Default::default());

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

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum RenderType
{
    Pathtrace,
    Falsecolor(lp::FalsecolorType),
    Debug(lp::DebugVizType),
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct DebugVizInfo
{
    pub multibounce: bool,
    pub heatmap_min: f32,
    pub heatmap_max: f32,
}

pub struct AppState<'a>
{
    pub device: &'a wgpu::Device,
    pub denoise_device: &'a lp::DenoiseDevice,
    pub queue: &'a wgpu::Queue,
    pub window: &'a winit::window::Window,

    // UI
    pub render_type: RenderType,
    pub pathtrace_type: lp::PathtraceType,
    pub max_accums: u32,
    pub max_bounces: u32,
    pub samples_per_pixel: u32,
    pub filmic: bool,
    pub srgb: bool,
    pub exposure: f32,
    pub debug_viz: DebugVizInfo,
    pub show_normals_when_moving: bool,
    pub should_rebuild_pathtrace_resources: bool,
    pub controlling_camera: bool,
    pub camera_params: lp::CameraParams,
    pub keep_aspect_ratio: bool,
    pub ui_panel_width: u32,
    pub cam_speed_multiplier: f32,
    pub tiled_rendering: bool,
    pub tile_params: lp::TileParams,
    pub denoising: bool,
    pub use_gbuffers_for_denoise: bool,
    pub max_radiance: f32,

    // Camera
    pub cam_pos: lp::Vec3,
    pub cam_rot: lp::Quat,
    pub cam_angles: lp::Vec2,
    pub cam_vel: lp::Vec3,
    pub cam_transform: lp::Mat3x4,

    // Lupin resources
    pub pathtrace_resources: lp::PathtraceResources,
    pub tonemap_resources: lp::TonemapResources,
    pub denoise_resources: lp::DenoiseResources,
    pub scene: lp::Scene,
    pub scene_cameras: Vec<lpl::SceneCamera>,
    pub selected_cam: i32,  // If -1, no cam is selected (free-roam).

    // Textures
    pub output: DoubleBufferedTexture,
    pub albedo: DoubleBufferedTexture,
    pub normals: DoubleBufferedTexture,
    pub denoised: wgpu::Texture,

    // Saved state for accumulation
    pub prev_cam_transform: lp::Mat3x4,
    pub accum_counter: u32,
    pub gbuffers_accum_counter: u32,
}

impl<'a> AppState<'a>
{
    pub fn new(device: &'a wgpu::Device, denoise_device: &'a lp::DenoiseDevice, queue: &'a wgpu::Queue, window: &'a winit::window::Window) -> Self
    {
        const DEFAULT_SAMPLES_PER_PIXEL: u32 = 5;
        const DEFAULT_MAX_BOUNCES: u32 = 8;

        let pathtrace_resources = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
            with_runtime_checks: false,
            samples_per_pixel: DEFAULT_SAMPLES_PER_PIXEL,
            max_bounces: DEFAULT_MAX_BOUNCES,
        });
        let tonemap_resources = lp::build_tonemap_resources(&device);

        let width = 1920;
        let height = 1080;

        let denoise_resources = lp::build_denoise_resources(&device, &denoise_device, width, height);

        let (scene, scene_cameras) = lpl::build_scene_cornell_box(&device, &queue, FORCE_SW_RT);

        let output = DoubleBufferedTexture::create(device, &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[]
        });

        let albedo = DoubleBufferedTexture::create(device, &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[]
        });

        let normals = DoubleBufferedTexture::create(device, &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[]
        });

        let denoised = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[]
        });

        let num_cameras = scene_cameras.len();

        let mut res = Self {
            device: device,
            denoise_device: denoise_device,
            queue: queue,
            window: window,

            // UI
            render_type: RenderType::Falsecolor(lp::FalsecolorType::Albedo),
            pathtrace_type: Default::default(),
            samples_per_pixel: DEFAULT_SAMPLES_PER_PIXEL,
            max_bounces: DEFAULT_MAX_BOUNCES,
            max_accums: 200,
            filmic: false,
            srgb: true,
            exposure: 0.0,
            debug_viz: Default::default(),
            show_normals_when_moving: true,
            should_rebuild_pathtrace_resources: false,
            controlling_camera: false,
            camera_params: Default::default(),
            selected_cam: -1,
            keep_aspect_ratio: true,
            ui_panel_width: 0,
            cam_speed_multiplier: 1.0,
            tiled_rendering: false,
            tile_params: Default::default(),
            denoising: false,
            use_gbuffers_for_denoise: false,
            max_radiance: 100.0,

            // Camera
            cam_pos: Default::default(),
            cam_rot: Default::default(),
            cam_angles: Default::default(),
            cam_vel: Default::default(),
            cam_transform: Default::default(),

            // Lupin resources
            pathtrace_resources,
            tonemap_resources,
            denoise_resources,
            scene,
            scene_cameras,

            // Textures
            output,
            albedo,
            normals,
            denoised,

            // Saved state for accumulation
            prev_cam_transform: lp::Mat3x4::zeros(),
            accum_counter: 0,
            gbuffers_accum_counter: 0,
        };

        res.switch_to_cam(if num_cameras > 0 { 0 } else { -1 });

        return res;
    }

    pub fn update_and_render(&mut self, egui_ctx: &egui::Context, egui_state: &mut egui_winit::State, egui_renderer: &mut egui_wgpu::Renderer, swapchain: &wgpu::Texture, input: &Input, delta_time: f32)
    {
        // Update
        if self.selected_cam == -1  // Free-roam
        {
            self.update_camera(input, delta_time);
            self.cam_transform = lp::xform_to_matrix(self.cam_pos, self.cam_rot, lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 });
        }

        self.controlling_camera = input.rmouse.pressing;
        if input.rmouse.pressing
        {
            self.reset_accumulation();
            if self.selected_cam != -1 {
                self.switch_to_freeroam();
            }
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
                    depth_slice: None,
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
        if self.cam_transform.m != self.prev_cam_transform.m {
            self.reset_accumulation();
        }
        self.prev_cam_transform = self.cam_transform;

        let viewport = lp::Viewport {
            x: self.ui_panel_width as f32,
            y: 0.0,
            w: swapchain.size().width as f32 - self.ui_panel_width as f32,
            h: swapchain.size().height as f32,
        };

        let mut desc = lp::PathtraceDesc {
            accum_params: Some(lp::AccumulationParams {
                prev_frame: self.output.back(),
                accum_counter: self.accum_counter,
            }),
            tile_params: Some(&self.tile_params),
            camera_params: self.camera_params,
            camera_transform: self.cam_transform,
            force_software_bvh: FORCE_SW_RT,
            advanced: lp::AdvancedParams {
                max_radiance: self.max_radiance,
            }
        };
        let desc_albedo = lp::PathtraceDesc {
            accum_params: Some(lp::AccumulationParams {
                prev_frame: self.albedo.back(),
                accum_counter: self.gbuffers_accum_counter,
            }),
            tile_params: Some(&self.tile_params),
            camera_params: self.camera_params,
            camera_transform: self.cam_transform,
            force_software_bvh: FORCE_SW_RT,
            advanced: lp::AdvancedParams {
                max_radiance: self.max_radiance,
            }
        };
        let desc_normals = lp::PathtraceDesc {
            accum_params: Some(lp::AccumulationParams {
                prev_frame: self.normals.back(),
                accum_counter: self.gbuffers_accum_counter,
            }),
            tile_params: Some(&self.tile_params),
            camera_params: self.camera_params,
            camera_transform: self.cam_transform,
            force_software_bvh: FORCE_SW_RT,
            advanced: lp::AdvancedParams {
                max_radiance: self.max_radiance,
            }
        };

        let mut desc_no_tiles = desc;
        desc_no_tiles.tile_params = None;
        let mut desc_albedo_no_tiles = desc_albedo;
        desc_albedo_no_tiles.tile_params = None;
        let mut desc_normals_no_tiles = desc_normals;
        desc_normals_no_tiles.tile_params = None;

        let mut tonemap_desc = lp::TonemapDesc {
            viewport: Some(viewport),
            exposure: self.exposure,
            filmic: self.filmic,
            srgb: self.srgb
        };

        let mut render_type = self.render_type;
        if render_type == RenderType::Pathtrace && self.controlling_camera && self.show_normals_when_moving {
            render_type = RenderType::Falsecolor(lp::FalsecolorType::NormalsUnsigned);
        }

        if self.controlling_camera || !self.tiled_rendering {
            desc.tile_params = None;
        }

        match render_type
        {
            RenderType::Falsecolor(falsecolor_type) =>
            {
                lp::pathtrace_scene_falsecolor(self.device, self.queue, &self.pathtrace_resources, &self.scene, self.output.front(),
                                               falsecolor_type, &desc);
                tonemap_desc.filmic = false;
                tonemap_desc.srgb = false;
            }
            RenderType::Pathtrace =>
            {
                if self.accum_counter < self.max_accums
                {
                    lp::pathtrace_scene(self.device, self.queue, &self.pathtrace_resources, &self.scene, self.output.front(),
                                        self.pathtrace_type, &desc);
                    if self.denoising
                    {
                        lp::pathtrace_scene_falsecolor(self.device, self.queue, &self.pathtrace_resources, &self.scene, self.albedo.front(),
                                                       lp::FalsecolorType::Albedo, &desc_albedo_no_tiles);
                        lp::pathtrace_scene_falsecolor(self.device, self.queue, &self.pathtrace_resources, &self.scene, self.normals.front(),
                                                       lp::FalsecolorType::Normals, &desc_albedo_no_tiles);
                    }
                }
            }
            RenderType::Debug(debug_viz_type) =>
            {
                let debug_desc = lp::DebugVizDesc {
                    viz_type: debug_viz_type,
                    heatmap_min: self.debug_viz.heatmap_min,
                    heatmap_max: self.debug_viz.heatmap_max,
                    first_hit_only: !self.debug_viz.multibounce,
                };
                lp::pathtrace_scene_debug(self.device, self.queue, &self.pathtrace_resources, &self.scene, self.output.front(),
                                    &debug_desc, &desc);
            }
        }

        // Denoise
        let do_denoise = self.denoising && self.tile_params.tile_idx == 0 && self.accum_counter > 30;
        if do_denoise
        {
            if self.use_gbuffers_for_denoise
            {
                lp::denoise(self.device, self.queue, self.denoise_device, &mut self.denoise_resources, &lp::DenoiseDesc {
                    pathtrace_output: self.output.front(),
                    albedo: Some(self.albedo.front()),
                    normals: Some(self.normals.front()),
                    denoise_output: &self.denoised,
                    quality: Default::default(),
                });
            }
            else
            {
                lp::denoise(self.device, self.queue, self.denoise_device, &mut self.denoise_resources, &lp::DenoiseDesc {
                    pathtrace_output: self.output.front(),
                    albedo: None,
                    normals: None,
                    denoise_output: &self.denoised,
                    quality: Default::default(),
                });
            }
        }

        // Tonemap
        {
            let src = if do_denoise { &self.denoised } else { self.output.front() };
            lp::tonemap_and_fit_aspect(&self.device, &self.queue, &self.tonemap_resources, src, &swapchain, &tonemap_desc);
        }

        // Increment tile_idx
        if self.tiled_rendering && !self.controlling_camera
        {
            let width = self.output.front().width();
            let height = self.output.front().height();
            self.tile_params.tile_idx = (self.tile_params.tile_idx + 1) % lp::get_num_tiles(self.tile_params.tile_size, width, height);
        }

        // Swap output textures
        if self.tile_params.tile_idx == 0 && self.accum_counter < self.max_accums
        {
            // Copy the contents of the front buffer onto the back buffer,
            // so that when we swap buffers it won't be as jarring.
            if self.tiled_rendering
            {
                self.output.copy_front_to_back(self.device, self.queue);
                if self.denoising
                {
                    self.albedo.copy_front_to_back(self.device, self.queue);
                    self.normals.copy_front_to_back(self.device, self.queue);
                }
            }

            self.output.flip();
            if self.denoising
            {
                self.albedo.flip();
                self.normals.flip();
            }

            self.accum_counter = (self.accum_counter + 1).min(self.max_accums);
            if self.denoising
            {
                self.gbuffers_accum_counter = (self.gbuffers_accum_counter + 1).min(self.max_accums);
            }
        }
    }

    fn update_camera(&mut self, input: &Input, delta_time: f32)
    {
        fn deg_to_rad(degrees: f32) -> f32 {
            return degrees * std::f32::consts::PI / 180.0;
        }

        let mouse_sensitivity = deg_to_rad(0.2);
        let mut mouse = lp::Vec2::default();
        if input.rmouse.pressing
        {
            mouse.x = input.mouse_dx * mouse_sensitivity;
            mouse.y = input.mouse_dy * mouse_sensitivity;
        }

        self.cam_angles += mouse;
        // Wrap self.cam_angles.x
        while self.cam_angles.x < 0.0                        { self.cam_angles.x += 2.0 * std::f32::consts::PI; }
        while self.cam_angles.x > 2.0 * std::f32::consts::PI { self.cam_angles.x -= 2.0 * std::f32::consts::PI; }

        self.cam_angles.y = self.cam_angles.y.clamp(deg_to_rad(-90.0), deg_to_rad(90.0));
        let y_rot = lp::angle_axis(lp::Vec3 { x: -1.0, y: 0.0, z: 0.0 }, self.cam_angles.y);
        let x_rot = lp::angle_axis(lp::Vec3 { x:  0.0, y: 1.0, z: 0.0 }, self.cam_angles.x);
        self.cam_rot = x_rot * y_rot;

        // Movement
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

        self.cam_vel = approach_linear(self.cam_vel, target_vel, move_accel * delta_time);
        self.cam_pos += self.cam_vel * self.cam_speed_multiplier * delta_time;

        fn approach_linear(cur: lp::Vec3, target: lp::Vec3, delta: f32) -> lp::Vec3
        {
            let diff = target - cur;
            let dist = lp::magnitude_vec3(diff);

            if dist <= delta { return target; }
            return cur + diff / dist * delta;
        }
    }

    fn resize_output_textures(&mut self, new_width: u32, new_height: u32)
    {
        self.reset_accumulation();
        self.output.resize(self.device, new_width, new_height);
        self.albedo.resize(self.device, new_width, new_height);
        self.normals.resize(self.device, new_width, new_height);
        self.denoise_resources = lp::build_denoise_resources(self.device, self.denoise_device, new_width, new_height);
        resize_texture(self.device, &mut self.denoised, new_width, new_height);
    }

    fn update_ui(&mut self, egui_ctx: &egui::Context)
    {
        let panel = egui::SidePanel::left("settings_panel").resizable(false).show(egui_ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                ui.add_space(4.0);
                ui.vertical_centered(|ui| {
                    ui.heading("Settings");
                });

                ui.separator();
                ui.add_space(12.0);

                ui.heading("Visualization:");
                ui.add_space(4.0);
                {
                    let pt_changed = self.ui_pathtrace_type(ui);
                    if pt_changed { self.reset_accumulation(); }
                    let rt_changed = self.ui_render_type(ui);
                    if rt_changed { self.reset_accumulation(); }

                    let old_debug_desc = self.debug_viz;

                    match self.render_type
                    {
                        RenderType::Pathtrace =>
                        {
                            ui.checkbox(&mut self.show_normals_when_moving, "Show normals when moving");
                            ui.horizontal(|ui| {
                                let response = ui.add(egui::DragValue::new(&mut self.max_radiance).range(0.0..=100000000.0));
                                if response.changed() { self.reset_accumulation(); }
                                ui.label("Max radiance");
                            });
                        }
                        RenderType::Debug(lp::DebugVizType::BVHTriChecks) =>
                        {
                            ui.checkbox(&mut self.debug_viz.multibounce, "Multiple bounces");

                            if rt_changed {
                                (self.debug_viz.heatmap_min, self.debug_viz.heatmap_max) = default_tri_check_heatmap_params();
                            }

                            self.ui_heatmap_params(ui);
                        }
                        RenderType::Debug(lp::DebugVizType::BVHAABBChecks) =>
                        {
                            ui.checkbox(&mut self.debug_viz.multibounce, "Multiple bounces");

                            if rt_changed {
                                (self.debug_viz.heatmap_min, self.debug_viz.heatmap_max) = default_aabb_check_heatmap_params();
                            }

                            self.ui_heatmap_params(ui);
                        }
                        RenderType::Debug(lp::DebugVizType::NumBounces) =>
                        {
                            if rt_changed {
                                (self.debug_viz.heatmap_min, self.debug_viz.heatmap_max) = default_num_bounces_heatmap_params();
                            }

                            self.ui_heatmap_params(ui);
                        }
                        _ => {}
                    }

                    if self.debug_viz != old_debug_desc {
                        self.reset_accumulation();
                    }

                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.cam_speed_multiplier).range(0.1..=1000.0).speed(0.01));
                        ui.label("Camera speed multiplier");
                    });
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);

                ui.heading("Pathtrace settings:");
                ui.add_space(4.0);
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
                        ui.add(egui::DragValue::new(&mut self.max_accums).range(1..=10000));
                        ui.label("Max accumulations");
                    });

                    let response = ui.checkbox(&mut self.tiled_rendering, "Tiled Rendering");
                    if response.changed() { self.reset_accumulation(); }
                    if self.tiled_rendering
                    {
                        ui.horizontal(|ui| {
                            let response = ui.add(egui::DragValue::new(&mut self.tile_params.tile_size).range(10..=512).speed(1));
                            if response.changed() { self.reset_accumulation(); }
                            ui.label("Tile size");
                        });
                    }

                    ui.checkbox(&mut self.denoising, "Denoise");

                    if self.denoising {
                        let response = ui.checkbox(&mut self.use_gbuffers_for_denoise, "Use GBuffers");
                        if response.changed() { self.reset_accumulation(); }
                    }

                    egui::CollapsingHeader::new("Stats").id_salt("stats_pathtrace").show(ui, |ui| {
                        ui.label(format!("Iteration: {:?}", self.accum_counter));
                        ui.label(format!("Tile: {:?}", self.tile_params.tile_idx));
                    });

                    ui.add(egui::ProgressBar::new(self.accum_counter as f32 / self.max_accums as f32)
                        .text(format!("{}/{}", self.accum_counter, self.max_accums)));
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);

                let old_camera_params = self.camera_params;

                ui.heading("Camera:");
                ui.add_space(4.0);
                {
                    ui.checkbox(&mut self.camera_params.is_orthographic, "Orthographic");

                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.camera_params.lens).range(0.0..=5.0).speed(0.0001));
                        ui.label("Lens");
                    });

                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.camera_params.film).range(0.0..=5.0).speed(0.0001));
                        ui.label("Film");
                    });

                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.camera_params.aspect).range(0.0..=100.0).speed(0.01));
                        ui.label("Aspect");
                    });

                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.camera_params.focus).range(0.0..=10000000000.0).speed(0.1));
                        ui.label("Focus");
                    });

                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.camera_params.aperture).range(0.0..=5.0).speed(0.0001));
                        ui.label("Aperture");
                    });

                    if ui.button("Reset").clicked() {
                        self.camera_params = Default::default();
                    }
                }

                if old_camera_params != self.camera_params
                {
                    self.reset_accumulation();
                    self.switch_to_freeroam();
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);

                ui.heading("Scene:");
                ui.add_space(4.0);
                {
                    if ui.button("Load scene").clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("json", &["json"])
                            .pick_file()
                        {
                            if let Ok(res) = lpl::load_scene_yoctogl_v24(&path, self.device, self.queue, FORCE_SW_RT)
                            {
                                (self.scene, self.scene_cameras) = res;
                                self.reset_accumulation();
                                if !self.scene_cameras.is_empty() {
                                    self.switch_to_cam(0);
                                }
                            }
                            else
                            {
                                println!("Failed to load json file!");
                            }
                        }
                    }

                    for i in 0..self.scene_cameras.len()
                    {
                        let selected = self.selected_cam == i as i32;
                        if ui.add(egui::RadioButton::new(selected, format!("Camera {}", i+1))).clicked() {
                            self.switch_to_cam(i as i32);
                        }
                    }

                    ui.label("Right-click to enter free-roam mode. (First-person camera controls)");

                    egui::CollapsingHeader::new("Stats").id_salt("stats_scene").show(ui, |ui| {
                        ui.label(format!("Num instances: {:?}", self.scene.instances.size() as usize / std::mem::size_of::<lp::Instance>()));
                    });
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);

                ui.heading("Tonemapping:");
                ui.add_space(4.0);
                {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut self.exposure).speed(0.05));
                        ui.label("Exposure");
                    });

                    ui.checkbox(&mut self.filmic, "Filmic");
                    ui.checkbox(&mut self.srgb, "SRGB");
                }

                ui.add_space(12.0);
                ui.separator();
                ui.add_space(12.0);

                ui.heading("Result image:");
                ui.add_space(4.0);
                {
                    let mut output_res_x = self.output.back().size().width;
                    let mut output_res_y = self.output.back().size().height;
                    let old_output_res_x = output_res_x;
                    let old_output_res_y = output_res_y;

                    ui.checkbox(&mut self.keep_aspect_ratio, "Keep aspect ratio");
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut output_res_x).range(1..=10000).speed(1));
                        ui.label("Image size width");
                    });
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut output_res_y).range(1..=10000).speed(1));
                        ui.label("Image size height");
                    });

                    if self.keep_aspect_ratio
                    {
                        if output_res_y != old_output_res_y {
                            output_res_x = u32::max(1, (output_res_y as f32 * self.camera_params.aspect) as u32);
                        } else {
                            output_res_y = u32::max(1, (output_res_x as f32 / self.camera_params.aspect) as u32);
                        }
                    }

                    if output_res_x != old_output_res_x || output_res_y != old_output_res_y
                    {
                        self.reset_accumulation();
                        self.resize_output_textures(output_res_x, output_res_y);
                    }

                    if ui.button("Save HDR").clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("Save HDR")
                            .add_filter("HDR Image", &["hdr", "exr"])
                            .save_file()
                        {
                            let res = if self.denoising {
                                lpl::save_texture(self.device, self.queue,
                                                  &path,
                                                  &self.denoised)
                            } else {
                                lpl::save_texture(self.device, self.queue,
                                                  &path,
                                                  self.output.front())
                            };

                            if let Err(err) = res {
                                println!("Failed to save texture: {}", err);
                            }
                        }
                    }

                    if ui.button("Save tonemapped").clicked()
                    {
                        if let Some(path) = rfd::FileDialog::new()
                            .set_title("Save tonemapped")
                            .add_filter("LDR Image", &["png", "jpg", "jpeg"])
                            .save_file()
                        {
                            let width  = self.output.back().size().width;
                            let height = self.output.back().size().height;

                            // Create texture
                            let tmp_tex = self.device.create_texture(&wgpu::TextureDescriptor {
                                label: None,
                                size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
                                mip_level_count: 1,
                                sample_count: 1,
                                dimension: wgpu::TextureDimension::D2,
                                format: wgpu::TextureFormat::Rgba8Unorm,
                                usage: wgpu::TextureUsages::STORAGE_BINDING |
                                       wgpu::TextureUsages::TEXTURE_BINDING |
                                       wgpu::TextureUsages::RENDER_ATTACHMENT |
                                       wgpu::TextureUsages::COPY_SRC,
                                view_formats: &[]
                            });

                            let src = if self.denoising { &self.denoised } else { self.output.front() };
                            lp::tonemap_and_fit_aspect(&self.device, &self.queue, &self.tonemap_resources, src, &tmp_tex, &lp::TonemapDesc {
                                viewport: None,
                                exposure: self.exposure,
                                filmic: self.filmic,
                                srgb: self.srgb,
                            });

                            let res = lpl::save_texture(&self.device, &self.queue,
                                                        &path,
                                                        &tmp_tex);
                            if let Err(err) = res {
                                // TODO: popup
                                println!("Failed to save texture: {}", err);
                            }
                        }
                    }
                }

                ui.add_space(12.0);
            });
        });

        let pixels_per_point = egui_ctx.pixels_per_point();
        self.ui_panel_width = (panel.response.rect.width() * pixels_per_point) as u32;
    }

    fn ui_render_type(&mut self, ui: &mut egui::Ui) -> bool
    {
        use lp::FalsecolorType::*;
        use lp::DebugVizType::*;

        let old_render_type = self.render_type;
        egui::ComboBox::from_label("Visualization")
            .selected_text(format!("{:?}", self.render_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.render_type, RenderType::Pathtrace, "Pathtrace");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Albedo), "Albedo");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Normals), "Normals");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(NormalsUnsigned), "Normals (Unsigned)");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(FrontFacing), "Front Facing");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Emission), "Emission");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Roughness), "Roughness");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Metallic), "Metallic");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Opacity), "Opacity");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(MatType), "Material Type");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(IsDelta), "Is Delta");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Instance), "Instance");
                ui.selectable_value(&mut self.render_type, RenderType::Falsecolor(Tri), "Triangle");
                ui.selectable_value(&mut self.render_type, RenderType::Debug(BVHTriChecks), "Triangle checks (Debug)");
                ui.selectable_value(&mut self.render_type, RenderType::Debug(BVHAABBChecks), "Box checks (Debug)");
                ui.selectable_value(&mut self.render_type, RenderType::Debug(NumBounces), "Number of bounces (Debug)");
            });
        return self.render_type != old_render_type;
    }

    fn ui_pathtrace_type(&mut self, ui: &mut egui::Ui) -> bool
    {
        let old_pathtrace_type = self.pathtrace_type;
        egui::ComboBox::from_label("Pathtrace Algorithm")
            .selected_text(format!("{:?}", self.pathtrace_type))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.pathtrace_type, lp::PathtraceType::Standard, "Standard");
                ui.selectable_value(&mut self.pathtrace_type, lp::PathtraceType::MIS, "MIS");
                ui.selectable_value(&mut self.pathtrace_type, lp::PathtraceType::Naive, "Naive");
                ui.selectable_value(&mut self.pathtrace_type, lp::PathtraceType::Direct, "Direct");
            });
        return self.pathtrace_type != old_pathtrace_type;
    }

    fn ui_heatmap_params(&mut self, ui: &mut egui::Ui)
    {
        ui_min_max(ui, "Heatmap:", &mut self.debug_viz.heatmap_min, &mut self.debug_viz.heatmap_max, 0.0..=1000.0);
    }

    fn reset_accumulation(&mut self)
    {
        self.accum_counter = 0;
        self.gbuffers_accum_counter = 0;
        self.tile_params.tile_idx = 0;
    }

    fn switch_to_freeroam(&mut self)
    {
        if self.selected_cam == -1 { return; }
        self.selected_cam = -1;

        if is_transform_flipped(self.cam_transform.to_mat4()) {
            println!("flipped");
        }

        (self.cam_pos, self.cam_rot, _) = lp::matrix_to_xform(self.cam_transform.to_mat4());
        self.cam_angles = get_euler_angles(self.cam_rot);
        //self.cam_angles.x += std::f32::consts::PI;
        self.cam_angles.y *= -1.0;
    }

    fn switch_to_cam(&mut self, cam_idx: i32)
    {
        if cam_idx < 0 { return; }
        self.selected_cam = cam_idx;
        self.cam_transform = self.scene_cameras[cam_idx as usize].transform;
        self.camera_params = self.scene_cameras[cam_idx as usize].params;
    }
}

pub fn is_transform_flipped(mat: lp::Mat4) -> bool
{
    let x_axis = lp::Vec3 { x: mat.m[0][0], y: mat.m[0][1], z: mat.m[0][2] };
    let y_axis = lp::Vec3 { x: mat.m[1][0], y: mat.m[1][1], z: mat.m[1][2] };
    let z_axis = lp::Vec3 { x: mat.m[2][0], y: mat.m[2][1], z: mat.m[2][2] };

    // Determinant of the 3x3 matrix
    let det = lp::dot_vec3(x_axis, lp::cross_vec3(y_axis, z_axis));
    return det < 0.0;
}

fn get_euler_angles(quat: lp::Quat) -> lp::Vec2
{
    use std::f32::consts::FRAC_PI_2;

    // Yaw (rotation around Y axis)
    let siny_cosp = 2.0 * (quat.w * quat.y + quat.x * quat.z);
    let cosy_cosp = 1.0 - 2.0 * (quat.y * quat.y + quat.x * quat.x);
    let yaw = siny_cosp.atan2(cosy_cosp);

    // Pitch (rotation around X axis)
    let sinp = 2.0 * (quat.w * quat.x - quat.z * quat.y);
    let pitch = if sinp.abs() >= 1.0 {
        sinp.signum() * FRAC_PI_2 // clamp to ±90 degrees
    } else {
        sinp.asin()
    };

    return lp::Vec2 { x: yaw, y: pitch }
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
    return (0.0, 400.0);
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

pub struct DoubleBufferedTexture
{
    pub textures: [wgpu::Texture; 2],
    pub front_idx: usize,
    pub back_idx: usize
}

impl<'a> DoubleBufferedTexture
{
    pub fn create(device: &wgpu::Device, desc: &wgpu::TextureDescriptor) -> DoubleBufferedTexture
    {
        return Self {
            textures: [
                device.create_texture(desc),
                device.create_texture(desc),
            ],
            front_idx: 0,
            back_idx: 1,
        }
    }

    pub fn front(&'a self) -> &'a wgpu::Texture
    {
        return &self.textures[self.front_idx];
    }

    pub fn back(&'a self) -> &'a wgpu::Texture
    {
        return &self.textures[self.back_idx];
    }

    pub fn copy_front_to_back(&self, device: &wgpu::Device, queue: &wgpu::Queue)
    {
        assert!(self.textures[0].format() == self.textures[1].format());

        let format = self.textures[0].format();
        let blitter = wgpu::util::TextureBlitter::new(device, format);
        let mut encoder = device.create_command_encoder(&Default::default());
        let src = self.textures[self.front_idx].create_view(&Default::default());
        let dst = self.textures[self.back_idx].create_view(&Default::default());
        blitter.copy(device, &mut encoder, &src, &dst);
        queue.submit(Some(encoder.finish()));
    }

    pub fn flip(&mut self)
    {
        let tmp = self.front_idx;
        self.front_idx = self.back_idx;
        self.back_idx = tmp;
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32)
    {
        resize_texture(device, &mut self.textures[0], width, height);
        resize_texture(device, &mut self.textures[1], width, height);
    }

    // TODO
    pub fn fill_black(&mut self, device: &wgpu::Device)
    {
        
    }
}
