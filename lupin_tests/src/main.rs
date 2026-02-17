
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

use std::sync::Arc;

pub use lupin as lp;
pub use lupin_loader as lpl;
use lupin::wgpu as wgpu;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use clap::Parser;

use std::path::PathBuf;
use std::io;
use std::fs::{self, DirEntry};
use std::path::Path;

use std::io::Write;

pub static COMPARE_SHADER_SRC: &str = include_str!("shaders/compare_textures.wgsl");

const SAMPLES_PER_PIXEL: u32 = 10;
const NUM_SAMPLES: u32 = 1000;
const NUM_BOUNCES: u32 = 8;
// NOTE: Low max radiance so we get a less noisy (but not necessarily realistic) result.
const MAX_RADIANCE: f32 = 10.0;

const COMPARE_EPSILON: f32 = 5.0;  // TODO: Fix.
// Tied to shader
const COMPARE_WORKGROUP_SIZE: u32 = 4;

pub struct App
{
    window: Option<Arc<Window>>,
    ctx: Option<GPUContext>,

    scenes: Vec<PathBuf>,
    cmd_line_args: CmdLineArgs,

    // Render progress
    scene_idx: usize,
    camera_idx: usize,
    changed_cam: bool,
    accum_count: u32,
    error: bool,

    scene: Option<lp::Scene>,
    cameras: Vec<lpl::SceneCamera>,
    expected_output: Option<wgpu::Texture>,
    expected_output_view: Option<wgpu::TextureView>,
    compare_bindgroup: Option<wgpu::BindGroup>,
}

#[derive(Parser, Debug)]
struct CmdLineArgs
{
    #[arg(long, default_value_t = false)]
    first_camera_only: bool,
    #[arg(long, default_value_t = false)]
    overwrite_renders: bool,
}

pub struct GPUContext
{
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,

    // Lupin resources
    pathtrace_res: lp::PathtraceResources,
    tonemap_res: lp::TonemapResources,
    output: lp::DoubleBufferedTexture,

    // Compare shader
    compare_pipeline: wgpu::ComputePipeline,
    compare_result: wgpu::Buffer,
    readback: wgpu::Buffer,
}

impl App
{
    pub fn update_and_render(&mut self, swapchain: &wgpu::Texture)
    {
        let scene_path = &self.scenes[self.scene_idx];
        let ctx = self.ctx.as_mut().unwrap();
        let des_path = build_render_output_path(scene_path, self.camera_idx);
        let err_path = build_render_error_path(scene_path, self.camera_idx);

        if matches!(self.scene, None)
        {
            let path_json = std::path::Path::new(&scene_path);
            let (lp_scene, cameras) = lpl::load_scene_yoctogl_v24(path_json, &ctx.device, &ctx.queue, false).unwrap();
            self.scene = Some(lp_scene);
            self.cameras = cameras;
        }

        if self.changed_cam
        {
            self.expected_output = None;
            self.expected_output_view = None;
            self.compare_bindgroup = None;

            let (width, height) = compute_dimensions_for_1080p(self.cameras[self.camera_idx].params.aspect);
            ctx.output.resize(&ctx.device, width, height);
            self.changed_cam = false;

            let texture_usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
            let loaded = lpl::load_texture_with_usage(&ctx.device, &ctx.queue, des_path.to_str().unwrap(), true, texture_usage);
            if let Ok(loaded) = loaded
            {
                self.expected_output = Some(loaded);
                self.expected_output_view = Some(self.expected_output.as_ref().unwrap().create_view(&Default::default()));
            }
        }

        let lp_scene = self.scene.as_ref().unwrap();
        lp::pathtrace_scene(&ctx.device, &ctx.queue, &ctx.pathtrace_res, lp_scene, ctx.output.front(), Default::default(), &lp::PathtraceDesc {
            accum_params: Some(lp::AccumulationParams {
                prev_frame: ctx.output.back(),
                accum_counter: self.accum_count,
            }),
            tile_params: None,
            camera_params: self.cameras[self.camera_idx].params,
            camera_transform: self.cameras[self.camera_idx].transform,
            force_software_bvh: false,
            advanced: lp::AdvancedParams {
                max_radiance: MAX_RADIANCE,
                ..Default::default()
            },
        });

        print!("\r'{}': Camera {}: {}/{} ", scene_path.display(), self.camera_idx, self.accum_count * SAMPLES_PER_PIXEL, NUM_SAMPLES);
        std::io::stdout().flush().unwrap();

        lp::tonemap_and_fit_aspect(&ctx.device, &ctx.queue, &ctx.tonemap_res, ctx.output.front(), &swapchain, &lp::TonemapDesc{
            viewport: Some(lp::Viewport { x: 0.0, y: 0.0, w: swapchain.width() as f32, h: swapchain.height() as f32 / 2.0 }),
            exposure: 0.0,
            filmic: true,
            srgb: true,
            clear: true,
        });
        if let Some(expected) = self.expected_output.as_ref()
        {
            lp::tonemap_and_fit_aspect(&ctx.device, &ctx.queue, &ctx.tonemap_res, &expected, &swapchain, &lp::TonemapDesc{
                viewport: Some(lp::Viewport { x: 0.0, y: swapchain.height() as f32 / 2.0, w: swapchain.width() as f32, h: swapchain.height() as f32 / 2.0 }),
                exposure: 0.0,
                filmic: true,
                srgb: true,
                clear: false,
            });
        }

        self.accum_count += 1;

        if self.accum_count * SAMPLES_PER_PIXEL > NUM_SAMPLES
        {
            if self.cmd_line_args.overwrite_renders
            {
                let res = lpl::save_texture(&ctx.device, &ctx.queue, &des_path, ctx.output.front());
                match res
                {
                    Err(res) => { println!("Failed to save texture: {}", res); }
                    Ok(res)  => { println!("Ok."); }
                }
            }
            else
            {
                if self.expected_output.is_none()
                {
                    print!("Could not find '{}'! Saving current render to that path... ", des_path.display());
                    let res = lpl::save_texture(&ctx.device, &ctx.queue, &des_path, ctx.output.front());
                    match res
                    {
                        Err(res) => { print!("Failed to save texture."); }
                        Ok(res)  => { }
                    }
                    println!("");
                }
                else
                {
                    self.compare_bindgroup = Some(create_compare_bindgroup(&ctx.device, &ctx.compare_result, ctx.output.front_view(), self.expected_output_view.as_ref().unwrap()));

                    ctx.queue.write_buffer(&ctx.compare_result, 0, lp::to_u8_slice(&[0]));

                    // Compare textures with a compute shader
                    let mut encoder = ctx.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: None,
                            timestamp_writes: None
                        });

                        pass.set_pipeline(&ctx.compare_pipeline);
                        pass.set_bind_group(0, &self.compare_bindgroup, &[]);
                        pass.set_immediates(0, lp::to_u8_slice(&[COMPARE_EPSILON]));

                        let num_workers_x = (ctx.output.front().width() + COMPARE_WORKGROUP_SIZE - 1) / COMPARE_WORKGROUP_SIZE;
                        let num_workers_y = (ctx.output.front().height() + COMPARE_WORKGROUP_SIZE - 1) / COMPARE_WORKGROUP_SIZE;
                        pass.dispatch_workgroups(num_workers_x, num_workers_y, 1);
                    }

                    encoder.copy_buffer_to_buffer(&ctx.compare_result, 0, &ctx.readback, 0, 4);
                    ctx.queue.submit(Some(encoder.finish()));

                    let buffer_slice = ctx.readback.slice(..);
                    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

                    // Wait until we receive results
                    ctx.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

                    let is_wrong;
                    {
                        let data = buffer_slice.get_mapped_range();
                        is_wrong = lp::from_u8_slice::<u32>(&data)[0] != 0;
                    }

                    if !is_wrong
                    {
                        println!("Ok.");
                    }
                    else
                    {
                        self.error = true;

                        print!("Error! Saving incorrect render to '{}'... ", err_path.display());
                        let res = lpl::save_texture(&ctx.device, &ctx.queue, &err_path, ctx.output.front());
                        match res
                        {
                            Err(res) => { print!("Failed to save texture."); }
                            Ok(res)  => { }
                        }
                        println!("");
                    }

                    ctx.readback.unmap();
                }
            }

            self.camera_idx = (self.camera_idx + 1) % self.cameras.len();
            if self.cmd_line_args.first_camera_only { self.camera_idx = 0; }
            if self.camera_idx == 0
            {
                self.scene_idx += 1;
                self.scene = None;
            }

            self.changed_cam = true;
            self.accum_count = 0;
        }

        ctx.output.flip();
    }
}

impl ApplicationHandler for App
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop)
    {
        let is_first_resumed = matches!(self.window, None);
        if is_first_resumed
        {
            let window_attributes = Window::default_attributes()
                .with_title("Lupin Tests")
                .with_visible(false)
                .with_inner_size(winit::dpi::LogicalSize::new(1000.0, 1000.0));
            self.window = Some(Arc::new(event_loop.create_window(window_attributes).unwrap()));


            let window = self.window.as_ref().unwrap().clone();
            let width = 1000;
            let height = 1000;

            let surface_config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Rgba8Unorm,
                width: width as u32,
                height: height as u32,
                present_mode: wgpu::PresentMode::AutoNoVsync,  // Disabling vsync makes it so tiled rendering isn't slowed down.
                desired_maximum_frame_latency: 0,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            };
            let (device, queue, surface, adapter) = lp::init_default_wgpu_context(&surface_config, window);

            let tonemap_res = lp::build_tonemap_resources(&device);
            let pathtrace_res = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
                with_runtime_checks: false,
                max_bounces: NUM_BOUNCES,
                samples_per_pixel: SAMPLES_PER_PIXEL,
            });

            let output = lp::DoubleBufferedTexture::create(&device, &wgpu::TextureDescriptor {
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

            // Create shader and pipeline for comparing textures.
            let compare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(COMPARE_SHADER_SRC.into()),
            });

            let compare_bindgroup_layout = create_compare_bindgroup_layout(&device);
            let compare_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[
                    &compare_bindgroup_layout,
                ],
                immediate_size: 4,
            });
            let compare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&compare_pipeline_layout),
                module: &compare_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let compare_result = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            queue.write_buffer(&compare_result, 0, lp::to_u8_slice(&[0]));

            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            self.ctx = Some(GPUContext {
                device, queue, surface,
                tonemap_res,
                pathtrace_res,
                output,
                surface_config,

                compare_pipeline,
                compare_result,
                readback,
            });

            // Get all scene paths
            for entry in fs::read_dir(Path::new(SCENES_DIR)).unwrap()
            {
                let entry = entry.unwrap();
                let path = entry.path();

                if path.is_dir()
                {
                    if let Some(dir_name) = path.file_name().and_then(|n| n.to_str())
                    {
                        let json_path = path.join(format!("{dir_name}.json"));
                        if json_path.is_file() {
                            self.scenes.push(json_path);
                        }
                    }
                }
            }

            let ctx = self.ctx.as_ref().unwrap();
            let frame = ctx.surface.get_current_texture().unwrap();
            self.update_and_render(&frame.texture);
            self.window.as_mut().unwrap().set_visible(true);
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent)
    {
        match event
        {
            WindowEvent::Resized(new_size) =>
            {
                let ctx = self.ctx.as_mut().unwrap();
                ctx.surface_config.width = new_size.width;
                ctx.surface_config.height = new_size.height;
                ctx.surface_config.width = new_size.width;
                ctx.surface.configure(&ctx.device, &ctx.surface_config);

                if let Some(window) = self.window.as_ref() {
                    window.request_redraw();
                }
            }
            WindowEvent::CloseRequested =>
            {
                event_loop.exit();
            },
            WindowEvent::RedrawRequested =>
            {
                let ctx = self.ctx.as_ref().unwrap();
                let frame = ctx.surface.get_current_texture().unwrap();
                self.update_and_render(&frame.texture);
                frame.present();

                if self.scene_idx >= self.scenes.len()
                {
                    event_loop.exit();

                    println!("");
                    println!("------------------------------");
                    if self.error {
                        println!("Failure: Tests completed with some errors.");
                    } else {
                        println!("Success.");
                    }
                }
                else
                {
                    // Continuously request drawing messages to let the main loop continue
                    self.window.as_ref().unwrap().request_redraw();
                }
            }
            _ => (),
        }
    }
}

impl Default for App
{
    fn default() -> Self
    {
        return Self {
            cmd_line_args: CmdLineArgs::parse(),
            window: None,
            ctx: None,
            scene_idx: 0,
            camera_idx: 0,
            changed_cam: true,
            accum_count: 0,
            error: false,

            scene: None,
            scenes: vec![],
            cameras: vec![],
            expected_output: None,
            expected_output_view: None,
            compare_bindgroup: None,
        }
    }
}

const SCENES_DIR: &str = "test_scenes";

fn main()
{
    if std::fs::exists(SCENES_DIR).is_err() {
        panic!("It appears that the \"{}\" directory is missing (it's not directly visible from the current working directory). The \"{}\" directory is located in the project's base directory.", SCENES_DIR, SCENES_DIR);
    }

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}

fn compute_dimensions_for_1080p(aspect: f32) -> (u32, u32)
{
    if aspect < 1.0 {  // Taller than wide
        return ((1920.0 * aspect) as u32, 1920);
    } else {  // Wider than tall
        return (1920, (1920.0 / aspect) as u32);
    }
}

fn build_render_output_path(scene_path: &PathBuf, cam_idx: usize) -> PathBuf
{
    let mut path = scene_path.clone();
    path.pop();
    path.push(format!("render_cam{}.hdr", cam_idx));
    return path;
}

fn build_render_error_path(scene_path: &PathBuf, cam_idx: usize) -> PathBuf
{
    let mut path = scene_path.clone();
    path.pop();
    path.push(format!("error_cam{}.hdr", cam_idx));
    return path;
}

fn create_compare_bindgroup_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout
{
    return device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None,
            },
        ]
    });
}

fn create_compare_bindgroup(device: &wgpu::Device, compare_result: &wgpu::Buffer, output: &wgpu::TextureView, expected_output: &wgpu::TextureView) -> wgpu::BindGroup
{
    return device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &create_compare_bindgroup_layout(device),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: compare_result, offset: 0, size: None }) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&output) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&expected_output) },
        ]
    });
}
