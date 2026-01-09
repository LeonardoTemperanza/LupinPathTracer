
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

#[derive(Copy, Clone, Debug)]
pub struct Scene
{
    pub name: &'static str,
    pub samples: u32,
    pub max_radiance: f32,
}

const SAMPLES_PER_PIXEL: u32 = 5;
const NUM_SAMPLES: u32 = 700;
const NUM_BOUNCES: u32 = 8;

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
    scene: Option<lp::Scene>,
    cameras: Vec<lpl::SceneCamera>,
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
}

impl App
{
    pub fn update_and_render(&mut self, swapchain: &wgpu::Texture)
    {
        let ctx = self.ctx.as_mut().unwrap();

        let scene_path = &self.scenes[self.scene_idx];
        if matches!(self.scene, None)
        {
            let path_json = std::path::Path::new(&scene_path);
            let (lp_scene, cameras) = lpl::load_scene_yoctogl_v24(path_json, &ctx.device, &ctx.queue, false).unwrap();
            self.scene = Some(lp_scene);
            self.cameras = cameras;
        }

        if self.changed_cam
        {
            let (width, height) = compute_dimensions_for_1080p(self.cameras[self.camera_idx].params.aspect);
            ctx.output.resize(&ctx.device, width, height);
            self.changed_cam = false;
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
            advanced: Default::default(),
        });

        print!("\r'{}': Camera {}: {}/{} ", scene_path.display(), self.camera_idx, self.accum_count * SAMPLES_PER_PIXEL, NUM_SAMPLES);
        std::io::stdout().flush().unwrap();

        lp::tonemap_and_fit_aspect(&ctx.device, &ctx.queue, &ctx.tonemap_res, ctx.output.front(), &swapchain, &lp::TonemapDesc{
            viewport: None,
            exposure: 0.0,
            filmic: true,
            srgb: true
        });

        self.accum_count += 1;

        if self.accum_count * SAMPLES_PER_PIXEL > NUM_SAMPLES
        {
            if self.cmd_line_args.overwrite_renders
            {
                let mut out_path = scene_path.clone();
                out_path.pop();
                out_path.push(format!("render_cam{}.hdr", self.camera_idx));
                let res = lpl::save_texture(&ctx.device, &ctx.queue, &out_path, ctx.output.front());
                match res
                {
                    Err(res) => { println!("Failed to save texture: {}", res); }
                    Ok(res)  => { println!("Ok."); }
                }
            }

            self.camera_idx = (self.camera_idx + 1) % self.cameras.len();
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
        self.window = Some(Arc::new(event_loop.create_window(Window::default_attributes()).unwrap()));
        if is_first_resumed
        {
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

            self.ctx = Some(GPUContext {
                device, queue, surface,
                tonemap_res,
                pathtrace_res,
                output,
                surface_config,
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

                if self.scene_idx >= self.scenes.len() {
                    event_loop.exit();
                } else {
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
            scene: None,
            scenes: vec![],
            cameras: vec![],
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
