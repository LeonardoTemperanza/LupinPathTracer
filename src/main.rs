
extern crate winit;
use winit::
{
    dpi::LogicalSize,
    event::{Event, WindowEvent, StartCause},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

extern crate egui;
extern crate egui_winit_platform;
use ::egui::FontDefinitions;
use egui_winit_platform::{Platform, PlatformDescriptor};

mod base;
pub use base::*;

// TODO: Choose renderer based on platform
mod renderer_wgpu;
use renderer_wgpu as r;

fn main()
{
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Lupin Raytracer")
                                     .with_inner_size(LogicalSize::new(1080.0, 720.0))
                                     .with_visible(false)
                                     .build(&event_loop)
                                     .unwrap();

    let mut renderer = r::init(&window);

    // We use the egui_winit_platform crate as the platform.
    let platform_desc = PlatformDescriptor
    {
        physical_width: 2000 as u32,
        physical_height: 1000 as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    };
    let mut platform = Platform::new(platform_desc);

    r::init_egui(&renderer);

    let mut demo_app = egui_demo_lib::DemoWindows::default();

    window.set_visible(true);

    let mut first_frame = true;
    let min_delta_time: f32 = 1.0/20.0;
    let mut delta_time: f32 = 0.0;
    let mut time_begin: f32 = Instant::now();
    event_loop.run(|event, target|
    {
        if let Event::WindowEvent { window_id, event } = event
        {
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    r::resize(&mut renderer, new_size.width, new_size.height);
                    window.request_redraw();
                },
                WindowEvent::CloseRequested =>
                {
                    target.exit();
                },
                WindowEvent::RedrawRequested =>
                {
                    if !first_frame
                    {
                        time_end = 0.0;
                        delta_time = min_delta_time.max(time_end - time_begin);
                        time_begin = 0.0;
                        // core::update();
                        // Wait till next frame
                    }
                    else { time_begin = 0.0; }

                    // Render here
                    r::draw_scene(&mut renderer);

                    // Need to continuously request redraws for the program main loop
                    window.request_redraw();
                    first_frame = false;
                }
                _ => {},
            }
        }
    }).unwrap();
}
