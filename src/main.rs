
use std::time::Instant;

use winit::
{
    dpi::LogicalSize,
    event::{Event, WindowEvent, StartCause},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

use ::egui::FontDefinitions;
use egui_winit_platform::{Platform, PlatformDescriptor};

mod base;
pub use base::*;

// Choose renderer between different backends
mod renderer_wgpu;
pub use renderer_wgpu::{Renderer, EGUIRenderState};

mod core;

fn main()
{
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Lupin Raytracer")
                                     .with_inner_size(LogicalSize::new(1080.0, 720.0))
                                     .with_visible(false)
                                     .build(&event_loop)
                                     .unwrap();

    let initial_win_size = window.inner_size();

    let mut renderer = Renderer::new(&window);

    // Init EGUI and font used
    let platform_desc = PlatformDescriptor
    {
        physical_width: initial_win_size.width as u32,
        physical_height: initial_win_size.height as u32,
        scale_factor: window.scale_factor(),
        font_definitions: FontDefinitions::default(),
        style: Default::default(),
    };
    let mut platform = Platform::new(platform_desc);

    // NOTE: Important for crisp font and icons on high DPI displays
    platform.context().set_pixels_per_point(window.scale_factor() as f32);

    // Init EGUI Rendering state
    let mut egui_state: EGUIRenderState = renderer.init_egui();

    // Init Core state
    let mut core = core::Core::new(&mut renderer);

    window.set_visible(true);

    let min_delta_time: f32 = 1.0/20.0;  // Reasonable min value to prevent degeneracies when updating state
    let mut delta_time: f32 = 0.0;
    let time_begin = Instant::now();
    event_loop.run(|event, target|
    {
        platform.handle_event(&event);

        if let Event::WindowEvent { window_id, event } = event
        {
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    // TODO: There are some major artifacts when resizing on windows.
                    // Not sure how if this can be fixed within the confines of winit API
                    renderer.resize(new_size.width as i32, new_size.height as i32);
                    window.request_redraw();
                },
                WindowEvent::CloseRequested =>
                {
                    target.exit();
                },
                WindowEvent::RedrawRequested =>
                {
                    delta_time = min_delta_time.max(time_begin.elapsed().as_secs_f32());
                    platform.update_time(delta_time as f64);

                    let win_size   = window.inner_size();
                    let win_width  = win_size.width as i32;
                    let win_height = win_size.height as i32;
                    let scale      = window.scale_factor() as f32;

                    let gui_output = core.main_update(&mut renderer, &mut platform, &window);

                    let paint_jobs = platform.context().tessellate(gui_output.shapes, gui_output.pixels_per_point);
                    let textures_delta = gui_output.textures_delta;

                    renderer.prepare_frame();
                    renderer.draw_scene();

                    // Draw gui last, as an overlay
                    renderer.draw_egui(&mut egui_state, &textures_delta, paint_jobs, win_width, win_height, scale);

                    // Notify winit that we're about to submit a new frame.
                    // Not sure if this actually does anything
                    window.pre_present_notify();
                    renderer.swap_buffers();

                    // NOTE: Continuously request drawing messages.
                    // Not sure if there's a better way to keep rendering on the screen
                    window.request_redraw();
                }
                _ => {},
            }
        }
    }).unwrap();
}
