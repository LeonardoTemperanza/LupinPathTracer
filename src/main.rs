
// Don't spawn a terminal window on windows
#![windows_subsystem = "windows"]

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

mod base;
pub use base::*;

// Choose renderer between different backends
mod renderer_wgpu;
pub use renderer_wgpu::{Renderer};

mod core;

mod loader;
pub use loader::*;

fn main()
{
    // Initialize window
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_title("Lupin Raytracer")
                                     .with_inner_size(LogicalSize::new(1080.0, 720.0))
                                     .with_visible(false)
                                     .build(&event_loop)
                                     .unwrap();

    let initial_win_size = window.inner_size();

    // Initialize renderer
    let mut renderer = Renderer::new(&window, initial_win_size.width as i32, initial_win_size.height as i32);
    renderer.log_backend();

    // Initialize egui
    let mut egui_ctx = egui::Context::default();
    let viewport_id = egui_ctx.viewport_id();

    let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None);

    // Init Core state
    let mut core = core::Core::new(&mut renderer);

    window.set_visible(true);

    let min_delta_time: f32 = 1.0/20.0;  // Reasonable min value to prevent degeneracies when updating state
    let mut delta_time: f32 = 0.0;
    let time_begin = Instant::now();
    event_loop.run(|event, target|
    {
        if let Event::WindowEvent { window_id, event } = event
        {
            // Collect inputs
            let _ = egui_state.on_window_event(&window, &event);
            
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    // NOTE: There are some major and ugly artifacts when resizing on windows.
                    // it seems to be a wgpu problem, not a winit problem?
                    // https://github.com/gfx-rs/wgpu/issues/1168
                    // This problem goes away using the opengl backend
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

                    core.main_update(&mut renderer, &window, &mut egui_ctx, &mut egui_state);

                    // Continuously request drawing messages to let the main loop continue
                    window.request_redraw();
                }
                _ => {},
            }
        }
    }).unwrap();
}
