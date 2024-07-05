
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

mod renderer;
pub use renderer::*;

mod editor;

mod loader;
pub use loader::*;

mod input;
pub use input::*;

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

    let mut renderer = Renderer::new(&window, initial_win_size.width as i32, initial_win_size.height as i32);
    renderer.log_backend();

    let mut egui_ctx = egui::Context::default();
    let viewport_id = egui_ctx.viewport_id();

    let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None);

    let mut core = editor::State::new(&mut renderer);

    window.set_visible(true);

    let mut input_diff = InputDiff::default();

    let min_delta_time: f32 = 1.0/20.0;  // Reasonable min value to prevent degeneracies when updating state
    let mut delta_time: f32 = 1.0/60.0;
    let mut time_begin = Instant::now();
    event_loop.run(|event, target|
    {
        collect_inputs_winit(&mut input_diff, &event);
        
        if let Event::WindowEvent { window_id, event } = event
        {
            // Collect inputs
            let _ = egui_state.on_window_event(&window, &event);
            
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    // NOTE: On vulkan and dx12 there are some artifacts when resizing
                    // This is a wgpu problem, and it goes away completely when using
                    // the opengl backend
                    renderer.resize(new_size.width as i32, new_size.height as i32);
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

                    core.main_update(&mut renderer, &window, &mut egui_ctx, &mut egui_state, delta_time, &mut input_diff);

                    // Continuously request drawing messages to let the main loop continue
                    window.request_redraw();
                },
                _ => {},
            }
        }
    }).unwrap();
}
