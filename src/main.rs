
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
//use egui_winit::{Platform, PlatformDescriptor};

mod base;
pub use base::*;

// Choose renderer between different backends
mod renderer_wgpu;
pub use renderer_wgpu::{Renderer};

mod core;

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
    let mut renderer = Renderer::new(&window);

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
            egui_state.on_window_event(&window, &event);
            
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    // NOTE: There are some major and ugly artifacts when resizing on windows.
                    // it seems to be a wgpu problem, not a winit problem?
                    // https://github.com/gfx-rs/wgpu/issues/1168
                    renderer.resize(new_size.width as i32, new_size.height as i32);
                    window.request_redraw();
                },
                WindowEvent::CloseRequested =>
                {
                    target.exit();
                },
                WindowEvent::RedrawRequested =>
                {
                    use egui::ClippedPrimitive;

                    // Consume the accumulated inputs
                    let egui_input = egui_state.take_egui_input(&window);

                    delta_time = min_delta_time.max(time_begin.elapsed().as_secs_f32());

                    let win_size   = window.inner_size();
                    let win_width  = win_size.width as i32;
                    let win_height = win_size.height as i32;
                    let scale      = window.scale_factor() as f32;

                    let gui_output = core.main_update(&mut renderer, &mut egui_ctx, egui_input);

                    egui_state.handle_platform_output(&window, gui_output.platform_output);
                    let tris: Vec<ClippedPrimitive> = egui_ctx.tessellate(gui_output.shapes,
                                                                          gui_output.pixels_per_point);

                    renderer.prepare_frame();
                    renderer.draw_scene();

                    // Draw gui last, as an overlay
                    renderer.draw_egui(&gui_output.textures_delta, tris, win_width, win_height, scale);

                    // Notify winit that we're about to submit a new frame.
                    // Not sure if this actually does anything...
                    window.pre_present_notify();
                    renderer.swap_buffers();

                    // Continuously request drawing messages to let the main loop continue
                    window.request_redraw();
                }
                _ => {},
            }
        }
    }).unwrap();
}
