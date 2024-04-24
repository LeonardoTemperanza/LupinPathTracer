
extern crate glfw;
use glfw::{Context};

mod base;
pub use base::*;

// TODO: Choose renderer based on platform
mod renderer_wgpu;
use renderer_wgpu::Renderer;
use renderer_wgpu as renderer;

fn main()
{
    use glfw::fail_on_errors;
    let mut glfw = glfw::init(fail_on_errors!()).unwrap();
    let (mut window, events) = glfw.create_window(2000, 1000, "GPU Raytracer",
                                                  glfw::WindowMode::Windowed).expect("Failed to create a window.");

    window.make_current();
    window.set_key_polling(true);

    let r = renderer::init(&window);

    let mut first_frame: bool = false;
    let min_delta_time: f32   = 1.0/20.0;
    let mut delta_time: f32   = 0.0;
    let mut time_begin: f32   = 0.0;
    let mut time_end: f32     = 0.0;
    while !window.should_close()
    {
        glfw.poll_events();
        for(_, _event) in glfw::flush_messages(&events)
        {
            // Event processing here
        }

        if !first_frame
        {
            time_end = 0.0;
            delta_time = min_delta_time.max(time_end - time_begin);
            time_begin = 0.0;
            //core::main_update(delta_time);
            window.swap_buffers();
        }

        renderer::draw_scene();
    }
}
