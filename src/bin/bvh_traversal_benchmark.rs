
use lupin_pathtracer::base::*;

use winit::
{
    dpi::LogicalSize,
    event::{Event, WindowEvent, StartCause},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

pub use lupin_pathtracer::base::*;
pub use lupin_pathtracer::renderer::*;
pub use lupin_pathtracer::loader::*;

fn main()
{
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_visible(false)
                                     .with_inner_size(LogicalSize::new(1080.0, 720.0))
                                     .build(&event_loop)
                                     .unwrap();

    let initial_win_size = window.inner_size();

    let mut renderer = Renderer::new(&window, initial_win_size.width as i32, initial_win_size.height as i32);

    println!("BVH Traversal Benchmark");
    println!("The BVH currently has maximum depth equal to {}.", BVH_MAX_DEPTH);
    const NUM_TESTS: i32 = 400;
    println!("Performing {} measurements...", NUM_TESTS);

    // Load model
    let mut obj_path = std::env::current_exe().unwrap();
    obj_path.pop();
    obj_path = append_to_path(obj_path, "/../assets/dragon.obj");
    let path: String = obj_path.into_os_string().to_str().unwrap().to_string();

    println!("Loading from disk...");
    let (scene, _) = load_scene_obj(&path, &mut renderer);
    println!("Done!");

    window.set_visible(true);

    let texture = renderer.create_texture(1920, 1080);

    let mut i = 0;

    event_loop.run(|event, target|
    {
        if let Event::WindowEvent { window_id, event } = event
        {
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    window.request_redraw();
                },
                WindowEvent::CloseRequested =>
                {
                    target.exit();
                },
                WindowEvent::RedrawRequested =>
                {
                    let angle_diff = 360.0 / NUM_TESTS as f32 * DEG_TO_RAD as f32;
                    let angle_x = angle_diff * i as f32;

                    let mut cam_transform = Transform::default();
                    cam_transform.rot = angle_axis(Vec3::UP, angle_x);
                    cam_transform.pos = rotate_vec3_with_quat(cam_transform.rot, Vec3::BACKWARD);
                    cam_transform.pos.y = 0.35;

                    renderer.begin_frame();
                    renderer.draw_scene(&scene, &texture, transform_to_matrix(cam_transform));
                    renderer.show_texture(&texture);
                    renderer.end_frame();

                    i += 1;
                    if i >= NUM_TESTS { target.exit(); }

                    // Continuously request drawing messages to let the main loop continue
                    window.request_redraw();
                },
                _ => {},
            }
        }
    }).unwrap();

    println!("");
    println!("Average: {}", 0.0);
    println!("Standard Deviation: {}", 0.0);
}
