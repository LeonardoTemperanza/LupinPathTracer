
use winit::
{
    event_loop::{EventLoop},
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
                                     .build(&event_loop)
                                     .unwrap();

    let mut renderer = Renderer::new(&window, 0, 0);

    println!("BVH Build Profiler");
    const NUM_TESTS: i32 = 10;
    println!("Performing {} measurements...", NUM_TESTS);

    let mut obj_path = std::env::current_exe().unwrap();
    obj_path.pop();
    obj_path = append_to_path(obj_path, "/../assets/dragon.obj");
    let path: String = obj_path.into_os_string().to_str().unwrap().to_string();

    let mut avg_time: f32 = 0.0;
    for i in 0..NUM_TESTS
    {
        let timer_start = std::time::Instant::now();
        load_scene_obj(&path, &mut renderer);
        let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;
        println!("Time {}: {}s", i + 1, time / 1000.0);

        avg_time += time;
    }

    avg_time /= NUM_TESTS as f32;
    println!("Average: {}", avg_time);
}
