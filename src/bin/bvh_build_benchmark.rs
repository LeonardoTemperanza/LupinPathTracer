
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
    println!("These times only include the BVH build, not the time spent loading from disk and parsing.");
    println!("The BVH currently has maximum depth equal to {}.", BVH_MAX_DEPTH);
    const NUM_TESTS: i32 = 20;
    println!("Performing {} measurements...", NUM_TESTS);

    let mut obj_path = std::env::current_exe().unwrap();
    obj_path.pop();
    obj_path = append_to_path(obj_path, "/../assets/dragon.obj");
    let path: String = obj_path.into_os_string().to_str().unwrap().to_string();

    let mut measurements: [f32; NUM_TESTS as usize] = Default::default();
    for i in 0..NUM_TESTS
    {
        let (_, times) = load_scene_obj(&path, &mut renderer);
        println!("Time {}: {}s", i + 1, times.bvh_build / 1000.0);

        measurements[i as usize] = times.bvh_build / 1000.0;
    }

    // Calculate average
    let mut avg_time = 0.0;
    for i in 0..NUM_TESTS
    {
        avg_time += measurements[i as usize];
    }
    avg_time /= NUM_TESTS as f32;

    // Calculate standard deviation
    let mut std_deviation = 0.0;
    for i in 0..NUM_TESTS
    {
        std_deviation += square_f32(measurements[i as usize] - avg_time);
    }
    std_deviation = (std_deviation / NUM_TESTS as f32).sqrt();

    println!("");
    println!("Average: {}", avg_time);
    println!("Standard Deviation: {}", std_deviation);
}
