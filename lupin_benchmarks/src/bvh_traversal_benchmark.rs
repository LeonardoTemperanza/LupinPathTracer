
use lupin::base::*;

use winit::
{
    dpi::LogicalSize,
    event::{Event, WindowEvent, StartCause},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
    window::WindowBuilder,
};

pub use lupin::base::*;
pub use lupin::renderer::*;

fn main()
{
    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().with_visible(false)
                                     .with_inner_size(LogicalSize::new(1280.0, 720.0))
                                     .with_title("Ray Query Benchmark")
                                     .with_resizable(false)
                                     .build(&event_loop)
                                     .unwrap();

    let initial_win_size = window.inner_size();

    let mut renderer = Renderer::new(&window, initial_win_size.width as i32, initial_win_size.height as i32);

    const RES_X: u32 = 1920;
    const RES_Y: u32 = 1080;
    const NUM_TESTS: u32 = 400;
    const NUM_REPEATS: u32 = 3;

    println!("BVH Traversal Benchmark");
    println!("The BVH currently has maximum depth equal to {}.", BVH_MAX_DEPTH);
    println!("The model is being rendered at a resolution of {}x{}, with vsync turned off.", RES_X, RES_Y);
    println!("Each frame will be repeated {} times.", NUM_REPEATS);
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

    let texture = renderer.create_texture(RES_X, RES_Y);
    let mut gpu_timer = renderer.create_gpu_timer(NUM_TESTS + 1);
    let mut times: [f32; NUM_TESTS as usize] = [0.0; NUM_TESTS as usize];

    let mut i = 0;

    event_loop.run(|event, target|
    {
        if let Event::WindowEvent { window_id, event } = event
        {
            match event
            {
                WindowEvent::Resized(new_size) =>
                {
                    renderer.resize(new_size.width as i32, new_size.height as i32);
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

                            renderer.add_timestamp(&mut gpu_timer);
                            renderer.begin_frame();
                            for i in 0..NUM_REPEATS
                            {
                                renderer.draw_scene(&scene, &texture, transform_to_matrix(cam_transform));
                            }
                            renderer.show_texture(&texture);
                            renderer.end_frame();

                            i += 1;
                    if i >= NUM_TESTS  // Terminate the application
                    {
                        renderer.add_timestamp(&mut gpu_timer);
                        renderer.get_gpu_times(&mut gpu_timer, &mut times);
                        

                        save_plot(&times, NUM_REPEATS);
                        target.exit();
                    }
                    else  // Continue the loop
                    {
                        window.request_redraw();
                    }
                    },
                    _ => {},
                }
            }
            }).unwrap();
}

use plotters::prelude::*;
pub fn save_plot(times: &[f32], num_repeats: u32)
{
    // Prepare output file name
    let output_path = generate_unique_filename("src/bin/benchmark_result", "png");

    let root_area = BitMapBackend::new(&output_path, (1920, 1080)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let root_area = root_area.titled("Benchmark Results", ("sans-serif", 60)).unwrap();

    let (upper, lower) = root_area.split_vertically(512);

    let angles = (0.0f32..360.0).step(360.0 / times.len() as f32);

    let mut cc = ChartBuilder::on(&upper)
    .margin(5)
    .set_all_label_area_size(50)
    .build_cartesian_2d(0.0f32..360.0, 0.0f32..24.0f32).unwrap();
    
    cc.configure_mesh()
    .x_labels(25)
    .y_labels(20)
    .disable_mesh()
    .x_label_formatter(&|v| format!("{:.1}", v))
    .y_label_formatter(&|v| format!("{:.1}", v))
    .x_desc("Angle (degrees)")
    .y_desc("Time (ms)")
    .draw().unwrap();

    let angles: Vec<f32> = (0..times.len())
        .map(|i| i as f32 * 360.0 / times.len() as f32)
        .collect();

    let time_per_repeat: Vec<f32> = times.iter().map(|&t| t / num_repeats as f32).collect();
    let data: Vec<(f32, f32)> = angles.iter().zip(time_per_repeat.iter()).map(|(&a, &t)| (a, t)).collect();

    cc.draw_series(LineSeries::new(data, &BLACK));

    cc.configure_series_labels().border_style(BLACK).draw().unwrap();

    let drawing_areas = lower.split_evenly((1, 2));

    root_area.present().expect("Unable to save benchmark result");
    println!("Result has been saved to {}", output_path.display());
}

fn generate_unique_filename(base_name: &str, extension: &str)->std::path::PathBuf
{
    use std::path::PathBuf;

    let mut counter = 0;
    let mut filename = format!("{}.{}", base_name, extension);
    let mut path = PathBuf::from(&filename);

    while path.exists()
    {
        filename = format!("{}_{}.{}", base_name, counter, extension);
        path = PathBuf::from(&filename);
        counter += 1;
    }

    return path;
}

// @copypaste
pub fn load_scene_obj(device: wgpu::Device, queue: wgpu::Queue, path: &str)->Scene
{
    let mut loading_times: LoadingTimes = Default::default();

    let timer_start = std::time::Instant::now();
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;

    loading_times.parsing = time;

    assert!(scene.is_ok());
    let (mut models, materials) = scene.expect("Failed to load OBJ file");

    let mut verts_pos: Vec<f32> = Vec::new();

    if models.len() > 0
    {
        let mesh = &mut models[0].mesh;

        // Construct the buffer to send to GPU. Include an extra float
        // for 16-byte padding (which seems to be required in WebGPU).
        
        verts_pos.reserve_exact(mesh.positions.len() + mesh.positions.len() / 3);
        for i in (0..mesh.positions.len()).step_by(3)
        {
            verts_pos.push(mesh.positions[i + 0]);
            verts_pos.push(mesh.positions[i + 1]);
            verts_pos.push(mesh.positions[i + 2]);
            verts_pos.push(0.0);
        }

        let timer_start = std::time::Instant::now();
        let bvh_buf = construct_bvh(&device, &queue, verts_pos.as_slice(), &mut mesh.indices);
        let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;
        loading_times.bvh_build = time;

        let verts_pos_buf = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts_pos) });
        let indices_buf = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&mesh.indices) });
        let mut verts: Vec<Vertex> = Vec::new();
        verts.reserve_exact(mesh.positions.len() / 3);
        for vert_idx in 0..(mesh.positions.len() / 3)
        {
            let mut normal = Vec3::default();
            if mesh.normals.len() > 0
            {
                normal.x = mesh.normals[vert_idx*3+0];
                normal.y = mesh.normals[vert_idx*3+1];
                normal.z = mesh.normals[vert_idx*3+2];
                normal = normalize_vec3(normal);
            };

            let mut tex_coords = Vec2::default();
            if mesh.texcoords.len() > 0
            {
                tex_coords.x = mesh.texcoords[vert_idx*2+0];
                tex_coords.y = mesh.texcoords[vert_idx*2+1];
                tex_coords = normalize_vec2(tex_coords);
            };

            let vert = Vertex { normal, padding0: 0.0, tex_coords, padding1: 0.0, padding2: 0.0 };

            verts.push(vert);
        }

        let verts_buf = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts) });

        return (Scene
        {
            verts_pos: verts_pos_buf,
            indices:   indices_buf,
            bvh_nodes: bvh_buf,
            verts:     verts_buf
        }, loading_times);
    }

    return Scene
    {
        verts_pos: create_empty_storage_buffer(&device, &queue),
        indices:   create_empty_storage_buffer(&device, &queue),
        bvh_nodes: create_empty_storage_buffer(&device, &queue),
        verts:     create_empty_storage_buffer(&device, &queue)
    };
}
