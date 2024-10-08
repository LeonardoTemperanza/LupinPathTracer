
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

pub use lupin::base::*;
pub use lupin::renderer::*;
pub use lupin::wgpu_utils::*;

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

    let win_size = window.inner_size();
    let (width, height) = (win_size.width, win_size.height);
    let (device, queue, surface, adapter) = init_wgpu_context(get_device_spec(), &window, width as i32, height as i32);
    log_backend(&adapter);

    window.set_visible(true);
    use std::thread;
    use std::time;
    std::thread::sleep(std::time::Duration::from_millis(1000));

    #[cfg(disable)]
    {
        //let mut renderer = Renderer::new(&window, initial_win_size.width as i32, initial_win_size.height as i32);
        //renderer.log_backend();
        //renderer.set_vsync(true);

        let mut egui_ctx = egui::Context::default();
        let viewport_id = egui_ctx.viewport_id();

        let mut egui_state = egui_winit::State::new(egui_ctx.clone(), viewport_id, &window, None, None);

        let mut core = State::new(&mut renderer);

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
}

// Contains all application logic that isn't already separated
// into different modules (such as serialization and rendering)

use egui::ClippedPrimitive;

#[cfg(disable)]
pub struct State
{
    // egui texture ids
    render_image_id: egui::TextureId,
    render_image: Texture,
    preview_window_size: (i32, i32),

    // Gui state
    slider_value: f32,

    // Scene info
    scene: Scene,
    camera_transform: Transform,

    // Input
    input: InputState,
    right_click_down: bool,
    mouse_delta: Vec2,
    initial_mouse_pos: Vec2,  // Mouse position before dragging
}

#[cfg(disable)]
impl State
{
    pub fn new(renderer: &mut Renderer)->State
    {
        let render_image = renderer.create_texture(1, 1);
        let render_image_id = renderer.texture_to_egui_texture(&render_image, true);

        // Load scene
        let mut obj_path = std::env::current_exe().unwrap();
        obj_path.pop();
        obj_path = append_to_path(obj_path, "/../assets/dragon.obj");

        println!("Loading scene from disk...");
        let (scene, _) = load_scene_obj(obj_path.into_os_string().to_str().unwrap(), renderer);
        println!("Done!");

        let camera_transform = Transform
        {
            pos: Vec3 { x: 0.0, y: 0.0, z: -0.5 },
            rot: Default::default(),
            scale: Vec3 { x: 1.0, y: 1.0, z: 1.0 }
        };

        return State
        {
            render_image_id,
            render_image,
            preview_window_size: (1, 1),

            // GUI
            slider_value: 0.0,

            // Scene info
            scene,
            camera_transform,

            // Input
            input: Default::default(),
            right_click_down: false,
            mouse_delta: Default::default(),
            initial_mouse_pos: Default::default()
        };
    }

    pub fn main_update(&mut self, renderer: &mut Renderer, window: &Window,
                       egui_ctx: &mut egui::Context, egui_state: &mut egui_winit::State, delta_time: f32, input_diff: &mut InputDiff)
    {
        // Poll input
        poll_input(&mut self.input, input_diff);

        // Consume the accumulated egui inputs
        let egui_input = egui_state.take_egui_input(&window);

        // Update UI
        let gui_output = egui_ctx.run(egui_input, |ui|
        {
            self.gui_update(renderer, &egui_ctx, window);
        });

        // Update scene entities
        if self.right_click_down
        {
            self.camera_transform = camera_first_person_update(self.camera_transform, delta_time, self.input);
        }

        // Rendering
        {
            let win_size   = window.inner_size();
            let win_width  = win_size.width as i32;
            let win_height = win_size.height as i32;
            let scale      = window.scale_factor() as f32;

            egui_state.handle_platform_output(&window, gui_output.platform_output);
            let tris: Vec<ClippedPrimitive> = egui_ctx.tessellate(gui_output.shapes,
                                                                  gui_output.pixels_per_point);

            renderer.begin_frame();
            renderer.draw_scene(&self.scene, &self.render_image, transform_to_matrix(self.camera_transform));
            renderer.draw_egui(tris, &gui_output.textures_delta, scale);
        }

        // Notify winit that we're about to submit a new frame
        window.pre_present_notify();
        renderer.end_frame();
    }

    pub fn gui_update(&mut self, renderer: &mut Renderer, ctx: &egui::Context, window: &Window)
    {
        menu_bar(ctx);

        egui::SidePanel::left("side_panel")
            .resizable(true)
            .min_width(250.0)
            .default_width(250.0)
            .show(ctx, |ui|
            {
                ui.add_space(5.0);
                ui.heading("Rendering Settings");
                ui.separator();

                egui::CollapsingHeader::new("Preview Settings")
                    .default_open(true)
                    .show(ui, |ui|
                    {
                        ui.horizontal(|ui|
                        {
                            ui.label("Slider Value:");
                            ui.add(egui::widgets::Slider::new(&mut self.slider_value, 0.0..=1.0));
                        });
                    });

                ui.collapsing("Final Render Settings", |ui|
                {
                    ui.horizontal(|ui|
                    {
                        ui.label("Slider Value:");
                        ui.add(egui::widgets::Slider::new(&mut self.slider_value, 0.0..=1.0));
                    });
                });
            });

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui|
            {
                let size = (ui.available_size() * ctx.pixels_per_point()).round();
                let size_int = (size.x as i32, size.y as i32);
                if size_int.0 != self.preview_window_size.0 || size_int.1 != self.preview_window_size.1
                {
                    self.preview_window_size = size_int;
                    resize_egui_image(renderer, &mut self.render_image, self.render_image_id,
                                      self.preview_window_size.0, self.preview_window_size.1, true);
                }

                let size_in_points = egui::Vec2::new((size_int.0 as f32 / ctx.pixels_per_point()).round(),
                                                         (size_int.1 as f32 / ctx.pixels_per_point()).round());
                let to_draw = egui::load::SizedTexture
                {
                    id: self.render_image_id,
                    size: size_in_points 
                };

                ui.image(to_draw);

                // Handle mouse movement for looking around in first person

                let right_click_down = ctx.input(|i| i.pointer.secondary_down());
                let right_clicked = right_click_down && !self.right_click_down;
                let right_released = !right_click_down && self.right_click_down;
                let mouse_pos = ctx.input(|i| i.pointer.interact_pos()).unwrap_or(egui::Pos2::ZERO);
                let is_mouse_in_this_panel = ui.min_rect().contains(mouse_pos);

                if is_mouse_in_this_panel
                {
                    if right_clicked
                    {
                        // Store the initial mouse position
                        self.right_click_down = true;
                        self.initial_mouse_pos = Vec2 { x: mouse_pos.x, y: mouse_pos.y };
                    }

                    if right_click_down && window.has_focus()
                    {
                        let mouse_delta = ctx.input(|i| i.pointer.delta());
                        self.mouse_delta = Vec2 { x: mouse_delta.x, y: mouse_delta.y };
                        ctx.output_mut(|i| i.cursor_icon = egui::CursorIcon::None);

                        // Keep the mouse in place (using winit)
                        let winit_pos = winit::dpi::LogicalPosition::new(self.initial_mouse_pos.x, self.initial_mouse_pos.y);
                        let _ = window.set_cursor_position(winit_pos);
                    }

                    if right_released
                    {
                        self.right_click_down = false;
                        self.mouse_delta = Vec2::default();

                        // Reset cursor icon to default
                        ctx.output_mut(|i| i.cursor_icon = egui::CursorIcon::Default);
                    }
                }
            });
    }
}

#[cfg(disable)]
pub fn camera_first_person_update(prev: Transform, delta_time: f32, input: InputState)->Transform
{
    // Camera rotation
    const ROTATE_X_SPEED: f32 = 120.0 * DEG_TO_RAD;
    const ROTATE_Y_SPEED: f32 = 80.0 * DEG_TO_RAD;
    const MOVE_SPEED: f32 = 1.0;
    const MOUSE_SENSITIVITY: f32 = 0.1 * DEG_TO_RAD;
    // TODO: Remove these static vars because the compiler will complain about them
    static mut ANGLE_X: f32 = 0.0;
    static mut ANGLE_Y: f32 = 0.0;

    let mouse_delta = input.mouse_state.delta;

    let mouse_x = mouse_delta.x * MOUSE_SENSITIVITY;
    let mouse_y = mouse_delta.y * MOUSE_SENSITIVITY;

    let mut new_transform = prev;
    let input = Vec3
    {
        x: (input.keyboard_state.keys[VirtualKeycode::D as usize] as i32 - input.keyboard_state.keys[VirtualKeycode::A as usize] as i32) as f32,
        y: (input.keyboard_state.keys[VirtualKeycode::E as usize] as i32 - input.keyboard_state.keys[VirtualKeycode::Q as usize] as i32) as f32,
        z: (input.keyboard_state.keys[VirtualKeycode::W as usize] as i32 - input.keyboard_state.keys[VirtualKeycode::S as usize] as i32) as f32
    };

    let mut local_movement = Vec3 { x: input.x, y: 0.0, z: input.z };
    local_movement *= MOVE_SPEED;
    // Movement cap
    if dot2_vec3(local_movement) > MOVE_SPEED * MOVE_SPEED
    {
        local_movement = normalize_vec3(local_movement) * MOVE_SPEED;
    }

    local_movement *= delta_time;

    unsafe
    {
        ANGLE_X += mouse_x;
        ANGLE_Y += mouse_y;
        ANGLE_Y = ANGLE_Y.clamp(-90.0 * DEG_TO_RAD, 90.0 * DEG_TO_RAD);
    }

    // TODO: Handle gamepad input

    unsafe
    {
        let y_rot = angle_axis(Vec3::LEFT, ANGLE_Y);
        let x_rot = angle_axis(Vec3::UP,   ANGLE_X);

        new_transform.rot = quat_mul(x_rot, y_rot);
        let global_movement = rotate_vec3_with_quat(new_transform.rot, local_movement);
        new_transform.pos += global_movement;
        new_transform.pos += Vec3::UP * input.y * MOVE_SPEED * delta_time;

        return new_transform;
    }
}

#[cfg(disable)]
pub fn menu_bar(ctx: &egui::Context)
{
    use egui::{menu, Button};
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui|
    {
        menu::bar(ui, |ui|
        {
            ui.menu_button("File", |ui|
            {
                if ui.button("Open OBJ").clicked()
                {
                    println!("Clicked open!");
                }
            });

            ui.menu_button("Window", |ui|
            {
                #[cfg(debug_assertions)]
                if ui.button("GPU Asserts").clicked()
                {
                    println!("Clicked debug!");
                }
            });
        })
    });
}

#[cfg(disable)]
pub fn resize_egui_image(renderer: &mut Renderer, texture: &mut Texture, texture_id: egui::TextureId,
                         width: i32, height: i32, filter_near: bool)
{
    renderer.resize_texture(texture, width, height);
    renderer.update_egui_texture(texture, texture_id, filter_near);
}
