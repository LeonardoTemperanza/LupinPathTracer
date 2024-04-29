
// Contains all application logic that isn't already separated
// into different modules (such as serialization and rendering)

// Select the appropriate backend
use crate::renderer_wgpu::*;

use winit::window::Window;

pub struct Core
{
    main_shader:  ShaderHandle,
    main_program: ProgramHandle
}

impl Core
{
    pub fn new(renderer: &mut Renderer)->Core
    {
        let main_shader: ShaderHandle = renderer.compile_shader(include_str!("../assets/raytracer.wgsl"));
        let main_program: ComputeProgramHandle = renderer.create_program(main_shader);

        return Core
        {
            main_shader,
            main_program
        };
    }

    pub fn main_update(&mut self, renderer: &mut Renderer,
                       egui_ctx: &mut egui::Context, egui_input: egui::RawInput)->egui::FullOutput
    {
        let gui_output = egui_ctx.run(egui_input, |ui|
        {
            gui_update(&egui_ctx);
        });

        // Update scene entities
        camera_first_person_update(2);

        return gui_output;
    }
}

pub fn camera_first_person_update(prev: i32)->i32
{
    return 0;
}

pub fn gui_update(ui: &egui::Context)
{
    egui::Window::new("Streamline CFD")
        // .vscroll(true)
        .default_open(true)
        .max_width(1000.0)
        .max_height(800.0)
        .default_width(800.0)
        .resizable(true)
        .anchor(egui::Align2::LEFT_TOP, [0.0, 0.0])
        .show(&ui, |ui| {
            if ui.add(egui::Button::new("Click me")).clicked() {
                println!("PRESSED")
            }

            ui.label("Slider");
            // ui.add(egui::Slider::new(_, 0..=120).text("age"));
            ui.end_row();

            // proto_scene.egui(ui);
        });
}