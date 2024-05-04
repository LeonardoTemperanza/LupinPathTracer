
// Contains all application logic that isn't already separated
// into different modules (such as serialization and rendering)

// Select the appropriate backend
use crate::renderer_wgpu::*;

use winit::window::Window;
use egui_winit::State;
use egui::ClippedPrimitive;

pub struct Core
{
    
}

impl Core
{
    pub fn new(renderer: &mut Renderer)->Core
    {
        return Core {};
    }

    pub fn main_update(&mut self, egui_renderer: &mut EGUIRenderState,
                       renderer: &mut Renderer, window: &Window,
                       egui_ctx: &mut egui::Context, egui_state: &mut State)
    {
        // Poll input
        // Consume the accumulated egui inputs
        let egui_input = egui_state.take_egui_input(&window);

        // Update UI
        let gui_output = egui_ctx.run(egui_input, |ui|
        {
            gui_update(&egui_ctx);
        });

        // Update scene entities
        camera_first_person_update(2);

        // Rendering
        let win_size   = window.inner_size();
        let win_width  = win_size.width as i32;
        let win_height = win_size.height as i32;
        let scale      = window.scale_factor() as f32;

        egui_state.handle_platform_output(&window, gui_output.platform_output);
        let tris: Vec<ClippedPrimitive> = egui_ctx.tessellate(gui_output.shapes,
                                                 gui_output.pixels_per_point);

        renderer.prepare_frame();
        renderer.draw_scene();

        // Draw gui last, as an overlay
        renderer.draw_egui(egui_renderer, tris, &gui_output.textures_delta, win_width, win_height, scale);

        // Notify winit that we're about to submit a new frame
        window.pre_present_notify();
        renderer.swap_buffers();
    }
}

pub fn camera_first_person_update(prev: i32)->i32
{
    return 0;
}

pub fn gui_update(ui: &egui::Context)
{
    menu_bar(ui);

    egui::Window::new("Streamline CFD")
        .default_open(true)
        .default_width(800.0)
        .resizable(true)
        .movable(true)
        .show(&ui, |ui| {
            if ui.add(egui::Button::new("Click me")).clicked() {
                println!("PRESSED")
            }

            ui.label("Slider");
            //ui.add(egui::Slider::new(_, 0..=120).text("age"));
            ui.end_row();

            // proto_scene.egui(ui);
        });
}

pub fn menu_bar(ui: &egui::Context)
{
    use egui::{menu, Button};
    egui::TopBottomPanel::top("menu_bar").show(ui, |ui|
    {
        menu::bar(ui, |ui|
        {
            ui.menu_button("File", |ui|
            {
                if ui.button("Open").clicked()
                {
                    println!("Clicked open!");
                }
            })
        })
    });
}
