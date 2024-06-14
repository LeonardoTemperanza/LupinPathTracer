
// Contains all application logic that isn't already separated
// into different modules (such as serialization and rendering)

// Select the appropriate backend
use crate::renderer_wgpu::*;

use winit::window::Window;
use egui_winit::State;
use egui::ClippedPrimitive;
use crate::loader::*;

use crate::renderer_wgpu::*;

pub struct Core
{
    // egui texture ids
    render_image_id: egui::TextureId,
    render_image: Texture,
    preview_window_size: (i32, i32),

    // Gui state
    slider_value: f32
}

impl Core
{
    pub fn new(renderer: &mut Renderer)->Core
    {
        let render_image = renderer.create_texture(1920, 1080);
        let render_image_id = renderer.egui_texture_from_wgpu(&render_image, true);

        return Core
        {
            render_image_id,
            render_image,
            preview_window_size: (1920, 1080),

            // GUI
            slider_value: 0.0
        };
    }

    pub fn main_update(&mut self, renderer: &mut Renderer, window: &Window,
                       egui_ctx: &mut egui::Context, egui_state: &mut State)
    {
        // Poll input
        // Consume the accumulated egui inputs
        let egui_input = egui_state.take_egui_input(&window);

        // Update UI
        let gui_output = egui_ctx.run(egui_input, |ui|
        {
            self.gui_update(renderer, &egui_ctx);
        });

        // Update scene entities
        camera_first_person_update(2);

        // Rendering
        {
            let win_size   = window.inner_size();
            let win_width  = win_size.width as i32;
            let win_height = win_size.height as i32;
            let scale      = window.scale_factor() as f32;

            egui_state.handle_platform_output(&window, gui_output.platform_output);
            let tris: Vec<ClippedPrimitive> = egui_ctx.tessellate(gui_output.shapes,
                                                                  gui_output.pixels_per_point);

            renderer.prepare_frame();
            renderer.draw_scene(&self.render_image);

            // Draw gui last, as an overlay
            renderer.draw_egui(tris, &gui_output.textures_delta, win_width, win_height, scale);
        }

        // Notify winit that we're about to submit a new frame
        window.pre_present_notify();
        renderer.swap_buffers();
    }

    pub fn gui_update(&mut self, renderer: &mut Renderer, ctx: &egui::Context)
    {
        menu_bar(ctx);

        let (size_x, size_y) = get_texture_size(&self.render_image);
        let size = egui::Vec2::new(size_x as f32, size_y as f32);

        let mut to_draw = egui::load::SizedTexture
        {
            id: self.render_image_id,
            size: size
        };

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
                let size = ui.available_size();
                let size_int = (size.x as i32, size.y as i32);
                if size_int.0 != self.preview_window_size.0 || size_int.1 != self.preview_window_size.1
                {
                    self.preview_window_size = (size.x as i32, size.y as i32);
                    to_draw.size = size;
                    resize_egui_image(renderer, &mut self.render_image, self.render_image_id,
                                      self.preview_window_size.0, self.preview_window_size.1, true);
                }
                ui.image(to_draw);
            });
    }
}

pub fn camera_first_person_update(prev: i32)->i32
{
    return 0;
}

pub fn menu_bar(ctx: &egui::Context)
{
    use egui::{menu, Button};
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui|
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

pub fn resize_egui_image(renderer: &mut Renderer, texture: &mut Texture, texture_id: egui::TextureId,
                         width: i32, height: i32, filter_near: bool)
{
    renderer.resize_texture(texture, width, height);
    renderer.update_egui_texture(texture, texture_id, filter_near);
}
