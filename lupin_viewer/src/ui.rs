
pub struct StateUI
{
    //
}

#[cfg(disable)]
impl StateUI
{
    pub fn new()->StateUI
    {
        return StateUI {};
    }

    pub fn main_update(&mut self, renderer: &mut Renderer, window: &Window,
                       egui_ctx: &mut egui::Context, egui_state: &mut egui_winit::StateUI, delta_time: f32, input_diff: &mut InputDiff)
    {
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

            //renderer.begin_frame();
            //renderer.draw_scene(&self.scene, &self.render_image, transform_to_matrix(self.camera_transform));
            //renderer.draw_egui(tris, &gui_output.textures_delta, scale);
        }
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
