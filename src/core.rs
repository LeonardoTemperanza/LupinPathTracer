
// Contains all application logic that isn't already separated
// into different modules (such as serialization and rendering)

// Select the appropriate backend
use crate::renderer_wgpu::*;

use egui_winit_platform::Platform;
use winit::window::Window;

pub struct Core
{
    main_shader:  ShaderHandle,
    main_program: ProgramHandle,

    // @tmp
    egui_demo_app: egui_demo_lib::DemoWindows
}

impl Core
{
    pub fn new(renderer: &mut Renderer)->Core
    {
        let main_shader: ShaderHandle = renderer.compile_shader(include_str!("../assets/raytracer.wgsl"));
        let main_program: ComputeProgramHandle = renderer.create_program(main_shader);

        let egui_demo_app = egui_demo_lib::DemoWindows::default();

        return Core
        {
            main_shader,
            main_program,
            egui_demo_app,
        };
    }

    pub fn main_update(&mut self, renderer: &mut Renderer, platform: &mut Platform, window: &Window)->egui::FullOutput
    {
        let gui_output = self.gui_update(platform, window);

        // Update scene entities
        camera_first_person_update(2);

        return gui_output;
    }

    pub fn gui_update(&mut self, platform: &mut Platform, window: &Window)->egui::FullOutput
    {
        platform.begin_frame();

        self.egui_demo_app.ui(&platform.context());

        return platform.end_frame(Some(window));
    }
}

pub fn camera_first_person_update(prev: i32)->i32
{
    return 0;
}
