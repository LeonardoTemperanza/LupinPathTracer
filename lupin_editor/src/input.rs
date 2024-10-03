
// System for quickly polling inputs per frame.
// One could also fetch the previous frame's inputs
// for quickly comparing state.

use winit::event::*;

// TODO: Do we want to use other types for vectors etc.?
use lupin::base::*;

#[repr(C)]
pub enum GamepadButtonField
{
    None          = 0,
    DpadUp        = 1 << 1,
    DpadDown      = 1 << 2,
    DpadRight     = 1 << 4,
    DpadLeft      = 1 << 3,
    Start         = 1 << 5,
    Back          = 1 << 6,
    LeftThumb     = 1 << 7,
    RightThumb    = 1 << 8,
    LeftShoulder  = 1 << 9,
    RightShoulder = 1 << 10,
    A             = 1 << 11,
    B             = 1 << 12,
    X             = 1 << 13,
    Y             = 1 << 14
}

#[repr(C)]
pub enum VirtualKeycode
{
    Null = 0,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,

    Count
}

#[derive(Default, Clone, Copy)]
pub struct MouseState
{
    // In pixels starting from the top-left corner
    // of the application window. This is guaranteed
    // to be < 0 if the cursor is not on the window
    pub x_pos: i64,
    pub y_pos: i64,
    pub delta: Vec2,

    pub left_click: bool,
    pub right_click: bool
}

// All inactive gamepads will have every property set
// to 0, so no need to check for the active flag unless needed.
#[derive(Default, Clone, Copy)]
pub struct GamepadState
{
    pub active: bool,
    pub buttons: u32,
    
    // Values are normalized from 0 to 1.
    pub left_trigger: f32,
    pub right_trigger: f32,
    
    // Values are normalized from -1 to 1.
    pub left_stick_x: f32,
    pub left_stick_y: f32,
    pub right_stick_x: f32,
    pub right_stick_y: f32
}

#[derive(Default, Clone, Copy)]
pub struct KeyboardState
{
    pub keys: [bool; VirtualKeycode::Count as usize]
}

const MAX_ACTIVE_CONTROLLERS: usize = 10;
#[derive(Default, Clone, Copy)]
pub struct InputState
{
    pub gamepads: [GamepadState; MAX_ACTIVE_CONTROLLERS],
    pub mouse_state: MouseState,
    pub keyboard_state: KeyboardState,

    pub prev_gamepads: [GamepadState; MAX_ACTIVE_CONTROLLERS],
    pub prev_mouse_state: MouseState,
    pub prev_keyboard_state: KeyboardState
}

#[derive(Default, Clone, Copy)]
pub struct InputDiff
{
    pub keys_pressed:  [bool; VirtualKeycode::Count as usize],
    pub keys_held:     [bool; VirtualKeycode::Count as usize],

    // Right and up are positive
    pub mouse_delta: Vec2,
}

pub fn collect_inputs_winit(diff: &mut InputDiff, event: &winit::event::Event<()>)
{
    use winit::event::*;
    use winit::keyboard::*;
    use winit::platform::modifier_supplement::*;

    if let Event::WindowEvent { window_id, event } = event
    {
        match event
        {
            WindowEvent::KeyboardInput
            {
                event,
                ..
            } =>
            {
                if !event.repeat
                {
                    match &event.logical_key
                    {
                        Key::Character(c) if c == "w" =>
                        {
                            if event.state == ElementState::Pressed
                            {
                                diff.keys_pressed[VirtualKeycode::W as usize] = true;
                                diff.keys_held[VirtualKeycode::W as usize] = true;
                            }
                            else if event.state == ElementState::Released
                            {
                                diff.keys_held[VirtualKeycode::W as usize] = false;
                            }
                        }
                        Key::Character(c) if c == "a" =>
                        {
                            if event.state == ElementState::Pressed
                            {
                                diff.keys_pressed[VirtualKeycode::A as usize] = true;
                                diff.keys_held[VirtualKeycode::A as usize] = true;
                            }
                            else if event.state == ElementState::Released
                            {
                                diff.keys_held[VirtualKeycode::A as usize] = false;
                            }
                        }
                        Key::Character(c) if c == "s" =>
                        {
                            if event.state == ElementState::Pressed
                            {
                                diff.keys_pressed[VirtualKeycode::S as usize] = true;
                                diff.keys_held[VirtualKeycode::S as usize] = true;
                            }
                            else if event.state == ElementState::Released
                            {
                                diff.keys_held[VirtualKeycode::S as usize] = false;
                            }
                        }
                        Key::Character(c) if c == "d" =>
                        {
                            if event.state == ElementState::Pressed
                            {
                                diff.keys_pressed[VirtualKeycode::D as usize] = true;
                                diff.keys_held[VirtualKeycode::D as usize] = true;
                            }
                            else if event.state == ElementState::Released
                            {
                                diff.keys_held[VirtualKeycode::D as usize] = false;
                            }
                        }
                        Key::Character(c) if c == "q" =>
                        {
                            if event.state == ElementState::Pressed
                            {
                                diff.keys_pressed[VirtualKeycode::Q as usize] = true;
                                diff.keys_held[VirtualKeycode::Q as usize] = true;
                            }
                            else if event.state == ElementState::Released
                            {
                                diff.keys_held[VirtualKeycode::Q as usize] = false;
                            }
                        }
                        Key::Character(c) if c == "e" =>
                        {
                            if event.state == ElementState::Pressed
                            {
                                diff.keys_pressed[VirtualKeycode::E as usize] = true;
                                diff.keys_held[VirtualKeycode::E as usize] = true;
                            }
                            else if event.state == ElementState::Released
                            {
                                diff.keys_held[VirtualKeycode::E as usize] = false;
                            }
                        }
                        _ => (),
                    }                
                }
            },
            _ => {}
        }
    }
    else if let Event::DeviceEvent { device_id, event } = event
    {
        match event
        {
            DeviceEvent::MouseMotion { delta } =>
            {
                diff.mouse_delta.x += delta.0 as f32;
                diff.mouse_delta.y -= delta.1 as f32;  // In winit, up = negative.
            },
            _ => {}
        }
    }
}

pub fn poll_input(state: &mut InputState, input_diff: &mut InputDiff)
{
    for i in 0..VirtualKeycode::Count as usize
    {
        let key_down = input_diff.keys_pressed[i] || input_diff.keys_held[i];
        state.keyboard_state.keys[i] = key_down;
    }

    state.mouse_state.delta = input_diff.mouse_delta;
    input_diff.mouse_delta = Vec2::default();

    // Reset all one shot booleans
    for i in 0..VirtualKeycode::Count as usize
    {
        if input_diff.keys_pressed[i] || input_diff.keys_held[i]
        {
            input_diff.keys_pressed[i] = false;
        }
    }
}
