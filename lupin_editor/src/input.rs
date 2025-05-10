
// System for quickly polling inputs per frame.

use winit::event::*;
use crate::base::*;

pub enum Key
{
    W = 0,  // NOTE: Do not change this.
    A,
    S,
    D,
    Q,
    E,
    P,
    _1,
    LSHIFT,
    LCTRL,
    SPACEBAR,

    NumValues,
}

impl<T> std::ops::Index<Key> for [T; Key::NumValues as usize]
{
    type Output = T;
    fn index(&self, idx: Key) -> &Self::Output
    {
        return &self[idx as usize];
    }
}
impl<T> std::ops::IndexMut<Key> for [T; Key::NumValues as usize]
{
    fn index_mut(&mut self, idx: Key) -> &mut Self::Output
    {
        return &mut self[idx as usize];
    }
}

#[derive(Default, Clone, Copy)]
pub struct Input
{
    pub lmouse: ButtonState,
    pub rmouse: ButtonState,
    pub mmouse: ButtonState,
    pub keys: [ButtonState; Key::NumValues as usize],
    pub mouse_dx: f32,  // Pixels/dpi (inches), right is positive
    pub mouse_dy: f32,  // Pixels/dpi (inches), up is positive
}

// "Pressed" and "Released" will be active for a single frame.
// "Pressing" will always be true when "Pressed" is true,
// "Pressing" will always be false when "Released" is true
#[derive(Default, Clone, Copy)]
pub struct ButtonState
{
    pub pressed: bool,
    pub pressing: bool,
    pub released: bool
}

pub fn begin_input_events(input: &mut Input)
{
    // Zero out mouse delta
    input.mouse_dx = 0.0;
    input.mouse_dy = 0.0;

    // Zero out "one shot" booleans
    input.lmouse.pressed  = false;
    input.lmouse.released = false;
    input.rmouse.pressed  = false;
    input.rmouse.pressed  = false;
    input.mmouse.released = false;
    input.mmouse.released = false;
    for key in &mut input.keys
    {
        key.pressed  = false;
        key.released = false;
    }
}

pub fn process_input_event(input: &mut Input, event: &winit::event::Event<()>)
{
    use winit::event::*;

    fn press(button: &mut ButtonState)
    {
        button.pressed  = true;
        button.pressing = true;
    }

    fn release(button: &mut ButtonState)
    {
        button.pressing = false;
        button.released = true;
    }

    if let Event::WindowEvent { window_id: _, event } = event
    {
        match event
        {
            WindowEvent::KeyboardInput { event, .. } =>
            'block: {
                if event.repeat { break 'block; }

                let action: fn(&mut ButtonState);
                if event.state == ElementState::Pressed {
                    action = press;
                } else if event.state == ElementState::Released {
                    action = release;
                } else {
                    break 'block;
                }

                use winit::keyboard::Key::Character;
                use winit::keyboard::Key::Named;
                use winit::keyboard::NamedKey;
                match &event.logical_key
                {
                    Character(c) if c.eq_ignore_ascii_case("w") => { action(&mut input.keys[Key::W]); }
                    Character(c) if c.eq_ignore_ascii_case("a") => { action(&mut input.keys[Key::A]); }
                    Character(c) if c.eq_ignore_ascii_case("s") => { action(&mut input.keys[Key::S]); }
                    Character(c) if c.eq_ignore_ascii_case("d") => { action(&mut input.keys[Key::D]); }
                    Character(c) if c.eq_ignore_ascii_case("q") => { action(&mut input.keys[Key::Q]); }
                    Character(c) if c.eq_ignore_ascii_case("e") => { action(&mut input.keys[Key::E]); }
                    Named(k) if *k == NamedKey::Shift => { action(&mut input.keys[Key::LSHIFT]); }
                    _ => (),
                }
            },
            WindowEvent::MouseInput { state, button, .. } =>
            'block: {
                let action: fn(&mut ButtonState);
                if *state == ElementState::Pressed {
                    action = press;
                } else if *state == ElementState::Released {
                    action = release;
                } else {
                    break 'block;
                }

                use winit::event::MouseButton;
                match button
                {
                    MouseButton::Left   => { action(&mut input.lmouse); }
                    MouseButton::Right  => { action(&mut input.rmouse); }
                    MouseButton::Middle => { action(&mut input.mmouse); }
                    _ => (),
                }
            }
            _ => {}
        }
    }
    else if let Event::DeviceEvent { device_id: _, event } = event
    {
        match event
        {
            DeviceEvent::MouseMotion { delta } =>
            {
                input.mouse_dx += delta.0 as f32;
                input.mouse_dy -= delta.1 as f32;  // In winit, up = negative.
            },
            _ => {}
        }
    }
}
