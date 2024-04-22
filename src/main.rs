
extern crate glfw;

use glfw::{Action, Context, Key};

fn main()
{
    use glfw::fail_on_errors;
    let mut glfw = glfw::init(fail_on_errors!()).unwrap();
    let (mut window, events) = glfw.create_window(300, 300, "Hello this is my window",
                                                  glfw::WindowMode::Windowed).expect("Failed to create a window.");

    window.make_current();
    window.set_key_polling(true);
        window.swap_buffers();

    while !window.should_close()
    {
        for(_, event) in glfw::flush_messages(&events)
        {
            println!("{:?}", event);
            match event
            {
                glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) =>
                {
                    window.set_should_close(true);
                },
                _ => {},
            }
        }
    }

    println!("Hello, world!");
}
