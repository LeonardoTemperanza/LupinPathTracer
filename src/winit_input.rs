
// System for quickly polling inputs per frame.
// One could also fetch the previous frame's inputs
// for quickly comparing state.

pub enum GamepadButtonField
{
    Gamepad_None           = 0,
    Gamepad_Dpad_Up        = 1 << 1,
    Gamepad_Dpad_Down      = 1 << 2,
    Gamepad_Dpad_Left      = 1 << 3,
    Gamepad_Dpad_Right     = 1 << 4,
    Gamepad_Start          = 1 << 5,
    Gamepad_Back           = 1 << 6,
    Gamepad_LeftThumb      = 1 << 7,
    Gamepad_RightThumb     = 1 << 8,
    Gamepad_LeftShoulder   = 1 << 9,
    Gamepad_RightShoulder  = 1 << 10,
    Gamepad_A              = 1 << 11,
    Gamepad_B              = 1 << 12,
    Gamepad_X              = 1 << 13,
    Gamepad_Y              = 1 << 14
}

pub enum VirtualKeycode
{
    Keycode_Null = 0,
    Keycode_LMouse,
    Keycode_RMouse,
    Keycode_MMouse,
    Keycode_A,
    Keycode_B,
    Keycode_C,
    Keycode_D,
    Keycode_E,
    Keycode_F,
    Keycode_G,
    Keycode_H,
    Keycode_I,
    Keycode_J,
    Keycode_K,
    Keycode_L,
    Keycode_M,
    Keycode_N,
    Keycode_O,
    Keycode_P,
    Keycode_Q,
    Keycode_R,
    Keycode_S,
    Keycode_T,
    Keycode_U,
    Keycode_V,
    Keycode_W,
    Keycode_X,
    Keycode_Y,
    Keycode_Z,

    Keycode_Count
}

struct OS_MouseState
{
    bool active; // On the window?
    
    // In pixels starting from the top-left corner
    // of the application window. This is guaranteed
    // to be < 0 if the cursor is not on the window
    int64_t xPos;
    int64_t yPos;
};

// All inactive gamepads will have every property set
// to 0, so no need to check for the active flag unless needed.
struct OS_GamepadState
{
    bool active;
    uint32_t buttons;
    
    // Values are normalized from 0 to 1.
    float leftTrigger;
    float rightTrigger;
    
    // Values are normalized from -1 to 1.
    float leftStickX;
    float leftStickY;
    float rightStickX;
    float rightStickY;
};

#define MaxActiveControllers 8
struct OS_InputState
{
    OS_GamepadState gamepads[MaxActiveControllers];
    OS_MouseState mouse;
    
    // This includes mouse buttons
    bool virtualKeys[Keycode_Count];
}

pub fn poll_input()
{

}