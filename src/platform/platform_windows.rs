
use crate::platform::*;

use windows_sys::*;
use windows_sys::core::*;
use windows_sys::Win32::*;
use windows_sys::Win32::Foundation::*;
use windows_sys::Win32::UI::WindowsAndMessaging::*;
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleA;
use windows_sys::Win32::System::Threading::*;
use windows_sys::Win32::UI::WindowsAndMessaging::SetProcessDPIAware;
use std::os::raw::c_void;

// A sort of channel used to communicate
// with the window_events_fiber
struct EventsData
{
    // Passed from main fiber
    main_fiber: *const c_void,
    window: isize,

    // Passed to main fiber
    quit: bool,
    is_in_modal_loop: bool
}

pub struct WindowsCtx
{
    // Fibers for window management
    main_fiber: *const c_void,
    events_fiber: *const c_void,
    events_data: Box<EventsData>,  // Requires heap allocation because it's passed to another fiber

    pub window_obj: Window
}

impl PlatformImpl for WindowsCtx
{
    fn new()->WindowsCtx
    {
        unsafe
        {
            let instance = GetModuleHandleA(std::ptr::null());
            debug_assert!(instance != 0);

            // Set DPI awareness of current process.
            // This will make it so the window is not blurry
            SetProcessDPIAware();

            let window_class = s!("window");

            let wc = WNDCLASSA
            {
                hCursor: LoadCursorW(0, IDC_ARROW),
                hInstance: instance,
                lpszClassName: window_class,
                style: CS_HREDRAW | CS_VREDRAW,
                lpfnWndProc: Some(wndproc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hIcon: 0,
                hbrBackground: 0,
                lpszMenuName: std::ptr::null(),
            };

            let atom = RegisterClassA(&wc);
            debug_assert!(atom != 0);

            let mut events_data_heap: Box<EventsData> = Box::new(std::mem::zeroed());
            let events_data_ptr = &mut *events_data_heap as *mut EventsData as *mut c_void;


            let window = CreateWindowExA(0, window_class, s!("This is a sample window"),
                                         WS_OVERLAPPEDWINDOW,
                                         CW_USEDEFAULT, CW_USEDEFAULT,
                                         CW_USEDEFAULT, CW_USEDEFAULT,
                                         0, 0, instance, std::ptr::null());

            let window_obj = Window { handle: window };

            // Pass info to created window
            SetWindowLongPtrA(window, GWLP_USERDATA, events_data_ptr as isize);

            events_data_heap.window = window;

            // NOTE: Microsoft Insanity(tm) incoming...
            // need to create 2 separate fibers, one for
            // application code and one for window event
            // handling, to implement "proper" window
            // resizing. Yes, you've read that right.
            // See WndProc for more details.
            let main_fiber  = ConvertThreadToFiber(std::ptr::null());
            let events_fiber = CreateFiber(0, Some(window_events_fiber), events_data_ptr);

            events_data_heap.main_fiber = main_fiber;

            return WindowsCtx
            {
                main_fiber,
                events_fiber,
                events_data: events_data_heap,
                window_obj
            }
        }
    }

    fn show_window(&self)
    {
        unsafe
        {
            ShowWindow(self.events_data.window, SW_SHOW);
        }
    }

    fn get_window_handle<'a>(&'a self)->&'a Window
    {
        return &self.window_obj;
    }

    fn get_framebuffer_size(&self)->(u32, u32)
    {
        unsafe
        {
            let mut win_rect: RECT = std::mem::zeroed();
            let ok = GetClientRect(self.events_data.window, &mut win_rect);
            assert!(ok != 0);

            let size_x = (win_rect.right  - win_rect.left).max(0) as u32;
            let size_y = (win_rect.bottom - win_rect.top ).max(0) as u32;
            return (size_x, size_y);
        }
    }

    fn handle_window_events(&self)->bool
    {
        unsafe
        {
            // The events fiber should set quit here
            SwitchToFiber(self.events_fiber);
            return self.events_data.quit;
        }
    }
}

unsafe extern "system" fn window_events_fiber(param: *mut c_void)
{
    let events_data: *mut EventsData = param as *mut EventsData;

    loop
    {
        (*events_data).quit = false;
        let mut msg = std::mem::zeroed();

        // Window messages
        while PeekMessageA(&mut msg, (*events_data).window, 0, 0, true as u32) != 0
        {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);
        }
        
        msg = std::mem::zeroed();

        // Thread messages
        while PeekMessageA(&mut msg, 0, 0, 0, true as u32) != 0
        {
            TranslateMessage(&msg);
            DispatchMessageA(&msg);

            if msg.message == WM_QUIT { (*events_data).quit = true; }
        }

        // Go back to application code
        SwitchToFiber((*events_data).main_fiber);
    }
}

unsafe fn break_out_of_modal_loop(events_data: *mut EventsData)
{
    // Setting is_in_modal_loop signal to the
    // main fiber that we're still in a modal loop,
    // and so retrieving messages from wndproc should
    // be done carefully.
    (*events_data).is_in_modal_loop = true;
    SwitchToFiber((*events_data).main_fiber);
    (*events_data).is_in_modal_loop = false;
}

// NOTE: Window callback function. Since Microsoft wants programs to get stuck
// inside this function at all costs, a weird hack is being used to "break free"
// of this function. Using a fiber I go back to regular code execution during
// modal loop events (which are events that don't let the program exit this)
// In addition to this nonsense, at some points when stuck in a modal loop, no
// messages are sent so there's no way to execute any code, so to solve that
// we need to fire "timer" messages which have a very low precision of 10ms.
// This will lead to stuttering during the modal loops (there is no way to fix it).
// This is insane but it's also the only -somewhat- clean way to keep rendering
// stuff while in the modal loop. (using a separate thread also has issues)
extern "system" fn wndproc(window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT
{
    unsafe
    {
        struct DeferMessage
        {
            active: bool,
            msg: u32,
            w_param: WPARAM,
            l_param: LPARAM
        }

        static mut CAPTION_CLICK: DeferMessage = unsafe { std::mem::zeroed() };
        static mut TOP_BUTTON_CLICK: DeferMessage = unsafe { std::mem::zeroed() };  // Close, minimize or maximize

        match message
        {
            WM_DESTROY =>
            {
                PostQuitMessage(0);
                return DefWindowProcA(window, message, wparam, lparam);
            },

            // Entering modal loops will start a timer,
            // that way the system will continue to
            // dispatch messages
            WM_ENTERSIZEMOVE =>
            {
                CAPTION_CLICK.active = false;
                TOP_BUTTON_CLICK.active = false;
                
                // USER_TIMER_MINIMUM is 10ms, so not very precise.
                SetTimer(window, 0, USER_TIMER_MINIMUM, None);
                return DefWindowProcA(window, message, wparam, lparam);
            },
            WM_EXITSIZEMOVE =>
            {
                KillTimer(window, 0);
                return DefWindowProcA(window, message, wparam, lparam);
            },
            // It's a low-priority message, so WM_SIZE
            // should be prioritized over this
            WM_TIMER =>
            {
                if window != 0
                {
                    let events_data = GetWindowLongPtrA(window, GWLP_USERDATA) as *mut EventsData;
                    if events_data as isize != 0  // Check it's not a nullptr
                    {
                        break_out_of_modal_loop(events_data);
                    }
                }
                
                return DefWindowProcA(window, message, wparam, lparam);
            },
            // Redraw immediately after resize (we don't want black bars)
            WM_SIZE =>
            {
                if window != 0
                {
                    let events_data = GetWindowLongPtrA(window, GWLP_USERDATA) as *mut EventsData;
                    if events_data as isize != 0  // Check it's not a nullptr
                    {
                        break_out_of_modal_loop(events_data);
                    }
                }

                // Reset the timer. Without this, the resizing is laggy
                // because we let the app update twice per resize.
                SetTimer(window, 0, USER_TIMER_MINIMUM, None);
                return DefWindowProcA(window, message, wparam, lparam);
            },

            ///////////////////////////////////////////////////////////
            // NOTE: Clicks on the client area completely block everything;
            // not event -timer events- are sent to the window, so there's
            // just no way of executing any code while the WM_NCLBUTTONDOWN
            // message is being handled. (for example, holding down the mouse
            // button over the title bar but not moving the mouse)
            // Luckily, we can actually defer sending that message until we
            // move the mouse. As soon as we move the mouse, we'll also receive
            // WM_ENTERSIZEMOVE and the WM_TIMER messages. As for clicking on buttons,
            // we can simply not handle the WM_NCLBUTTONDOWN message at all and just
            // reimplement the functionality of the buttons on WM_NCLBUTTONUP.
            // This technique is also used in chromium and here:
            // https://github.com/glfw/glfw/pull/1426
            // TODO

            _ => return DefWindowProcA(window, message, wparam, lparam)
        }
    }
}

pub struct Window
{
    handle: isize
}

// Window trait implementation
use raw_window_handle::*;
impl HasWindowHandle for Window
{
    fn window_handle(&self)->Result<WindowHandle, HandleError>
    {
        use std::num::NonZeroIsize;
        let hwnd: NonZeroIsize = match NonZeroIsize::new(self.handle)
        {
            Some(v) => v,
            None => return Err(HandleError::Unavailable),
        };

        let mut win32_handle = Win32WindowHandle::new(hwnd);
        win32_handle.hinstance = Some(NonZeroIsize::new(unsafe { GetWindowLongPtrW(self.handle, GWLP_HINSTANCE) }).unwrap());
        let raw_window_handle = RawWindowHandle::Win32(win32_handle);
        unsafe
        {
            return Ok(WindowHandle::borrow_raw(raw_window_handle));
        }
    }
}

impl HasDisplayHandle for Window
{
    fn display_handle(&self)->Result<DisplayHandle, HandleError>
    {
        return Ok(DisplayHandle::windows());
    }
}