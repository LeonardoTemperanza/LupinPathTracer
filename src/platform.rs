
use raw_window_handle::{ HasWindowHandle, HasDisplayHandle };

#[cfg(target_os = "windows")]
mod platform_windows;
#[cfg(target_os = "linux")]
mod platform_linux;

pub trait PlatformImpl
{
    fn new()->Self;
    fn show_window(&self);
    fn get_window_handle<'a>(&'a self)->&'a Window;
    fn get_framebuffer_size(&self)->(u32, u32);
    fn handle_window_events(&self)->bool;
}

pub trait WindowTrait: HasWindowHandle + HasDisplayHandle {}

#[cfg(target_os = "windows")]
pub type Platform = platform_windows::WindowsCtx;
#[cfg(target_os = "windows")]
pub type Window = platform_windows::Window;

#[cfg(target_os = "linux")]
pub type Platform = platform_linux::LinuxCtx;

