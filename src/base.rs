
use std::
{
    future::Future,
    ptr,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};


////////
// Synchronization
// https://stackoverflow.com/questions/56252798/how-do-i-execute-an-async-await-function-without-using-any-external-dependencies

pub fn wait_for<F>(f: F) -> F::Output
where
    F: Future,
{
    let waker = my_waker();
    let mut context = Context::from_waker(&waker);

    let mut t = Box::pin(f);
    let t = t.as_mut();

    loop
    {
        match t.poll(&mut context)
        {
            Poll::Ready(v) => return v,
            Poll::Pending => panic!("This executor does not support futures that are not ready"),
        }
    }
}

type WakerData = *const ();

unsafe fn clone(_: WakerData) -> RawWaker { my_raw_waker() }
unsafe fn wake(_: WakerData) {}
unsafe fn wake_by_ref(_: WakerData) {}
unsafe fn drop(_: WakerData) {}

static MY_VTABLE: RawWakerVTable = RawWakerVTable::new(clone, wake, wake_by_ref, drop);

fn my_raw_waker() -> RawWaker { return RawWaker::new(ptr::null(), &MY_VTABLE) }

fn my_waker() -> Waker { unsafe { return Waker::from_raw(my_raw_waker()) } }


////////
// Math

pub struct Vec2
{
    x: f32,
    y: f32
}

pub struct Vec3
{
    x: f32,
    y: f32,
    z: f32
}

pub struct Vec4
{
    x: f32,
    y: f32,
    z: f32,
    w: f32
}


////////
// Miscellaneous

// From https://stackoverflow.com/questions/74322541/how-to-append-to-pathbuf
pub fn append_to_path(p: std::path::PathBuf, s: &str)->std::path::PathBuf
{
    let mut p = p.into_os_string();
    p.push(s);
    return p.into();
}

// Useful for passing buffers to the GPU
pub fn to_u8_slice<T>(slice: &[T])->&[u8]
{
    let buf_size = slice.len() * std::mem::size_of::<T>();
    return unsafe
    {
        std::slice::from_raw_parts(slice.as_ptr() as *const _ as *const u8, buf_size)
    };
}