
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

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Vec2
{
    pub x: f32,
    pub y: f32
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct Vec3
{
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Vec4
{
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Aabb
{
    pub min: Vec3,
    pub max: Vec3
}

impl Aabb
{
    // Initialization for a neutral value with
    // respect to "grow" types of operations
    pub fn neutral()->Self
    {
        return Aabb
        {
            min: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX },
            max: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN },
        }
    }
}

pub fn lerp_f32(a: f32, b: f32, t: f32)->f32
{
    return a + (b - a) * t;
}

impl std::ops::Add for Vec3
{
    type Output = Self;

    fn add(self, other: Self)->Self::Output
    {
        return Self
        {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        };
    }
}

impl std::ops::Sub for Vec3
{
    type Output = Self;

    fn sub(self, other: Self)->Self::Output
    {
        return Self
        {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        };
    }
}

impl std::ops::Mul<f32> for Vec3
{
    type Output = Vec3;

    fn mul(self, rhs: f32)->Vec3
    {
        return Self
        {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs
        };
    }
}

impl std::ops::Div<f32> for Vec3
{
    type Output = Vec3;

    fn div(self, rhs: f32)->Vec3
    {
        return Self
        {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs
        };
    }
}

impl std::ops::Index<usize> for Vec3
{
    type Output = f32;
    fn index<'a>(&'a self, i: usize) -> &'a f32
    {
        debug_assert!(i < 3);
        let ptr = &self.x as *const f32;
        return unsafe { &*ptr.add(i) };
    }
}

impl std::ops::IndexMut<usize> for Vec3
{
    fn index_mut<'a>(&'a mut self, i: usize) -> &'a mut f32
    {
        debug_assert!(i < 3);
        let ptr = &mut self.x as *mut f32;
        return unsafe { &mut *ptr.add(i) };
    }
}

pub trait VectorOps
{
    fn normalize(&mut self);
}

impl VectorOps for Vec3
{
    fn normalize(&mut self)
    {
        let magnitude = self.x * self.x + self.y * self.y + self.z * self.z;
        self.x /= magnitude;
        self.y /= magnitude;
        self.z /= magnitude;
    }
}

impl VectorOps for Vec2
{
    fn normalize(&mut self)
    {
        let magnitude = self.x * self.x + self.y * self.y;
        self.x /= magnitude;
        self.y /= magnitude;
    }
}

impl std::fmt::Display for Vec3
{
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>)->std::result::Result<(), std::fmt::Error>
    {
        println!("(x: {}, y: {}, z: {})", self.x, self.y, self.z);
        return Ok(());
    }
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

pub fn to_u64_slice<T>(slice: &[T])->&[u64]
{
    let buf_size = slice.len() * std::mem::size_of::<T>();
    return unsafe
    {
        std::slice::from_raw_parts(slice.as_ptr() as *const _ as *const u64, buf_size / 8)
    };
}

pub fn grow_aabb_to_include_tri(aabb: &mut Aabb, t0: Vec3, t1: Vec3, t2: Vec3)
{
    aabb.min.x = aabb.min.x.min(t0.x);
    aabb.min.x = aabb.min.x.min(t1.x);
    aabb.min.x = aabb.min.x.min(t2.x);
    aabb.max.x = aabb.max.x.max(t0.x);
    aabb.max.x = aabb.max.x.max(t1.x);
    aabb.max.x = aabb.max.x.max(t2.x);

    aabb.min.y = aabb.min.y.min(t0.y);
    aabb.min.y = aabb.min.y.min(t1.y);
    aabb.min.y = aabb.min.y.min(t2.y);
    aabb.max.y = aabb.max.y.max(t0.y);
    aabb.max.y = aabb.max.y.max(t1.y);
    aabb.max.y = aabb.max.y.max(t2.y);

    aabb.min.z = aabb.min.z.min(t0.z);
    aabb.min.z = aabb.min.z.min(t1.z);
    aabb.min.z = aabb.min.z.min(t2.z);
    aabb.max.z = aabb.max.z.max(t0.z);
    aabb.max.z = aabb.max.z.max(t1.z);
    aabb.max.z = aabb.max.z.max(t2.z);
}

pub fn grow_aabb_to_include_aabb(aabb: &mut Aabb, to_include: Aabb)
{
    aabb.min.x = aabb.min.x.min(to_include.min.x);
    aabb.min.y = aabb.min.y.min(to_include.min.y);
    aabb.min.z = aabb.min.z.min(to_include.min.z);
    aabb.max.x = aabb.max.x.max(to_include.max.x);
    aabb.max.y = aabb.max.y.max(to_include.max.y);
    aabb.max.z = aabb.max.z.max(to_include.max.z);
}

pub fn compute_tri_bounds(t0: Vec3, t1: Vec3, t2: Vec3)->Aabb
{
    return Aabb
    {
        min: Vec3
        {
            x: t0.x.min(t1.x.min(t2.x)),
            y: t0.y.min(t1.y.min(t2.y)),
            z: t0.z.min(t1.z.min(t2.z)),
        },
        max: Vec3
        {
            x: t0.x.max(t1.x.max(t2.x)),
            y: t0.y.max(t1.y.max(t2.y)),
            z: t0.z.max(t1.z.max(t2.z)),
        }
    };
}

pub fn compute_tri_centroid(t0: Vec3, t1: Vec3, t2: Vec3)->Vec3
{
    return (t0 + t1 + t2) * 0.33333333333;
}

pub fn is_point_in_aabb(v: Vec3, aabb_min: Vec3, aabb_max: Vec3)->bool
{
    return v.x >= aabb_min.x && v.x <= aabb_max.x &&
           v.y >= aabb_min.y && v.y <= aabb_max.y &&
           v.z >= aabb_min.z && v.z <= aabb_max.z;
}