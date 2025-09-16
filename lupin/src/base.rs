
use std::
{
    future::Future,
    ptr,
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

#[allow(unused_macros)]
macro_rules! static_assert {
    ($($tt:tt)*) => {
        const _: () = assert!($($tt)*);
    }
}

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

// Useful constants
pub const DEG_TO_RAD: f32 = 0.017453292;
pub const RAD_TO_DEG: f32 = 57.29578049;

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vec2
{
    pub x: f32,
    pub y: f32
}

#[derive(Default, Clone, Copy, Debug)]
#[repr(C)]
pub struct Vec3
{
    pub x: f32,
    pub y: f32,
    pub z: f32
}

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Aabb
{
    pub min: Vec3,
    pub max: Vec3
}

impl Vec3
{
    pub const LEFT:     Vec3 = Vec3 { x: -1.0, y: 0.0,  z: 0.0  };
    pub const RIGHT:    Vec3 = Vec3 { x: 1.0,  y: 0.0,  z: 0.0  };
    pub const UP:       Vec3 = Vec3 { x: 0.0,  y: 1.0,  z: 0.0  };
    pub const DOWN:     Vec3 = Vec3 { x: 0.0,  y: -1.0, z: 0.0  };
    pub const FORWARD:  Vec3 = Vec3 { x: 0.0,  y: 0.0,  z: 1.0  };
    pub const BACKWARD: Vec3 = Vec3 { x: 0.0,  y: 0.0,  z: -1.0 };

    pub fn new(x: f32, y: f32, z: f32) -> Self
    {
        return Self { x, y, z };
    }

    pub fn is_zero(&self) -> bool
    {
        return self.x == 0.0 && self.y == 0.0 && self.z == 0.0;
    }
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

impl Vec4
{
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self
    {
        return Self { x, y, z, w };
    }

    #[inline]
    pub fn is_zero(&self) -> bool
    {
        return self.x == 0.0 && self.y == 0.0 && self.z == 0.0 && self.w == 0.0;
    }

    #[inline]
    pub fn normalized(&self) -> Vec4
    {
        let magnitude = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        return Vec4 { x: self.x / magnitude, y: self.y / magnitude, z: self.z / magnitude, w: self.w / magnitude };
    }
}

impl Aabb
{
    /// Initialization for a neutral value with
    /// respect to "grow" types of operations
    #[inline]
    pub fn neutral()->Self
    {
        return Aabb
        {
            min: Vec3 { x: f32::MAX, y: f32::MAX, z: f32::MAX },
            max: Vec3 { x: f32::MIN, y: f32::MIN, z: f32::MIN },
        }
    }
}

#[inline]
pub fn lerp_f32(a: f32, b: f32, t: f32)->f32
{
    return a + (b - a) * t;
}

#[inline]
pub fn square_f32(n: f32)->f32
{
    return n*n;
}

impl std::ops::AddAssign<Vec2> for Vec2
{
    #[inline]
    fn add_assign(&mut self, rhs: Vec2)
    {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Vec3
{
    pub fn min(self, other: Self) -> Self
    {
        return Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        };
    }

    pub fn max(self, other: Self) -> Self
    {
        return Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        };
    }

    pub fn ones() -> Self
    {
        return Self { x: 1.0, y: 1.0, z: 1.0 };
    }
}

impl std::ops::Add for Vec3
{
    type Output = Self;

    #[inline]
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

impl std::ops::AddAssign<Vec3> for Vec3
{
    #[inline]
    fn add_assign(&mut self, rhs: Vec3)
    {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}

impl std::ops::Sub for Vec3
{
    type Output = Self;

    #[inline]
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

    #[inline]
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

impl std::ops::MulAssign<f32> for Vec3
{
    #[inline]
    fn mul_assign(&mut self, rhs: f32)
    {
        self.x *= rhs;
        self.y *= rhs;
        self.z *= rhs;
    }
}

impl std::ops::Div<f32> for Vec3
{
    type Output = Vec3;

    #[inline]
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

impl std::ops::Index<isize> for Vec3
{
    type Output = f32;
    #[inline]
    fn index<'a>(&'a self, i: isize) -> &'a f32
    {
        debug_assert!(i < 3 && i >= 0);
        let ptr = &self.x as *const f32;
        return unsafe { &*ptr.add(i as usize) };
    }
}

impl std::ops::IndexMut<isize> for Vec3
{
    #[inline]
    fn index_mut<'a>(&'a mut self, i: isize) -> &'a mut f32
    {
        debug_assert!(i < 3 && i >= 0);
        let ptr = &mut self.x as *mut f32;
        return unsafe { &mut *ptr.add(i as usize) };
    }
}

#[inline]
pub fn normalize_vec3(v: Vec3)->Vec3
{
    let magnitude = v.x * v.x + v.y * v.y + v.z * v.z;
    return Vec3 { x: v.x / magnitude, y: v.y / magnitude, z: v.z / magnitude };
}

#[inline]
pub fn normalize_vec2(v: Vec2)->Vec2
{
    let magnitude = v.x * v.x + v.y * v.y;
    return Vec2 { x: v.x / magnitude, y: v.y / magnitude };
}

impl std::fmt::Display for Vec2
{
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>)->std::result::Result<(), std::fmt::Error>
    {
        print!("(x: {}, y: {})", self.x, self.y);
        return Ok(());
    }
}

impl std::fmt::Display for Vec3
{
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>)->std::result::Result<(), std::fmt::Error>
    {
        print!("(x: {}, y: {}, z: {})", self.x, self.y, self.z);
        return Ok(());
    }
}

// Matrices (they are all column major)

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Mat4
{
    pub m: [[f32; 4]; 4]
}

impl Default for Mat4
{
    fn default() -> Self
    {
        return Self::IDENTITY;
    }
}

impl Mat4
{
    pub const IDENTITY: Self = Self
    {
        m: [
            [ 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ]
    };

    pub fn zeros() -> Self
    {
        return Mat4{ m: [
            [ 0.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 0.0 ]
        ]};
    }
}

impl std::ops::Mul<Mat4> for Mat4
{
    type Output = Mat4;

    #[inline]
    fn mul(self, rhs: Mat4)->Mat4
    {
        let mut res = Mat4::zeros();
        for i in 0..4
        {
            for j in 0..4
            {
                for k in 0..4
                {
                    res.m[j][i] += self.m[k][i] * rhs.m[j][k];
                }
            }
        }

        return res;
    }
}

impl std::ops::Mul<Vec4> for Mat4
{
    type Output = Vec4;

    #[inline]
    fn mul(self, rhs: Vec4) -> Vec4
    {
        let res_x = self.m[0][0]*rhs.x + self.m[1][0]*rhs.y + self.m[2][0]*rhs.z + self.m[3][0]*rhs.w;
        let res_y = self.m[0][1]*rhs.x + self.m[1][1]*rhs.y + self.m[2][1]*rhs.z + self.m[3][1]*rhs.w;
        let res_z = self.m[0][2]*rhs.x + self.m[1][2]*rhs.y + self.m[2][2]*rhs.z + self.m[3][2]*rhs.w;
        let res_w = self.m[0][3]*rhs.x + self.m[1][3]*rhs.y + self.m[2][3]*rhs.z + self.m[3][3]*rhs.w;
        return Vec4 { x: res_x, y: res_y, z: res_z, w: res_w };
    }
}

impl std::ops::Mul<Vec3> for Mat4
{
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3
    {
        let res_x = self.m[0][0]*rhs.x + self.m[1][0]*rhs.y + self.m[2][0]*rhs.z + self.m[3][0]*1.0;
        let res_y = self.m[0][1]*rhs.x + self.m[1][1]*rhs.y + self.m[2][1]*rhs.z + self.m[3][1]*1.0;
        let res_z = self.m[0][2]*rhs.x + self.m[1][2]*rhs.y + self.m[2][2]*rhs.z + self.m[3][2]*1.0;
        let res_w = self.m[0][3]*rhs.x + self.m[1][3]*rhs.y + self.m[2][3]*rhs.z + self.m[3][3]*1.0;
        return Vec3 { x: res_x / res_w, y: res_y / res_w, z: res_z / res_w };
    }
}

/// The naming follows this convention: MatRxC, where R
/// is the number of rows, and C is the number of columns.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Mat3x4
{
    pub m: [[f32; 3]; 4]
}

impl Default for Mat3x4
{
    fn default() -> Self
    {
        return Self {
            m: [
                [ 1.0, 0.0, 0.0 ],
                [ 0.0, 1.0, 0.0 ],
                [ 0.0, 0.0, 1.0 ],
                [ 0.0, 0.0, 0.0 ]
            ]
        };
    }
}

impl Mat3x4
{
    pub const IDENTITY: Self = Self
    {
        m: [
            [ 1.0, 0.0, 0.0 ],
            [ 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 1.0 ],
            [ 0.0, 0.0, 0.0 ]
        ]
    };

    #[inline]
    pub fn zeros() -> Self
    {
        return Mat3x4 { m: [
            [ 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 0.0 ]
        ]};
    }

    #[inline]
    pub fn transpose(&self) -> Mat4x3
    {
        return Mat4x3 {
            m: [
                [ self.m[0][0], self.m[1][0], self.m[2][0], self.m[3][0] ],
                [ self.m[0][1], self.m[1][1], self.m[2][1], self.m[3][1] ],
                [ self.m[0][2], self.m[1][2], self.m[2][2], self.m[3][2] ],
            ]
        };
    }

    #[inline]
    pub fn to_mat4(&self) -> Mat4
    {
        return Mat4 {
            m: [
                [ self.m[0][0], self.m[0][1], self.m[0][2], 0.0 ],
                [ self.m[1][0], self.m[1][1], self.m[1][2], 0.0 ],
                [ self.m[2][0], self.m[2][1], self.m[2][2], 0.0 ],
                [ self.m[3][0], self.m[3][1], self.m[3][2], 1.0 ],
            ]
        };
    }

    // TODO: @speed
    #[inline]
    pub fn inverse(&self) -> Mat3x4
    {
        let mat4 = self.to_mat4();
        let inv = mat4_inverse(mat4);
        return Mat3x4 {
            m: [
                [ inv.m[0][0], inv.m[0][1], inv.m[0][2] ],
                [ inv.m[1][0], inv.m[1][1], inv.m[1][2] ],
                [ inv.m[2][0], inv.m[2][1], inv.m[2][2] ],
                [ inv.m[3][0], inv.m[3][1], inv.m[3][2] ],
            ]
        };
    }
}

impl std::ops::Mul<Vec3> for Mat3x4
{
    type Output = Vec3;

    #[inline]
    fn mul(self, rhs: Vec3) -> Vec3
    {
        let res_x = self.m[0][0]*rhs.x + self.m[1][0]*rhs.y + self.m[2][0]*rhs.z + self.m[3][0]*1.0;
        let res_y = self.m[0][1]*rhs.x + self.m[1][1]*rhs.y + self.m[2][1]*rhs.z + self.m[3][1]*1.0;
        let res_z = self.m[0][2]*rhs.x + self.m[1][2]*rhs.y + self.m[2][2]*rhs.z + self.m[3][2]*1.0;
        return Vec3 { x: res_x, y: res_y, z: res_z };
    }
}

impl std::ops::Mul<Mat3x4> for Mat3x4
{
    type Output = Mat3x4;

    #[inline]
    fn mul(self, rhs: Mat3x4) -> Mat3x4
    {
        let mut res = Mat3x4::zeros();
        let rhs_mat4 = rhs.to_mat4();
        for i in 0..3
        {
            for j in 0..4
            {
                for k in 0..4 {
                    res.m[j][i] += self.m[k][i] * rhs_mat4.m[j][k];
                }
            }
        }

        return res;
    }
}

/// The naming follows this convention: MatRxC, where R
/// is the number of rows, and C is the number of columns.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Mat4x3
{
    pub m: [[f32; 4]; 3]
}

impl Mat4x3
{
    #[inline]
    pub fn transpose(&self) -> Mat3x4
    {
        return Mat3x4 {
            m: [
                [ self.m[0][0], self.m[1][0], self.m[2][0] ],
                [ self.m[0][1], self.m[1][1], self.m[2][1] ],
                [ self.m[0][2], self.m[1][2], self.m[2][2] ],
                [ self.m[0][3], self.m[1][3], self.m[2][3] ],
            ]
        };
    }
}

impl Default for Mat4x3
{
    fn default() -> Self
    {
        return Self {
            m: [
                [ 1.0, 0.0, 0.0, 0.0 ],
                [ 0.0, 1.0, 0.0, 0.0 ],
                [ 0.0, 0.0, 1.0, 0.0 ],
            ]
        };
    }
}

pub fn mat4_inverse(m: Mat4) -> Mat4
{
    let s0 = m.m[0][0] * m.m[1][1] - m.m[1][0] * m.m[0][1];
    let s1 = m.m[0][0] * m.m[1][2] - m.m[1][0] * m.m[0][2];
    let s2 = m.m[0][0] * m.m[1][3] - m.m[1][0] * m.m[0][3];
    let s3 = m.m[0][1] * m.m[1][2] - m.m[1][1] * m.m[0][2];
    let s4 = m.m[0][1] * m.m[1][3] - m.m[1][1] * m.m[0][3];
    let s5 = m.m[0][2] * m.m[1][3] - m.m[1][2] * m.m[0][3];
    let c5 = m.m[2][2] * m.m[3][3] - m.m[3][2] * m.m[2][3];
    let c4 = m.m[2][1] * m.m[3][3] - m.m[3][1] * m.m[2][3];
    let c3 = m.m[2][1] * m.m[3][2] - m.m[3][1] * m.m[2][2];
    let c2 = m.m[2][0] * m.m[3][3] - m.m[3][0] * m.m[2][3];
    let c1 = m.m[2][0] * m.m[3][2] - m.m[3][0] * m.m[2][2];
    let c0 = m.m[2][0] * m.m[3][1] - m.m[3][0] * m.m[2][1];

    // Should check for 0 determinant
    let invdet = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

    let mut b = Mat4::zeros();
    b.m[0][0] = ( m.m[1][1] * c5 - m.m[1][2] * c4 + m.m[1][3] * c3) * invdet;
    b.m[0][1] = (-m.m[0][1] * c5 + m.m[0][2] * c4 - m.m[0][3] * c3) * invdet;
    b.m[0][2] = ( m.m[3][1] * s5 - m.m[3][2] * s4 + m.m[3][3] * s3) * invdet;
    b.m[0][3] = (-m.m[2][1] * s5 + m.m[2][2] * s4 - m.m[2][3] * s3) * invdet;
    b.m[1][0] = (-m.m[1][0] * c5 + m.m[1][2] * c2 - m.m[1][3] * c1) * invdet;
    b.m[1][1] = ( m.m[0][0] * c5 - m.m[0][2] * c2 + m.m[0][3] * c1) * invdet;
    b.m[1][2] = (-m.m[3][0] * s5 + m.m[3][2] * s2 - m.m[3][3] * s1) * invdet;
    b.m[1][3] = ( m.m[2][0] * s5 - m.m[2][2] * s2 + m.m[2][3] * s1) * invdet;
    b.m[2][0] = ( m.m[1][0] * c4 - m.m[1][1] * c2 + m.m[1][3] * c0) * invdet;
    b.m[2][1] = (-m.m[0][0] * c4 + m.m[0][1] * c2 - m.m[0][3] * c0) * invdet;
    b.m[2][2] = ( m.m[3][0] * s4 - m.m[3][1] * s2 + m.m[3][3] * s0) * invdet;
    b.m[2][3] = (-m.m[2][0] * s4 + m.m[2][1] * s2 - m.m[2][3] * s0) * invdet;
    b.m[3][0] = (-m.m[1][0] * c3 + m.m[1][1] * c1 - m.m[1][2] * c0) * invdet;
    b.m[3][1] = ( m.m[0][0] * c3 - m.m[0][1] * c1 + m.m[0][2] * c0) * invdet;
    b.m[3][2] = (-m.m[3][0] * s3 + m.m[3][1] * s1 - m.m[3][2] * s0) * invdet;
    b.m[3][3] = ( m.m[2][0] * s3 - m.m[2][1] * s1 + m.m[2][2] * s0) * invdet;
    return b;
}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Quat
{
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32
}

impl Default for Quat
{
    #[inline(always)]
    fn default()->Quat
    {
        return Quat::IDENTITY;
    }
}

impl std::ops::Mul<Quat> for Quat
{
    type Output = Quat;

    #[inline]
    fn mul(self, rhs: Quat)->Quat
    {
        return Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y + self.y * rhs.w + self.z * rhs.x - self.x * rhs.z,
            z: self.w * rhs.z + self.z * rhs.w + self.x * rhs.y - self.y * rhs.x,
        }
    }
}

impl std::ops::MulAssign<Quat> for Quat
{
    #[inline]
    fn mul_assign(&mut self, rhs: Quat)
    {
        self.w = self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z;
        self.x = self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y;
        self.y = self.w * rhs.y + self.y * rhs.w + self.z * rhs.x - self.x * rhs.z;
        self.z = self.w * rhs.z + self.z * rhs.w + self.x * rhs.y - self.y * rhs.x;
    }
}

pub fn vec3_quat_mul(q: Quat, v: Vec3) -> Vec3
{
    let num   = q.x * 2.0;
    let num2  = q.y * 2.0;
    let num3  = q.z * 2.0;
    let num4  = q.x * num;
    let num5  = q.y * num2;
    let num6  = q.z * num3;
    let num7  = q.x * num2;
    let num8  = q.x * num3;
    let num9  = q.y * num3;
    let num10 = q.w * num;
    let num11 = q.w * num2;
    let num12 = q.w * num3;

    return Vec3 {
        x: (1.0 - (num5 + num6)) * v.x + (num7 - num12) * v.y + (num8 + num11) * v.z,
        y: (num7 + num12) * v.x + (1.0 - (num4 + num6)) * v.y + (num9 - num10) * v.z,
        z: (num8 - num11) * v.x + (num9 + num10) * v.y + (1.0 - (num4 + num5)) * v.z,
    };
}

impl Quat
{
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    #[inline(always)]
    pub fn xyz(&self)->Vec3 { return Vec3 { x: self.x, y: self.y, z: self.z }; }
}

impl std::fmt::Display for Quat
{
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>)->std::result::Result<(), std::fmt::Error>
    {
        print!("(x: {}, y: {}, z: {}, w: {})", self.x, self.y, self.z, self.w);
        return Ok(());
    }
}

pub fn rotate_vec3_with_quat(q: Quat, v: Vec3)->Vec3
{
    let num   = q.x * 2.0;
    let num2  = q.y * 2.0;
    let num3  = q.z * 2.0;
    let num4  = q.x * num;
    let num5  = q.y * num2;
    let num6  = q.z * num3;
    let num7  = q.x * num2;
    let num8  = q.x * num3;
    let num9  = q.y * num3;
    let num10 = q.w * num;
    let num11 = q.w * num2;
    let num12 = q.w * num3;
    return Vec3 {
        x: (1.0 - (num5 + num6)) * v.x + (num7 - num12) * v.y + (num8 + num11) * v.z,
        y: (num7 + num12) * v.x + (1.0 - (num4 + num6)) * v.y + (num9 - num10) * v.z,
        z: (num8 - num11) * v.x + (num9 + num10) * v.y + (1.0 - (num4 + num5)) * v.z
    };
}

// Applies b first, a second
#[inline]
pub fn quat_mul(a: Quat, b: Quat)->Quat
{
    return Quat {
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y + a.y * b.w + a.z * b.x - a.x * b.z,
        z: a.w * b.z + a.z * b.w + a.x * b.y - a.y * b.x
    };
}

#[inline]
pub fn dot_vec3(v1: Vec3, v2: Vec3)->f32
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

#[inline]
pub fn length_vec3(v: Vec3)->f32
{
    return f32::sqrt(dot_vec3(v, v));
}

#[inline]
pub fn cross_vec3(v1: Vec3, v2: Vec3)->Vec3
{
    return Vec3 {
        x: v1.y*v2.z - v2.y*v1.z,
        y: v1.z*v2.x - v2.z*v1.x,
        z: v1.x*v2.y - v2.x*v1.y,
    };
}

#[inline]
pub fn magnitude_vec3(v: Vec3)->f32
{
    return (v.x*v.x + v.y*v.y + v.z*v.z).sqrt();
}

#[inline]
pub fn normalize_quat(q: Quat)->Quat
{
    let mag = magnitude_quat(q);
    return Quat { w: q.w / mag, x: q.x / mag, y: q.y / mag, z: q.z / mag };
}

#[inline]
pub fn magnitude_quat(q: Quat)->f32
{
    return (q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w).sqrt();
}

#[inline]
pub fn inverse(q: Quat)->Quat
{
    let length_sqr: f32 = q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w;
    if length_sqr == 0.0 { return q; }

    let i: f32 = 1.0 / length_sqr;
    return Quat { w: q.w*i, x: q.x*-i, y: q.y*-i, z: q.z*-i };
}

pub fn slerp(q1: Quat, q2: Quat, t: f32)->Quat
{
    let length_sqr1: f32 = q1.x*q1.x + q1.y*q1.y + q1.z*q1.z + q1.w*q1.w;
    let length_sqr2: f32 = q2.x*q2.x + q2.y*q2.y + q2.z*q2.z + q2.w*q2.w;
    let mut q2 = q2;

    if length_sqr1 == 0.0
    {
        if length_sqr2 == 0.0
        {
            return Quat::IDENTITY;
        }
        return q2;
    }
    else if length_sqr2 == 0.0
    {
        return q1;
    }

    let mut cos_half_angle: f32 = q1.w * q2.w + dot_vec3(q1.xyz(), q2.xyz());
    if cos_half_angle >= 1.0 || cos_half_angle <= -1.0
    {
        return q1;
    }

    if cos_half_angle < 0.0
    {
        q2 = Quat { w: -q2.w, x: -q2.x, y: -q2.y, z: -q2.w };
        cos_half_angle = -cos_half_angle;
    }

    let blend_a;
    let blend_b;
    if cos_half_angle < 0.99
    {
        // Do proper slerp for big angles
        let half_angle = cos_half_angle.acos();
        let sin_half_angle = half_angle.sin();
        let one_over_sin_half_angle = 1.0 / sin_half_angle;
        blend_a = (half_angle * (1.0 - t)).sin() * one_over_sin_half_angle;
        blend_b = (half_angle * t).sin() * one_over_sin_half_angle;
    }
    else
    {
        // Do lerp if angle is really small
        blend_a = 1.0 - t;
        blend_b = t;
    }

    let result = Quat { w: blend_a*q1.w + blend_b*q2.w,
                        x: blend_a*q1.x + blend_b*q2.x,
                        y: blend_a*q1.y + blend_b*q2.y,
                        z: blend_a*q1.z + blend_b*q2.z };
    let length_sqr: f32 = q1.x*q1.x + q1.y*q1.y + q1.z*q1.z + q1.w*q1.w;
    if length_sqr > 0.0 { return normalize_quat(result); }

    return Quat::IDENTITY;
}

// https://stackoverflow.com/questions/46156903/how-to-lerp-between-two-quaternions
pub fn lerp_quat(q1: Quat, q2: Quat, t: f32)->Quat
{
    let dot_q: f32 = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w;
    let mut q2 = q2;

    if dot_q < 0.0
    {
        // Negate q2
        q2.x = -q2.x;
        q2.y = -q2.y;
        q2.z = -q2.z;
        q2.w = -q2.w;
    }

    let mut res = Quat::IDENTITY;
    res.x = q1.x - t*(q1.x - q2.x);
    res.y = q1.y - t*(q1.y - q2.y);
    res.z = q1.z - t*(q1.z - q2.z);
    res.w = q1.w - t*(q1.w - q2.w);
    return normalize_quat(res);
}

pub fn angle_axis(axis: Vec3, angle: f32)->Quat
{
    if dot_vec3(axis, axis) == 0.0
    {
        return Quat::IDENTITY;
    }

    let mut axis = axis;
    let mut angle = angle;

    angle *= 0.5;
    axis = normalize_vec3(axis);
    axis *= angle.sin();

    return Quat { w: angle.cos(), x: axis.x, y: axis.y, z: axis.z };
}

pub fn angle_diff_quat(a: Quat, b: Quat)->f32
{
    let f: f32 = dot_vec3(Vec3 { x: a.x, y: a.y, z: a.z }, Vec3 { x: b.x, y: b.y, z: b.z }) + a.w*b.w;
    return f.abs().min(1.0).acos() * 2.0;
}

pub fn rotate_torwards_quat(current: Quat, target: Quat, delta: f32)->Quat
{
    let angle: f32 = angle_diff_quat(current, target);
    if angle == 0.0 { return target; }

    let t: f32 = (delta / angle).min(1.0);
    return slerp(current, target, t);
}

pub fn rotation_matrix(q: Quat)->Mat3x4
{
    let x  = q.x * 2.0; let y  = q.y * 2.0; let z  = q.z * 2.0;
    let xx = q.x * x;   let yy = q.y * y;   let zz = q.z * z;
    let xy = q.x * y;   let xz = q.x * z;   let yz = q.y * z;
    let wx = q.w * x;   let wy = q.w * y;   let wz = q.w * z;

    // Calculate 3x3 matrix from orthonormal basis
    let res = Mat3x4 {
        m: [
            [ 1.0 - (yy + zz), xy + wz, xz - wy ],
            [ xy - wz, 1.0 - (xx + zz), yz + wx ],
            [ xz + wy, yz - wx, 1.0 - (xx + yy) ],
            [ 0.0, 0.0, 0.0 ],
        ]
    };
    return res;
}

pub fn scale_matrix(scale: Vec3)->Mat3x4
{
    let mut res = Mat3x4::zeros();
    res.m[0][0] = scale.x;
    res.m[1][1] = scale.y;
    res.m[2][2] = scale.z;
    return res;
}

pub fn position_matrix(pos: Vec3)->Mat3x4
{
    let mut res = Mat3x4::IDENTITY;
    res.m[3][0] = pos.x;
    res.m[3][1] = pos.y;
    res.m[3][2] = pos.z;
    return res;
}

pub fn xform_to_matrix(pos: Vec3, rot: Quat, scale: Vec3) -> Mat3x4
{
    return position_matrix(pos) * rotation_matrix(rot) * scale_matrix(scale);
}

pub fn matrix_to_xform(mat: Mat4) -> (Vec3, Quat, Vec3)
{
    let translation = Vec3 {
        x: mat.m[3][0],
        y: mat.m[3][1],
        z: mat.m[3][2],
    };

    let mut col0 = Vec3 { x: mat.m[0][0], y: mat.m[0][1], z: mat.m[0][2] };
    let mut col1 = Vec3 { x: mat.m[1][0], y: mat.m[1][1], z: mat.m[1][2] };
    let mut col2 = Vec3 { x: mat.m[2][0], y: mat.m[2][1], z: mat.m[2][2] };

    let sx = (col0.x*col0.x + col0.y*col0.y + col0.z*col0.z).sqrt();
    let sy = (col1.x*col1.x + col1.y*col1.y + col1.z*col1.z).sqrt();
    let sz = (col2.x*col2.x + col2.y*col2.y + col2.z*col2.z).sqrt();
    let scale = Vec3 { x: sx, y: sy, z: sz };

    if sx > 0.0001 { col0.x /= sx; col0.y /= sx; col0.z /= sx; }
    if sy > 0.0001 { col1.x /= sy; col1.y /= sy; col1.z /= sy; }
    if sz > 0.0001 { col2.x /= sz; col2.y /= sz; col2.z /= sz; }

    let rot = [
        [col0.x, col1.x, col2.x],
        [col0.y, col1.y, col2.y],
        [col0.z, col1.z, col2.z],
    ];

    let trace = rot[0][0] + rot[1][1] + rot[2][2];
    let quat = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        Quat {
            w: 0.25 * s,
            x: (rot[2][1] - rot[1][2]) / s,
            y: (rot[0][2] - rot[2][0]) / s,
            z: (rot[1][0] - rot[0][1]) / s,
        }
    } else if rot[0][0] > rot[1][1] && rot[0][0] > rot[2][2] {
        let s = (1.0 + rot[0][0] - rot[1][1] - rot[2][2]).sqrt() * 2.0;
        Quat {
            w: (rot[2][1] - rot[1][2]) / s,
            x: 0.25 * s,
            y: (rot[0][1] + rot[1][0]) / s,
            z: (rot[0][2] + rot[2][0]) / s,
        }
    } else if rot[1][1] > rot[2][2] {
        let s = (1.0 + rot[1][1] - rot[0][0] - rot[2][2]).sqrt() * 2.0;
        Quat {
            w: (rot[0][2] - rot[2][0]) / s,
            x: (rot[0][1] + rot[1][0]) / s,
            y: 0.25 * s,
            z: (rot[1][2] + rot[2][1]) / s,
        }
    } else {
        let s = (1.0 + rot[2][2] - rot[0][0] - rot[1][1]).sqrt() * 2.0;
        Quat {
            w: (rot[1][0] - rot[0][1]) / s,
            x: (rot[0][2] + rot[2][0]) / s,
            y: (rot[1][2] + rot[2][1]) / s,
            z: 0.25 * s,
        }
    };

    return (translation, quat, scale);
}

////////
// Miscellaneous

// Useful for passing buffers to the GPU
pub fn to_u8_slice<T>(slice: &[T])->&[u8]
{
    let buf_size = slice.len() * std::mem::size_of::<T>();
    return unsafe {
        std::slice::from_raw_parts(slice.as_ptr() as *const _ as *const u8, buf_size)
    };
}

pub fn to_u64_slice<T>(slice: &[T])->&[u64]
{
    let buf_size = slice.len() * std::mem::size_of::<T>();
    return unsafe {
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

pub fn grow_aabb_to_include_vert(aabb: &mut Aabb, to_include: Vec3)
{
    aabb.min.x = aabb.min.x.min(to_include.x);
    aabb.max.x = aabb.max.x.max(to_include.x);
    aabb.min.y = aabb.min.y.min(to_include.y);
    aabb.max.y = aabb.max.y.max(to_include.y);
    aabb.min.z = aabb.min.z.min(to_include.z);
    aabb.max.z = aabb.max.z.max(to_include.z);
}

pub fn transform_aabb(min: Vec3, max: Vec3, transform: Mat3x4) -> Aabb
{
    let verts = [
        Vec3 { x: min.x, y: min.y, z: min.z },
        Vec3 { x: min.x, y: min.y, z: max.z },
        Vec3 { x: min.x, y: max.y, z: min.z },
        Vec3 { x: min.x, y: max.y, z: max.z },
        Vec3 { x: max.x, y: min.y, z: min.z },
        Vec3 { x: max.x, y: min.y, z: max.z },
        Vec3 { x: max.x, y: max.y, z: min.z },
        Vec3 { x: max.x, y: max.y, z: max.z },
    ];

    let mut res = Aabb::neutral();
    for vert in verts
    {
        let vert_trans = transform * vert;
        grow_aabb_to_include_vert(&mut res, vert_trans);
    }

    return res;
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

#[inline]
pub fn compute_tri_centroid(t0: Vec3, t1: Vec3, t2: Vec3)->Vec3
{
    return (t0 + t1 + t2) / 3.0;
}

#[inline]
pub fn is_point_in_aabb(v: Vec3, aabb_min: Vec3, aabb_max: Vec3)->bool
{
    return v.x >= aabb_min.x && v.x <= aabb_max.x &&
           v.y >= aabb_min.y && v.y <= aabb_max.y &&
           v.z >= aabb_min.z && v.z <= aabb_max.z;
}

////////
// Printing utils

pub fn print_type<T>(_: &T)
{
    println!("{}", std::any::type_name::<T>());
}