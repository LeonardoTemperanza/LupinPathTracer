
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

// Useful constants
pub const DEG_TO_RAD: f32 = 0.017453292;
pub const RAD_TO_DEG: f32 = 57.29578049;

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

impl Vec3
{
    pub const LEFT:     Vec3 = Vec3 { x: -1.0, y: 0.0,  z: 0.0  };
    pub const RIGHT:    Vec3 = Vec3 { x: 1.0,  y: 0.0,  z: 0.0  };
    pub const UP:       Vec3 = Vec3 { x: 0.0,  y: 1.0,  z: 0.0  };
    pub const DOWN:     Vec3 = Vec3 { x: 0.0,  y: -1.0, z: 0.0  };
    pub const FORWARD:  Vec3 = Vec3 { x: 0.0,  y: 0.0,  z: 1.0  };
    pub const BACKWARD: Vec3 = Vec3 { x: 0.0,  y: 0.0,  z: -1.0 };
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

impl Into<lp::Aabb> for Aabb
{
    fn into(self) -> lp::Aabb
    {
        return lp::Aabb { min: self.min.into(), max: self.max.into() }
    }
}

#[inline]
pub fn lerp_f32(a: f32, b: f32, t: f32) -> f32
{
    return a + (b - a) * t;
}

#[inline]
pub fn square_f32(n: f32) -> f32
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
    fn sub(self, other: Self) -> Self::Output
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

#[derive(Debug, Default, Clone, Copy)]
#[repr(C)]
pub struct Mat4
{
    pub m: [[f32; 4]; 4]
}

impl Mat4
{
    pub const IDENTITY: Mat4 = Mat4
    {
        m:
        [
            [ 1.0, 0.0, 0.0, 0.0 ],
            [ 0.0, 1.0, 0.0, 0.0 ],
            [ 0.0, 0.0, 1.0, 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ]
        ]
    };
}

impl std::ops::Mul<Mat4> for Mat4
{
    type Output = Mat4;

    #[inline]
    fn mul(self, rhs: Mat4) -> Mat4
    {
        let mut res = Mat4::default();
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
    fn mul(self, rhs: Quat) -> Quat
    {
        return Quat {
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

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Transform
{
    pub pos: Vec3,
    pub rot: Quat,
    pub scale: Vec3
}

impl Default for Transform
{
    fn default()->Transform
    {
        return Transform
        {
            pos: Default::default(),
            rot: Default::default(),
            scale: Vec3 { x: 1.0, y: 1.0, z: 1.0 }
        };
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

#[inline]
pub fn dot_vec3(v1: Vec3, v2: Vec3)->f32
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

#[inline]
pub fn dot2_vec3(v: Vec3)->f32
{
    return v.x*v.x + v.y*v.y + v.z*v.z;
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

    // res = q1 + t(q2 - q1)  -->  res = q1 - t(q1 - q2)
    // The latter is slightly better on x64
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

pub fn rotation_matrix(q: Quat)->Mat4
{
    let x  = q.x * 2.0; let y  = q.y * 2.0; let z  = q.z * 2.0;
    let xx = q.x * x;   let yy = q.y * y;   let zz = q.z * z;
    let xy = q.x * y;   let xz = q.x * z;   let yz = q.y * z;
    let wx = q.w * x;   let wy = q.w * y;   let wz = q.w * z;

    // Calculate 3x3 matrix from orthonormal basis
    let res = Mat4
    {
        m:
        [
            [ 1.0 - (yy + zz), xy + wz, xz - wy, 0.0 ],
            [ xy - wz, 1.0 - (xx + zz), yz + wx, 0.0 ],
            [ xz + wy, yz - wx, 1.0 - (xx + yy), 0.0 ],
            [ 0.0, 0.0, 0.0, 1.0 ],
        ]
    };
    return res;
}

pub fn scale_matrix(scale: Vec3)->Mat4
{
    let mut res = Mat4::default();
    res.m[0][0] = scale.x;
    res.m[1][1] = scale.y;
    res.m[2][2] = scale.z;
    res.m[3][3] = 1.0;
    return res;
}

pub fn position_matrix(pos: Vec3)->Mat4
{
    let mut res = Mat4::default();
    res.m[0][0] = 1.0;
    res.m[1][1] = 1.0;
    res.m[2][2] = 1.0;
    res.m[3][3] = 1.0;
    res.m[3][0] = pos.x;
    res.m[3][1] = pos.y;
    res.m[3][2] = pos.z;
    return res;
}

pub fn transform_to_matrix(transform: Transform)->Mat4
{
    return position_matrix(transform.pos) * rotation_matrix(transform.rot) * scale_matrix(transform.scale);
}

pub fn xform_to_matrix(pos: Vec3, rot: Quat, scale: Vec3) -> Mat4
{
    return position_matrix(pos) * rotation_matrix(rot) * scale_matrix(scale);
}

pub use lupin as lp;

// Example implementation of Into trait to convert
// from your own algebraic types into lupin's.
impl Into<lp::Vec3> for Vec3
{
    fn into(self) -> lp::Vec3
    {
        return lp::Vec3 { x: self.x, y: self.y, z: self.z }
    }
}

impl Into<lp::Vec2> for Vec2
{
    fn into(self) -> lp::Vec2
    {
        return lp::Vec2 { x: self.x, y: self.y }
    }
}

impl Into<lp::Mat4> for Mat4
{
    fn into(self) -> lp::Mat4
    {
        return lp::Mat4 { m: self.m }
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
pub unsafe fn to_u8_slice<T>(slice: &[T])->&[u8]
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

pub fn grow_aabb_to_include_vert(aabb: &mut Aabb, to_include: Vec3)
{
    aabb.min.x = aabb.min.x.min(to_include.x);
    aabb.max.x = aabb.max.x.max(to_include.x);
    aabb.min.y = aabb.min.x.min(to_include.y);
    aabb.max.y = aabb.max.x.max(to_include.y);
    aabb.min.z = aabb.min.x.min(to_include.z);
    aabb.max.z = aabb.max.x.max(to_include.z);
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
