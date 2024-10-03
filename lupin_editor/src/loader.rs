
use tobj::*;
use std::ptr::NonNull;

use lupin::base::*;
use lupin::renderer::*;
use lupin::wgpu_utils::*;

#[derive(Default)]
pub struct LoadingTimes
{
    pub parsing: f32,
    pub bvh_build: f32
}

pub fn load_scene_custom_format(path: &str)
{

}

pub fn load_scene_obj(device: wgpu::Device, queue: wgpu::Queue, path: &str)->(Scene, LoadingTimes)
{
    let mut loading_times: LoadingTimes = Default::default();

    let timer_start = std::time::Instant::now();
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;

    loading_times.parsing = time;

    assert!(scene.is_ok());
    let (mut models, materials) = scene.expect("Failed to load OBJ file");

    let mut verts_pos: Vec<f32> = Vec::new();

    if models.len() > 0
    {
        let mesh = &mut models[0].mesh;

        // Construct the buffer to send to GPU. Include an extra float
        // for 16-byte padding (which seems to be required in WebGPU).
        
        verts_pos.reserve_exact(mesh.positions.len() + mesh.positions.len() / 3);
        for i in (0..mesh.positions.len()).step_by(3)
        {
            verts_pos.push(mesh.positions[i + 0]);
            verts_pos.push(mesh.positions[i + 1]);
            verts_pos.push(mesh.positions[i + 2]);
            verts_pos.push(0.0);
        }

        let timer_start = std::time::Instant::now();
        let bvh_buf = construct_bvh(&device, &queue, verts_pos.as_slice(), &mut mesh.indices);
        let time = timer_start.elapsed().as_micros() as f32 / 1_000.0;
        loading_times.bvh_build = time;

        let verts_pos_buf = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts_pos) });
        let indices_buf = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&mesh.indices) });
        let mut verts: Vec<Vertex> = Vec::new();
        verts.reserve_exact(mesh.positions.len() / 3);
        for vert_idx in 0..(mesh.positions.len() / 3)
        {
            let mut normal = Vec3::default();
            if mesh.normals.len() > 0
            {
                normal.x = mesh.normals[vert_idx*3+0];
                normal.y = mesh.normals[vert_idx*3+1];
                normal.z = mesh.normals[vert_idx*3+2];
                normal = normalize_vec3(normal);
            };

            let mut tex_coords = Vec2::default();
            if mesh.texcoords.len() > 0
            {
                tex_coords.x = mesh.texcoords[vert_idx*2+0];
                tex_coords.y = mesh.texcoords[vert_idx*2+1];
                tex_coords = normalize_vec2(tex_coords);
            };

            let vert = Vertex { normal, padding0: 0.0, tex_coords, padding1: 0.0, padding2: 0.0 };

            verts.push(vert);
        }

        let verts_buf = upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts) });

        return (Scene
        {
            verts_pos: verts_pos_buf,
            indices:   indices_buf,
            bvh_nodes: bvh_buf,
            verts:     verts_buf
        }, loading_times);
    }

    return (Scene
    {
        verts_pos: create_empty_storage_buffer(&device, &queue),
        indices:   create_empty_storage_buffer(&device, &queue),
        bvh_nodes: create_empty_storage_buffer(&device, &queue),
        verts:     create_empty_storage_buffer(&device, &queue)
    }, loading_times);
}
