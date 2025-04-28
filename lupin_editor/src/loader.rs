
use lupin as lp;

use crate::base::*;

pub fn load_scene_obj(device: &wgpu::Device, queue: &wgpu::Queue, path: &str)->lp::SceneDesc
{
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);

    assert!(scene.is_ok());
    let (mut models, _materials) = scene.expect("Failed to load OBJ file");

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

        let bvh_buf = lp::build_bvh(&device, &queue, verts_pos.as_slice(), &mut mesh.indices);

        let verts_pos_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts_pos) });
        let indices_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&mesh.indices) });
        let mut verts: Vec<lp::Vertex> = Vec::new();
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

            let vert = lp::Vertex { normal: normal.into(), padding0: 0.0, tex_coords: tex_coords.into(), padding1: 0.0, padding2: 0.0 };

            verts.push(vert);
        }

        let verts_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts) });

        return lp::SceneDesc
        {
            verts_pos: verts_pos_buf,
            indices:   indices_buf,
            bvh_nodes: bvh_buf,
            verts:     verts_buf
        };
    }

    return lp::SceneDesc
    {
        verts_pos: lp::create_empty_storage_buffer(&device, &queue),
        indices:   lp::create_empty_storage_buffer(&device, &queue),
        bvh_nodes: lp::create_empty_storage_buffer(&device, &queue),
        verts:     lp::create_empty_storage_buffer(&device, &queue)
    };
}
