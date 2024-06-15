
use crate::base::*;

use tobj::*;
use std::ptr::NonNull;
use crate::renderer_wgpu::*;

pub fn load_scene_custom_format(path: &str)
{

}

pub fn load_scene_obj(path: &str, renderer: &mut Renderer)->Scene
{
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    assert!(scene.is_ok());

    let (models, materials) = scene.expect("Failed to load OBJ file");
    //let materials = materials.expect("Failed to load MTL file");

    println!("Num models: {}", models.len());
    //println!("Num materials: {}", materials.len());

    if models.len() > 0
    {
        let mesh = &models[0].mesh;

        // This includes padding
        let mut verts_pos: Vec<f32> = Vec::new();
        verts_pos.reserve_exact(mesh.positions.len() + mesh.positions.len() / 3);
        for i in (0..mesh.positions.len()).step_by(3)
        {
            verts_pos.push(mesh.positions[i + 0]);
            verts_pos.push(mesh.positions[i + 1]);
            verts_pos.push(mesh.positions[i + 2]);
            verts_pos.push(0.0);  // Padding for GPU memory
        }

        let verts_pos_buf = renderer.upload_buffer(to_u8_slice(&verts_pos));
        let indices_buf = renderer.upload_buffer(to_u8_slice(&mesh.indices));

        return Scene
        {
            verts: verts_pos_buf,
            indices: indices_buf
        }
    }

    return Scene
    {
        verts: renderer.empty_buffer(),
        indices: renderer.empty_buffer()
    }
}

pub fn unload_scene(scene: &mut Scene)
{

}

pub fn load_image(path: &str, required_channels: i32)
{
    
}
