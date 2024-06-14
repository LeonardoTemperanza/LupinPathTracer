
use crate::base::*;

use tobj::*;
use std::ptr::NonNull;
use crate::renderer_wgpu::*;

pub fn load_scene_custom_format(path: &str)
{

}

pub fn load_scene_obj(path: &str)->u32
{
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    assert!(scene.is_ok());

    let (models, materials) = scene.expect("Failed to load OBJ file");
    let materials = materials.expect("Failed to load MTL file");

    println!("Num models: {}", models.len());
    println!("Num materials: {}", models.len());

    for (i, m) in models.iter().enumerate()
    {
        let mesh = &m.mesh;

        println!("model[{}].name = \'{}\'", i, m.name);
        println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);
        println!("size of model[{}].face_arities: {}", i, mesh.face_arities.len());

        let mut next_face = 0;
        for f in 0..mesh.positions.len() / 3
        {
            let end = next_face + mesh.face_arities[f] as usize;
            let face_indices: Vec<_> = mesh.indices[next_face..end].iter().collect();
            println!("    face[{}] = {:?}", f, face_indices);
            next_face = end;
        }
    }

    for (i, m) in materials.iter().enumerate()
    {
        println!("material[{}].name = \'{}\'", i, m.name);

        if let Some(ambient) = m.ambient
        {

        }

        if let Some(diffuse) = m.diffuse
        {

        }

        if let Some(specular) = m.specular
        {

        }

        // ... Some other stuff
    }

    return 0;

//    return Scene
//    {
//
//    }
}

pub fn load_image(path: &str, required_channels: i32)
{
    
}
