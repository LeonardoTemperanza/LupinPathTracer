
use lupin as lp;
use lupin::wgpu as wgpu;

pub fn build_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> lp::Scene
{
    let mut verts_pos_array: Vec<Vec<lp::VertexPos>> = Default::default();
    let mut verts_array: Vec<Vec<lp::Vertex>> = Default::default();
    let mut indices_array: Vec<Vec<u32>> = Default::default();
    let mut bvh_nodes_array: Vec<Vec<lp::BvhNode>> = Default::default();
    let mut mesh_aabbs: Vec<lp::Aabb> = Default::default();
    let mut materials: Vec<lp::Material> = Default::default();
    let mut environments: Vec<lp::Environment> = Default::default();

    let mut textures: Vec<wgpu::Texture> = Default::default();
    let mut samplers: Vec<wgpu::Sampler> = Default::default();

    // Textures
    let white_tex = push_asset(&mut textures, lp::create_white_texture(device, queue));
    let bunny_color = push_asset(&mut textures, load_texture(device, queue, "bunny_color.png", false).unwrap());
    //let (env_map_cpu, env_map_gpu) = load_hdr_texture_and_keep_cpu_copy(device, queue, "poly_haven_studio_1k.hdr");
    let (env_map_cpu, env_map_gpu) = load_hdr_texture_and_keep_cpu_copy(device, queue, "sky.hdr");
    let env_map = push_asset(&mut textures, env_map_gpu);

    // Samplers
    let linear_sampler = push_asset(&mut samplers, lp::create_linear_sampler(device));

    // Meshes
    let bunny_mesh  = load_mesh_obj("stanford_bunny.obj", &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs);
    let quad_mesh   = load_mesh_obj("quad.obj",           &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs);
    let gazebo_mesh = load_mesh_obj("gazerbo.obj",        &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs);
    let dragon_80k_mesh = load_mesh_obj("Dragon_80K.obj", &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs);
    let ply_test = load_mesh_ply(std::path::Path::new("C:/work/LupinPathTracer/assets/yocto-scenes/coffee/shapes/shape22.ply"), &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs).unwrap();

    // Materials
    let bunny_matte = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Matte,            // Mat type
        lp::Vec4::new(1.0, 1.0, 1.0, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.01,                               // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        1,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let glossy = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Glossy,           // Mat type
        lp::Vec4::new(0.9, 0.2, 0.2, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.1,                               // Roughness
        0.0,                                // Metallic
        1.0,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let reflective = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Reflective,       // Mat type
        lp::Vec4::new(0.9, 0.2, 0.2, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.0,                                // Roughness
        0.0,                                // Metallic
        0.0,                                // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let transparent = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Transparent,      // Mat type
        lp::Vec4::new(0.1, 0.1, 1.0, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.0,                                // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let refractive = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Refractive,       // Mat type
        lp::Vec4::new(0.1, 0.1, 1.0, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.0,                                // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let brown_matte = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Matte,            // Mat type
        lp::Vec4::new(1.0, 1.0, 1.0, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.01,                                // Roughness
        0.0,                                // Metallic
        1.5,                                // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    // Rough reflective
    // 6
    let rough_reflective = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Reflective,       // Mat type
        lp::Vec4::new(0.9, 0.2, 0.2, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.2,                                // Roughness
        0.0,                                // Metallic
        1.5,                                // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let rough_glossy = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Glossy,           // Mat type
        lp::Vec4::new(0.9, 0.2, 0.2, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.1,                               // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let glft = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::GltfPbr,           // Mat type
        lp::Vec4::new(0.9, 0.2, 0.2, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.1,                                // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let rough_transparent = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Transparent,      // Mat type
        lp::Vec4::new(0.1, 0.1, 1.0, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.008,                               // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let rough_reflective = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Refractive,       // Mat type
        lp::Vec4::new(0.1, 0.1, 1.0, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.05,                                // Roughness
        0.0,                                // Metallic
        1.33,                               // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    let emissive = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Matte,       // Mat type
        lp::Vec4::new(0.0, 0.0, 0.0, 1.0),  // Color
        lp::Vec4::new(100.0, 100.0, 100.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.0,                                // Roughness
        0.0,                                // Metallic
        0.0,                                // ior
        0.0,                                // anisotropy
        0.0,                                // depth
        0,                                  // Color tex
        0,                                  // Emission tex
        0,                                  // Roughness tex
        0,                                  // Scattering tex
        0,                                  // Normal tex
    ));

    // Stress-test
    /*
    let mut instances = Vec::<lp::Instance>::default();
    for i in 0..120
    {
        for j in 0..120
        {
            let offset: f32 = 1.5;
            instances.push(lp::Instance {
                inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: offset * i as f32, y: 0.0, z: offset * j as f32 }, lp::Quat::default(), lp::Vec3::ones())),
                mesh_idx: 0,
                mat_idx: 0,
                padding0: 0.0, padding1: 0.0
            });
        }
    }
    */

    // Instances
    let instances = vec![
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: bunny_matte, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: -2.0, y: 0.0, z: 0.0 }, lp::angle_axis(lp::Vec3::RIGHT, 45.0 * 3.1415 / 180.0), lp::Vec3::ones())), mesh_idx: 0, mat_idx: glossy, padding0: 0.0, padding1: 0.0},
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: -2.0, y: 0.0, z: -2.0 }, lp::angle_axis(lp::Vec3::UP, 90.0 * 3.1415 / 180.0), lp::Vec3 { x: 0.2, y: 1.0, z: 1.0 })), mesh_idx: 0, mat_idx: glossy, padding0: 0.0, padding1: 0. },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 2.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 })), mesh_idx: 0, mat_idx: reflective, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 2.0, y: 0.0, z: -2.0 }, lp::Quat::default(), lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 })), mesh_idx: 0, mat_idx: transparent, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 4.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: refractive, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 4.0, y: 0.0, z: -2.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: rough_glossy, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 6.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: rough_reflective, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 6.0, y: 0.0, z: -2.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: rough_transparent, padding0: 0.0, padding1: 0.0 },
        //lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 8.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: emissive, padding0: 0.0, padding1: 0.0 },
        //lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 10.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: emissive, padding0: 0.0, padding1: 0.0 },
        // Floor
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: -0.01, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones() * 20.0)), mesh_idx: 1, mat_idx: brown_matte, padding0: 0.0, padding1: 0.0 },
        // Studio light
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: 10.0, z: -10.0 }, lp::angle_axis(lp::Vec3::RIGHT, -80.0 * 3.1415 / 180.0), lp::Vec3::ones())), mesh_idx: 1, mat_idx: emissive, padding0: 0.0, padding1: 0.0 },
        // Gazebo
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 30.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: gazebo_mesh, mat_idx: brown_matte, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 30.0, y: 0.0, z: -20.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: gazebo_mesh, mat_idx: brown_matte, padding0: 0.0, padding1: 0.0 },
        // Dragon
        //lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: 2.5, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones() * 10.0)), mesh_idx: dragon_80k_mesh, mat_idx: transparent, padding0: 0.0, padding1: 0.0 },
        // Ply test
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 30.0, y: 0.0, z: 0.0 }, lp::angle_axis(lp::Vec3::RIGHT, -80.0 * 3.1415 / 180.0), lp::Vec3::ones() * 20.0)), mesh_idx: ply_test, mat_idx: emissive, padding0: 0.0, padding1: 0.0 },
    ];

    let tlas_nodes = lp::build_tlas(instances.as_slice(), &mesh_aabbs);

    let env = push_asset(&mut environments, lp::Environment {
        emission: lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 },
        emission_tex_idx: env_map,
    });

    let env_map_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let scene_cpu = lp::SceneCPU {
        verts_pos_array,
        verts_array,
        indices_array,
        bvh_nodes_array,
        mesh_aabbs,
        tlas_nodes,
        instances,
        materials,
        environments,
    };

    lp::validate_scene(&scene_cpu, textures.len() as u32, samplers.len() as u32);

    return lp::upload_scene_to_gpu(device, queue, &scene_cpu, textures, samplers, &[env_map_cpu]);
}

// Useful for library testing more than anything else.
pub fn build_scene_empty(device: &wgpu::Device, queue: &wgpu::Queue) -> lp::Scene
{
    let scene_cpu = lp::SceneCPU {
        verts_pos_array: vec![],
        verts_array: vec![],
        indices_array: vec![],
        bvh_nodes_array: vec![],
        mesh_aabbs: vec![],
        tlas_nodes: vec![],
        instances: vec![],
        materials: vec![],
        environments: vec![],
    };

    lp::validate_scene(&scene_cpu, 0, 0);

    return lp::upload_scene_to_gpu(device, queue, &scene_cpu, vec![], vec![], &[]);
}

/*
pub fn build_scene_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue) -> lp::Scene
{
    let mut verts_pos_array: Vec<Vec<lp::VertexPos>> = Default::default();
    let mut verts_array: Vec<Vec<lp::Vertex>> = Default::default();
    let mut indices_array: Vec<Vec<u32>> = Default::default();
    let mut bvh_nodes_array: Vec<Vec<lp::BvhNode>> = Default::default();
    let mut mesh_aabbs: Vec<lp::Aabb> = Default::default();
    let mut materials: Vec<lp::Material> = Default::default();
    let mut environments: Vec<lp::Environment> = Default::default();

    let mut textures: Vec<wgpu::Texture> = Default::default();
    let mut samplers: Vec<wgpu::Sampler> = Default::default();

    let white_tex = push_asset(&mut textures, lp::create_white_texture(device, queue));
    let matte_white = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Matte,                  // Mat type
        lp::Vec4::new(1.0, 1.0, 1.0, 1.0),        // Color
        lp::Vec4::new(100.0, 100.0, 100.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),        // Scattering
        0.05,                                     // Roughness
        0.0,                                      // Metallic
        1.33,                                     // ior
        0.0,                                      // anisotropy
        0.0,                                      // depth
        0,                                        // Color tex
        0,                                        // Emission tex
        0,                                        // Roughness tex
        0,                                        // Scattering tex
        0,                                        // Normal tex
    ));

    let mut instances = vec![
    ];

    let tlas_nodes = lp::build_tlas(instances.as_slice(), &mesh_aabbs);

    let scene_cpu = lp::SceneCPU {
        verts_pos_array,
        verts_array,
        indices_array,
        bvh_nodes_array,
        mesh_aabbs,
        tlas_nodes,
        instances,
        materials,
        environments: vec![],
    };

    lp::validate_scene(&scene_cpu, textures.len() as u32, samplers.len() as u32);

    return lp::upload_scene_to_gpu(device, queue, &scene_cpu, textures, samplers, &[]);
}
*/

pub fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &str, hdr: bool) -> Result<wgpu::Texture, image::ImageError>
{
    use image::GenericImageView;

    let img = image::open(path)?;
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if hdr { wgpu::TextureFormat::Rgba16Float } else { wgpu::TextureFormat::Rgba8UnormSrgb },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[]
    });

    if hdr
    {
        let rgba_f32 = img.to_rgba32f();
        let rgba = rgba32f_to_rgba16f(&rgba_f32);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            lp::to_u8_slice(&rgba),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(8 * dimensions.0),
                rows_per_image: Some(dimensions.1)
            },
            size
        );
    }
    else
    {
        let rgba = img.to_rgba8();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            lp::to_u8_slice(&rgba.into_raw()),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * dimensions.0),
                rows_per_image: Some(dimensions.1)
            },
            size
        );
    }

    return Ok(texture);
}

pub fn load_hdr_texture_and_keep_cpu_copy(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> (lp::EnvMapInfo, wgpu::Texture)
{
    use image::GenericImageView;

    let img = image::open(path).expect("Failed to load image");
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[]
    });

    let rgba_f32 = img.to_rgba32f();
    let rgba = rgba32f_to_rgba16f(&rgba_f32);

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        lp::to_u8_slice(&rgba),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(8 * dimensions.0),
            rows_per_image: Some(dimensions.1)
        },
        size
    );

    let vec4_data: Vec<lp::Vec4> = rgba_f32
        .chunks_exact(4)
        .map(|chunk| lp::Vec4 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
            w: chunk[3],
        })
        .collect();

    return (lp::EnvMapInfo {
        data: vec4_data,
        width: rgba_f32.width(),
        height: rgba_f32.height(),
    }, texture);
}

fn rgba32f_to_rgba16f(image_rgba32f: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>) -> Vec<half::f16>
{
    return image_rgba32f.pixels()
        .flat_map(|p| p.0.iter().map(|&f| half::f16::from_f32(f)))
        .collect();
}

/*
fn rgba16f_to_imagebuf(image_rgba16f: &Vec<half::f16>) -> image::ImageBuffer<image::Rgba<f32>, Vec<f32>>
{
    return image_rgba16f.iter()
        .flat_map(|p| image::Rgba::<f32>(p.into()))
        .collect();
}
*/

fn push_asset<T>(vec: &mut Vec<T>, el: T) -> u32
{
    vec.push(el);
    return (vec.len() - 1) as u32;
}

fn load_mesh_obj<P: AsRef<std::path::Path>>(path: P, verts_pos: &mut Vec<Vec<lp::VertexPos>>, verts: &mut Vec<Vec<lp::Vertex>>, indices: &mut Vec<Vec<u32>>, bvh_nodes: &mut Vec<Vec<lp::BvhNode>>, aabbs: &mut Vec<lp::Aabb>) -> u32
{
    assert!(verts_pos.len() == verts.len() && verts.len() == indices.len() && indices.len() == aabbs.len() && aabbs.len() == bvh_nodes.len());

    let scene = tobj::load_obj(path.as_ref().to_str().unwrap(), &tobj::GPU_LOAD_OPTIONS);
    assert!(scene.is_ok());

    let (mut models, _materials) = scene.expect("Failed to load OBJ file");

    let mesh = &mut models[0].mesh;

    let mut aabb = lp::Aabb::neutral();
    let mut mesh_verts_pos = Vec::<lp::VertexPos>::with_capacity(mesh.positions.len());
    for i in (0..mesh.positions.len()).step_by(3)
    {
        let pos = lp::Vec3 { x: mesh.positions[i + 0], y: mesh.positions[i + 1], z: mesh.positions[i + 2] };
        mesh_verts_pos.push(lp::VertexPos { v: lp::Vec3 { x: pos.x, y: pos.y, z: pos.z }, _padding: 0.0 });
        lp::grow_aabb_to_include_vert(&mut aabb, pos);
    }

    let bvh = lp::build_bvh(mesh_verts_pos.as_slice(), &mut mesh.indices);

    let mut mesh_verts = Vec::<lp::Vertex>::with_capacity(mesh.positions.len());
    for vert_idx in 0..(mesh.positions.len() / 3)
    {
        let mut normal = lp::Vec3::default();
        if mesh.normals.len() > 0
        {
            normal.x = mesh.normals[vert_idx*3+0];
            normal.y = mesh.normals[vert_idx*3+1];
            normal.z = mesh.normals[vert_idx*3+2];
            normal = lp::normalize_vec3(normal);
        };

        let mut tex_coords = lp::Vec2::default();
        if mesh.texcoords.len() > 0
        {
            tex_coords.x = mesh.texcoords[vert_idx*2+0];
            // WGPU Convention is +=right,down and tipically it's +=right,up
            tex_coords.y = 1.0 - mesh.texcoords[vert_idx*2+1];
        };

        let vert = lp::Vertex { normal: normal, _padding0: 0.0, tex_coords: tex_coords, _padding1: 0.0, _padding2: 0.0 };

        mesh_verts.push(vert);
    }

    verts_pos.push(mesh_verts_pos);
    verts.push(mesh_verts);
    indices.push(mesh.indices.clone());
    aabbs.push(aabb);
    bvh_nodes.push(bvh);

    return (verts.len() - 1) as u32;
}

#[derive(Default)]
pub struct SceneCamera
{
    pub transform: lp::Mat4,
    pub params: lp::CameraParams,
}

pub fn load_scene_json(path: &std::path::Path, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(lp::Scene, Vec<SceneCamera>), LoadError>
{
    let parent_dir = path.parent().unwrap_or(std::path::Path::new(""));

    let mut verts_pos_array = Vec::<Vec::<lp::VertexPos>>::new();
    let mut verts_array = Vec::<Vec::<lp::Vertex>>::new();
    let mut indices_array = Vec::<Vec::<u32>>::new();
    let mut bvh_nodes_array = Vec::<Vec::<lp::BvhNode>>::new();
    let mut mesh_aabbs = Vec::<lp::Aabb>::new();
    let mut materials = Vec::<lp::Material>::new();
    let environments = Vec::<lp::Environment>::new();

    let mut textures = Vec::<wgpu::Texture>::new();
    let mut samplers = Vec::<wgpu::Sampler>::new();

    let linear_sampler = push_asset(&mut samplers, lp::create_linear_sampler(device));

    let mut instances = Vec::<lp::Instance>::new();

    let mut scene_cams = Vec::<SceneCamera>::new();

    // Default assets at index 0
    let default_tex = push_asset(&mut textures, lp::create_white_texture(device, queue));
    assert!(default_tex == 0);
    let base_tex_id = 1;

    let default_mat = lp::Material::default();  // Default texture is 0 which is what we want

    // Conversion matrix to Lupin's coordinate system, which is left-handed.
    let mut conversion = lp::Mat4::IDENTITY;
    conversion.m[2][2] *= -1.0;

    // Parse json
    let json = std::fs::read(&path)?;
    let mut p = Parser::new(&json[..]);
    p.expect_char('{');

    let mut dict_continue = true;
    while dict_continue
    {
        let strlit = p.next_strlit();
        match strlit
        {
            "cameras" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    let mut scene_cam = SceneCamera::default();
                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let strlit = p.next_strlit();
                        match strlit
                        {
                            "name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            },
                            "aspect" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.aspect = p.parse_f32();
                            },
                            "focus" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.focus = p.parse_f32();
                            },
                            "frame" =>
                            {
                                p.expect_char(':');
                                scene_cam.transform = conversion * p.parse_mat3x4f() * conversion;
                            },
                            "lens" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.lens = p.parse_f32();
                            },
                            _ => {}
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');
                    scene_cams.push(scene_cam);
                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            "environments" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let strlit = p.next_strlit();
                        match strlit
                        {
                            "emission" =>
                            {
                                p.expect_char(':');
                                let emission = p.parse_vec3f();
                            }
                            "emission_tex" =>
                            {
                                p.expect_char(':');
                                let emission_tex = p.parse_u32();
                            }
                            _ => {}
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');
                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            "textures" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let strlit = p.next_strlit();
                        match strlit
                        {
                            "uri" =>
                            {
                                p.expect_char(':');
                                let path_str = p.next_strlit();
                                if !path_str.is_empty()
                                {
                                    let path = std::path::Path::new(path_str);
                                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                                    let is_hdr = matches!(ext.to_lowercase().as_str(), "hdr" | "exr");
                                    let full_path = parent_dir.join(path);

                                    let res = load_texture(device, queue, full_path.to_str().unwrap(), is_hdr);
                                    if let Err(err) = res {
                                        return Err(err.into());
                                    }
                                    push_asset(&mut textures, res.unwrap());
                                }
                            }
                            "name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            }
                            _ => {}
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');

                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            "materials" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let mat = p.parse_material(base_tex_id, default_mat);
                        push_asset(&mut materials, mat);

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');

                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            "shapes" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let strlit = p.next_strlit();
                        match strlit
                        {
                            "uri" =>
                            {
                                p.expect_char(':');
                                let path_str = p.next_strlit();
                                if !path_str.is_empty()
                                {
                                    let path = std::path::Path::new(path_str);
                                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                                    let full_path = parent_dir.join(path);

                                    match ext
                                    {
                                        "ply" =>
                                        {
                                            let res = load_mesh_ply(&full_path, &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs);
                                            if let Err(err) = res {
                                                return Err(err);
                                            }
                                        },
                                        "obj" =>
                                        {
                                            load_mesh_obj(&full_path, &mut verts_pos_array, &mut verts_array, &mut indices_array, &mut bvh_nodes_array, &mut mesh_aabbs);
                                        },
                                        _ =>
                                        {
                                            assert!(false);
                                        }
                                    }
                                }
                            }
                            "name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            }
                            _ => {}
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');
                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            "instances" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    let mut instance = lp::Instance::default();
                    let default_transform = conversion * lp::Mat4::IDENTITY * conversion;
                    instance.inv_transform = lp::mat4_inverse(default_transform);

                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let strlit = p.next_strlit();
                        match strlit
                        {
                            "name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            },
                            "frame" =>
                            {
                                p.expect_char(':');
                                let transform = conversion * p.parse_mat3x4f() * conversion;
                                instance.inv_transform = lp::mat4_inverse(transform);
                            },
                            "material" =>
                            {
                                p.expect_char(':');
                                instance.mat_idx = p.parse_u32();
                            },
                            "shape" =>
                            {
                                p.expect_char(':');
                                instance.mesh_idx = p.parse_u32();
                            },
                            _ => {},
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');
                    instances.push(instance);

                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            "asset" =>
            {
                p.expect_char(':');
                p.expect_char('{');

                while p.buf.len() > 0 && p.buf[0] != b'}' {
                    p.buf = &p.buf[1..];
                }

                p.expect_char('}');
            }
            _ => {}
        }

        dict_continue = p.next_list_el();
        if p.found_error { return Err(LoadError::InvalidJson); }
    }

    p.expect_char('}');

    if p.found_error { return Err(LoadError::InvalidJson); }

    let tlas_nodes = lp::build_tlas(instances.as_slice(), &mesh_aabbs);

    let scene_cpu = lp::SceneCPU {
        verts_pos_array,
        verts_array,
        indices_array,
        bvh_nodes_array,
        mesh_aabbs,
        tlas_nodes,
        instances,
        materials,
        environments,
    };

    lp::validate_scene(&scene_cpu, textures.len() as u32, samplers.len() as u32);

    let scene = lp::upload_scene_to_gpu(device, queue, &scene_cpu, textures, samplers, &[/*env_map_cpu*/]);
    return Ok((scene, scene_cams));
}

// Utility functions used for parsing any simple ASCII textual format.
struct Parser<'a>
{
    pub buf: &'a [u8],
    pub found_error: bool,
}

impl<'a> Parser<'a>
{
    fn new(buf: &'a [u8]) -> Self
    {
        return Self {
            buf,
            found_error: false
        };
    }

    // Returns strlit without "" chars.
    fn next_strlit(&mut self) -> &str
    {
        self.eat_whitespace();

        self.expect_char('\"');

        if self.found_error { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| b == b'"')
            .unwrap_or(self.buf.len());
        let strlit = &self.buf[0..trimmed_start];

        self.buf = &self.buf[trimmed_start..];
        self.expect_char('"');
        if self.found_error { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }

        return std::str::from_utf8(strlit).unwrap();
    }

    fn go_to_next_line(&mut self)
    {
        if self.found_error { return; }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| b == b'\n')
            .unwrap_or(self.buf.len());

        self.buf = &self.buf[trimmed_start..];
        if self.buf.len() > 0 && self.buf[0] == b'\n' {
            self.buf = &self.buf[1..];
        } else {
            self.found_error = true;
        }
    }

    fn next_ident(&mut self) -> &str
    {
        self.eat_whitespace();

        if self.buf.is_empty() { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }
        if !u8::is_ascii_alphabetic(&self.buf[0]) && self.buf[0] != b'_' { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| !(u8::is_ascii_alphabetic(&b) || u8::is_ascii_digit(&b) || b == b'_'))
            .unwrap_or(self.buf.len());
        let token = &self.buf[0..trimmed_start];

        self.buf = &self.buf[trimmed_start..];
        if self.found_error { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }

        return std::str::from_utf8(token).unwrap();
    }

    fn expect_ident(&mut self, ident: &str)
    {
        self.eat_whitespace();

        if self.buf.is_empty() { return; }
        if !u8::is_ascii_alphabetic(&self.buf[0]) && self.buf[0] != b'_' { return; }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| !(u8::is_ascii_alphabetic(&b) || u8::is_ascii_digit(&b) || b == b'_'))
            .unwrap_or(self.buf.len());
        let token = &self.buf[0..trimmed_start];

        self.buf = &self.buf[trimmed_start..];
        if self.found_error { return; }

        if std::str::from_utf8(&token).unwrap() != ident { self.found_error = true; }
    }

    fn peek_ident(&mut self) -> &str
    {
        self.eat_whitespace();

        if self.buf.is_empty() { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }
        if !u8::is_ascii_alphabetic(&self.buf[0]) && self.buf[0] != b'_' { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| !(u8::is_ascii_alphabetic(&b) || u8::is_ascii_digit(&b) || b == b'_'))
            .unwrap_or(self.buf.len());
        let token = &self.buf[0..trimmed_start];

        if self.found_error { return std::str::from_utf8(&self.buf[0..0]).unwrap(); }

        return std::str::from_utf8(token).unwrap();
    }

    fn expect_char(&mut self, c: char)
    {
        self.eat_whitespace();
        if self.buf.len() == 0 { self.found_error = true; return; }
        if !c.is_ascii()       { self.found_error = true; return; }
        if self.buf[0] != c as u8 { self.found_error = true; return; }
        self.buf = &self.buf[1..];
    }

    fn eat_whitespace(&mut self)
    {
        let trimmed_start = self.buf
            .iter()
            .position(|&b| !matches!(b, b' ' | b'\t' | b'\n' | b'\r'))
            .unwrap_or(self.buf.len());
        self.buf = &self.buf[trimmed_start..];
    }

    fn next_list_el(&mut self) -> bool
    {
        self.eat_whitespace();

        if self.buf.len() == 0 { return false; }

        if self.buf[0] != b',' { return false; }

        self.buf = &self.buf[1..];
        return true;
    }

    fn parse_vec3f(&mut self) -> lp::Vec3
    {
        self.eat_whitespace();

        self.expect_char('[');
        let x = self.parse_f32();
        self.expect_char(',');
        let y = self.parse_f32();
        self.expect_char(',');
        let z = self.parse_f32();
        self.expect_char(']');

        return lp::Vec3 { x, y, z };
    }

    fn parse_mat3x4f(&mut self) -> lp::Mat4
    {
        let mut res = lp::Mat4::IDENTITY;

        self.eat_whitespace();

        self.expect_char('[');
        const ROWS: usize = 3;
        const COLUMNS: usize = 4;
        for column in 0..COLUMNS
        {
            for row in 0..ROWS
            {
                res.m[column][row] = self.parse_f32();

                if row < ROWS - 1 || column < COLUMNS - 1 {
                    self.expect_char(',');
                }
            }
        }

        self.expect_char(']');
        return res;
    }

    fn parse_f32(&mut self) -> f32
    {
        self.eat_whitespace();

        if self.buf.is_empty()
        {
            self.found_error = true;
            return 0.0;
        }

        let mut end = 0;
        for (i, &b) in self.buf.iter().enumerate()
        {
            if !(b.is_ascii_digit() || b == b'.' || b == b'-' || b == b'+' || b == b'e' || b == b'E') {
                break;
            }
            end = i + 1;
        }

        if end == 0
        {
            self.found_error = true;
            return 0.0
        }

        let (num_str, rest) = self.buf.split_at(end);

        self.buf = rest;
        match std::str::from_utf8(num_str)
        {
            Ok(s) => match s.parse::<f32>()
            {
                Ok(num) => return num,
                Err(_) =>
                {
                    self.found_error = true;
                    return 0.0
                }
            },
            Err(_) =>
            {
                self.found_error = true;
                return 0.0
            }
        };
    }

    fn parse_u32(&mut self) -> u32
    {
        self.eat_whitespace();

        if self.buf.is_empty()
        {
            self.found_error = true;
            return 0;
        }

        let mut end = 0;
        for (i, &b) in self.buf.iter().enumerate()
        {
            if !(b.is_ascii_digit() || b == b'-' || b == b'+') {
                break;
            }
            end = i + 1;
        }

        if end == 0
        {
            self.found_error = true;
            return 0
        }

        let (num_str, rest) = self.buf.split_at(end);

        self.buf = rest;
        match std::str::from_utf8(num_str)
        {
            Ok(s) => match s.parse::<u32>()
            {
                Ok(num) => return num,
                Err(_) =>
                {
                    self.found_error = true;
                    return 0
                }
            },
            Err(_) =>
            {
                self.found_error = true;
                return 0
            }
        };
    }

    fn parse_material(&mut self, base_tex_idx: u32, default_mat: lp::Material) -> lp::Material
    {
        let mut mat = default_mat;

        let mut dict_continue = true;
        while dict_continue
        {
            let strlit = self.next_strlit();
            match strlit
            {
                "name" =>
                {
                    self.expect_char(':');
                    let name = self.next_strlit();
                }
                "color" =>
                {
                    self.expect_char(':');

                    let color = self.parse_vec3f();
                    mat.color = lp::Vec4 { x: color.x, y: color.y, z: color.z, w: mat.color.w };
                    mat.color.w = 1.0;
                },
                "emission" =>
                {
                    self.expect_char(':');

                    let color = self.parse_vec3f();
                    mat.emission = lp::Vec4 { x: color.x, y: color.y, z: color.z, w: mat.emission.w };
                },
                "scattering" =>
                {
                    self.expect_char(':');

                    let color = self.parse_vec3f();
                    mat.scattering = lp::Vec4 { x: color.x, y: color.y, z: color.z, w: mat.scattering.w }
                },
                "roughness" =>
                {
                    self.expect_char(':');
                    mat.roughness = self.parse_f32();
                },
                "metallic" =>
                {
                    self.expect_char(':');
                    mat.metallic = self.parse_f32();
                },
                "ior" =>
                {
                    self.expect_char(':');
                    mat.ior = self.parse_f32();
                },
                "scanisotropy" =>
                {
                    self.expect_char(':');
                    mat.sc_anisotropy = self.parse_f32();
                },
                "trdepth" =>
                {
                    self.expect_char(':');
                    mat.tr_depth = self.parse_f32();
                },
                "opacity" =>
                {
                    self.expect_char(':');
                    mat.color.w = self.parse_f32();
                },
                "type" =>
                {
                    self.expect_char(':');
                    let mat_type_str = self.next_strlit();
                    match mat_type_str
                    {
                        "matte" => mat.mat_type = lp::MaterialType::Matte,
                        "glossy" => mat.mat_type = lp::MaterialType::Glossy,
                        "reflective" => mat.mat_type = lp::MaterialType::Reflective,
                        "transparent" => mat.mat_type = lp::MaterialType::Transparent,
                        "refractive" => mat.mat_type = lp::MaterialType::Refractive,
                        "subsurface" => mat.mat_type = lp::MaterialType::Subsurface,
                        "volumetric" => mat.mat_type = lp::MaterialType::Volumetric,
                        "gltfpbr" => mat.mat_type = lp::MaterialType::GltfPbr,
                        _ => {}
                    }
                },
                "color_tex" =>
                {
                    self.expect_char(':');
                    mat.color_tex_idx = self.parse_u32() + base_tex_idx;
                },
                "emission_tex" =>
                {
                    self.expect_char(':');
                    mat.emission_tex_idx = self.parse_u32() + base_tex_idx;
                },
                "roughness_tex" =>
                {
                    self.expect_char(':');
                    mat.roughness_tex_idx = self.parse_u32() + base_tex_idx;
                },
                "scattering_tex" =>
                {
                    self.expect_char(':');
                    mat.scattering_tex_idx = self.parse_u32() + base_tex_idx;
                },
                "normal_tex" =>
                {
                    self.expect_char(':');
                    mat.normal_tex_idx = self.parse_u32() + base_tex_idx;
                },
                _ => {}
            }

            dict_continue = self.next_list_el();
        }

        return mat;
    }

}

// .PLY Format

#[derive(Debug)]
pub enum LoadError
{
    Io(std::io::Error),
    ImageErr(image::ImageError),
    InvalidPly,
    InvalidJson,
}

impl From<std::io::Error> for LoadError
{
    fn from(err: std::io::Error) -> Self
    {
        return LoadError::Io(err);
    }
}

impl From<image::ImageError> for LoadError
{
    fn from(err: image::ImageError) -> Self
    {
        return LoadError::ImageErr(err);
    }
}

fn load_mesh_ply(path: &std::path::Path, mesh_verts_pos: &mut Vec<Vec<lp::VertexPos>>, mesh_verts: &mut Vec<Vec<lp::Vertex>>, mesh_indices: &mut Vec<Vec<u32>>, bvh_nodes: &mut Vec<Vec<lp::BvhNode>>, aabbs: &mut Vec<lp::Aabb>) -> Result<u32, LoadError>
{
    assert!(mesh_verts_pos.len() == mesh_verts.len() && mesh_verts.len() == mesh_indices.len() && mesh_indices.len() == aabbs.len() && aabbs.len() == bvh_nodes.len());

    // Parse header
    let ply = std::fs::read(path)?;
    let mut p = Parser::new(&ply);
    let header_continue = true;

    let mut x_buf = Buffer::default();
    let mut y_buf = Buffer::default();
    let mut z_buf = Buffer::default();
    let mut nx_buf = Buffer::default();
    let mut ny_buf = Buffer::default();
    let mut nz_buf = Buffer::default();
    let mut u_buf = Buffer::default();
    let mut v_buf = Buffer::default();
    let indices = Vec::<u32>::default();
    let mut num_verts = 0;
    let mut num_faces = 0;
    let mut vert_size = 0;
    let indices_offset = 0;

    p.expect_ident("ply");

    loop
    {
        let ident = p.next_ident();
        match ident
        {
            "format" =>
            {
                p.expect_ident("binary_little_endian");
                let major_version = p.parse_u32();
                p.expect_char('.');
                let minor_version = p.parse_u32();

                if major_version != 1 || minor_version != 0
                {
                    assert!(false);
                }
            }
            "comment" =>
            {
                p.go_to_next_line();
            }
            "element" =>
            {
                let el_name = p.next_ident();
                if el_name == "vertex"
                {
                    num_verts = p.parse_u32();

                    let mut offset = 0;

                    while p.peek_ident() == "property"
                    {
                        p.next_ident();

                        let mut prop_size = 0;
                        let prop_type_str = p.next_ident();
                        if prop_type_str == "float"
                        {
                            prop_size = 4;
                        }

                        let prop_name = p.next_ident();
                        match prop_name
                        {
                            "x"  => { x_buf.present  = true; x_buf.offset  = offset; },
                            "y"  => { y_buf.present  = true; y_buf.offset  = offset; },
                            "z"  => { z_buf.present  = true; z_buf.offset  = offset; },
                            "nx" => { nx_buf.present = true; nx_buf.offset = offset; },
                            "ny" => { ny_buf.present = true; ny_buf.offset = offset; },
                            "nz" => { nz_buf.present = true; nz_buf.offset = offset; },
                            "u"  => { u_buf.present  = true; u_buf.offset  = offset; },
                            "v"  => { v_buf.present  = true; v_buf.offset  = offset; },
                            _    => {},
                        }

                        offset += prop_size;
                    }

                    let total_size = offset;
                    x_buf.stride = total_size;
                    y_buf.stride = total_size;
                    z_buf.stride = total_size;
                    nx_buf.stride = total_size;
                    ny_buf.stride = total_size;
                    nz_buf.stride = total_size;
                    u_buf.stride = total_size;
                    v_buf.stride = total_size;
                    vert_size = total_size;
                }
                else if el_name == "face"
                {
                    num_faces = p.parse_u32();
                    p.expect_ident("property");
                    p.expect_ident("list");
                    p.expect_ident("uchar");
                    p.expect_ident("int");
                    p.expect_ident("vertex_indices");
                }
            }
            "end_header" => { p.go_to_next_line(); break; }
            _ => { p.found_error = true; }
        }

        if p.found_error {
            return Err(LoadError::InvalidPly);
        }
    }

    if p.found_error {
        return Err(LoadError::InvalidPly);
    }

    let (mut verts_pos, mut verts) = ply_extract_verts(p.buf, &x_buf, &y_buf, &z_buf, &nx_buf, &ny_buf, &nz_buf, &u_buf, &v_buf, num_verts);
    assert!(verts_pos.len() == verts.len());

    let mut indices = ply_extract_indices(&p.buf[(num_verts * vert_size as u32) as usize..], num_faces);

    let missing_normals = !nx_buf.present || !ny_buf.present || !nz_buf.present;
    ply_fill_missing_info(&mut verts_pos, &mut verts, &mut indices, missing_normals);

    // Check indices
    for idx in &indices
    {
        if *idx as usize >= verts_pos.len() {
            return Err(LoadError::InvalidPly);
        }
    }

    let mut aabb = lp::Aabb::neutral();
    for pos in &verts_pos
    {
        lp::grow_aabb_to_include_vert(&mut aabb, pos.v);
    }

    let bvh = lp::build_bvh(verts_pos.as_slice(), &mut indices);

    for vert in &mut verts
    {
        vert.normal = lp::normalize_vec3(vert.normal);

        // TODO: If normal is not present just compute the geometric normal I guess.

        // WGPU Convention is +=right,down and tipically it's +=right,up
        vert.tex_coords.y = 1.0 - vert.tex_coords.y;
    }

    mesh_verts_pos.push(verts_pos);
    mesh_verts.push(verts.clone());
    mesh_indices.push(indices.clone());
    aabbs.push(aabb);
    bvh_nodes.push(bvh);

    return Ok((mesh_verts.len() - 1) as u32);
}

fn ply_extract_verts(buf: &[u8], x: &Buffer, y: &Buffer, z: &Buffer, nx: &Buffer, ny: &Buffer, nz: &Buffer, u: &Buffer, v: &Buffer, num_verts: u32) -> (Vec<lp::VertexPos>, Vec<lp::Vertex>)
{
    let mut verts_pos = vec![lp::VertexPos::default(); num_verts as usize];
    let mut verts = vec![lp::Vertex::default(); num_verts as usize];

    if x.present
    {
        for i in 0..num_verts as usize
        {
            let offset = x.offset + i * x.stride;
            verts_pos[i].v.x = extract_f32(buf, offset);
        }
    }
    if y.present
    {
        for i in 0..num_verts as usize
        {
            let offset = y.offset + i * y.stride;
            verts_pos[i].v.y = extract_f32(buf, offset);
        }
    }
    if z.present
    {
        for i in 0..num_verts as usize
        {
            let offset = z.offset + i * z.stride;
            verts_pos[i].v.z = -extract_f32(buf, offset);
        }
    }
    if nx.present
    {
        for i in 0..num_verts as usize
        {
            let offset = nx.offset + i * nx.stride;
            verts[i].normal.x = extract_f32(buf, offset);
        }
    }
    if ny.present
    {
        for i in 0..num_verts as usize
        {
            let offset = ny.offset + i * ny.stride;
            verts[i].normal.y = extract_f32(buf, offset);
        }
    }
    if nz.present
    {
        for i in 0..num_verts as usize
        {
            let offset = nz.offset + i * nz.stride;
            verts[i].normal.z = -extract_f32(buf, offset);
        }
    }
    if u.present
    {
        for i in 0..num_verts as usize
        {
            let offset = u.offset + i * u.stride;
            verts[i].tex_coords.x = extract_f32(buf, offset);
        }
    }
    if v.present
    {
        for i in 0..num_verts as usize
        {
            let offset = v.offset + i * v.stride;
            verts[i].tex_coords.y = extract_f32(buf, offset);
        }
    }

    assert!(verts_pos.len() == verts.len());
    return (verts_pos, verts);
}

fn ply_extract_indices(buf: &[u8], num_faces: u32) -> Vec<u32>
{
    let mut indices = Vec::<u32>::new();

    let mut s = Serializer { buf: buf };
    for i in 0..num_faces
    {
        let num_indices = s.read_u8();
        assert!(num_indices >= 3);

        let mut idx0 = s.read_u32();
        let mut idx1 = s.read_u32();
        let mut idx2 = s.read_u32();

        indices.push(idx0);
        indices.push(idx1);
        indices.push(idx2);

        for j in 0..num_indices-3
        {
            idx0 = idx1;
            idx1 = idx2;
            idx2 = s.read_u32();
            indices.push(idx0);
            indices.push(idx1);
            indices.push(idx2);
        }
    }

    return indices;
}

fn ply_fill_missing_info(verts_pos: &mut Vec<lp::VertexPos>, verts: &mut Vec<lp::Vertex>, indices: &mut Vec<u32>, missing_normals: bool)
{
    if !missing_normals { return; }

    let mut verts_pos_new = Vec::<lp::VertexPos>::new();
    let mut verts_new = Vec::<lp::Vertex>::new();
    let mut indices_new = Vec::<u32>::new();

    for i in (0..indices.len()).step_by(3)
    {
        let vp0 = verts_pos[indices[i+0] as usize];
        let vp1 = verts_pos[indices[i+1] as usize];
        let vp2 = verts_pos[indices[i+2] as usize];
        let mut v0 = verts[indices[i+0] as usize];
        let mut v1 = verts[indices[i+1] as usize];
        let mut v2 = verts[indices[i+2] as usize];

        let geom_normal = lp::normalize_vec3(lp::cross_vec3(vp1.v - vp0.v, vp2.v - vp1.v));
        v0.normal = geom_normal;
        v1.normal = geom_normal;
        v2.normal = geom_normal;

        let old_len = verts_pos_new.len() as u32;
        verts_pos_new.push(vp0);
        verts_pos_new.push(vp1);
        verts_pos_new.push(vp2);
        verts_new.push(v0);
        verts_new.push(v1);
        verts_new.push(v2);

        let i0 = old_len + 0;
        let i1 = old_len + 1;
        let i2 = old_len + 2;

        indices_new.push(i0);
        indices_new.push(i1);
        indices_new.push(i2);
    }

    *verts_pos = verts_pos_new;
    *verts = verts_new;
    *indices = indices_new;
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Buffer
{
    present: bool,
    offset: usize,
    stride: usize,
}

// Used for binary serialization.
struct Serializer<'a>
{
    pub buf: &'a [u8]
}

impl<'a> Serializer<'a>
{
    fn read_u8(&mut self) -> u8
    {
        if self.buf.len() < 1 {
            return 0;
        }

        let ptr = self.buf.as_ptr() as *const u8;
        let val = unsafe { std::ptr::read_unaligned(ptr) };
        self.buf = &self.buf[1..];
        return val;
    }

    fn read_u32(&mut self) -> u32
    {
        if self.buf.len() < 4 {
            return 0;
        }

        let ptr = self.buf.as_ptr() as *const u32;
        let val = unsafe { std::ptr::read_unaligned(ptr) };
        self.buf = &self.buf[4..];
        return u32::from_le(val);
    }

    fn read_f32(&mut self) -> f32
    {
        if self.buf.len() < 4 {
            return 0.0;
        }

        let ptr = self.buf.as_ptr() as *const f32;
        let val = unsafe { std::ptr::read_unaligned(ptr) };
        self.buf = &self.buf[4..];
        return val;
    }
}

fn extract_f32(buf: &[u8], offset: usize) -> f32
{
    if buf.len() - (offset as usize) < 4 {
        return 0.0;
    }

    let ptr = buf[offset as usize..].as_ptr() as *const f32;
    let val = unsafe { std::ptr::read_unaligned(ptr) };
    return val;
}

// Saving

/// Detects file format from its extension. Supports rgba8_unorm, rgba16f
pub fn save_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path, texture: &wgpu::Texture) -> Result<(), image::ImageError>
{
    // TODO: Check texture format.
    let format = texture.format();
    assert!(format == wgpu::TextureFormat::Rgba8Unorm || format == wgpu::TextureFormat::Rgba16Float);

    let width = texture.size().width;
    let height = texture.size().height;

    let bytes_per_pixel = match format {
        wgpu::TextureFormat::Rgba8Unorm => 4,
        wgpu::TextureFormat::Rgba16Float => 8,
        _ => { panic!("unsupported texture format"); }
    };
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture into buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Copy Encoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map synchronously
    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    // Wait until GPU work is done and buffer is ready
    device.poll(wgpu::Maintain::Wait);

    // Access mapped data
    let data = buffer_slice.get_mapped_range();

    // Save as .png file
    match format
    {
        wgpu::TextureFormat::Rgba8Unorm =>
        {
            let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, data).unwrap();
            let res = img.save(path);
            return res;
        },
        wgpu::TextureFormat::Rgba16Float =>
        {
            assert!(data.len() % 2 == 0, "Data must be aligned to 2 bytes");
            let data_f16: &[half::f16] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
            };

            let mut data_f32_no_alpha = Vec::with_capacity((width * height * 3) as usize);
            for px in data_f16.chunks_exact(4)
            {
                data_f32_no_alpha.push(f32::from(px[0]));
                data_f32_no_alpha.push(f32::from(px[1]));
                data_f32_no_alpha.push(f32::from(px[2]));
            }
            let img = image::ImageBuffer::<image::Rgb<f32>, _>::from_raw(width, height, data_f32_no_alpha).unwrap();
            let res = img.save(path);
            return res;
        },
        _ => { panic!("unsupported texture type"); }
    }
}

/*
fn save_texture_hdr(device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path, texture: &wgpu::Texture) -> Result<(), image::ImageError>
{
    // TODO: Check texture format.

    let width = texture.size().width;
    let height = texture.size().height;

    let bytes_per_pixel = 16; // R32G32B32A32_FLOAT = 4 x f32
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture into buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Copy Encoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map synchronously
    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});

    // Wait until GPU work is done and buffer is ready
    device.poll(wgpu::Maintain::Wait);

    // Access mapped data
    let data = buffer_slice.get_mapped_range();

    // Convert to Vec<RGBE8Pixel>
    /*
    let mut pixels: Vec<RGBE8Pixel> = Vec::with_capacity((width * height) as usize);
    for chunk in data.chunks(padded_bytes_per_row as usize) {
        let row = &chunk[..unpadded_bytes_per_row as usize];
        let floats: &[f32] = bytemuck::cast_slice(row);

        for px in floats.chunks(4) {
            let (r, g, b, _a) = (px[0], px[1], px[2], px[3]);
            pixels.push(RGBE8Pixel::from_f32(r, g, b));
        }
    }
    */

    // Save as .hdr file
    let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, data)
        .expect("Vec length does not match width * height * 4");
    let res = img.save(path);
    return res;
}
*/
