
use lupin as lp;

pub fn build_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> lp::SceneDesc
{
    let meshes_aabbs = vec![
        load_obj_mesh(device, queue, "stanford_bunny.obj"),
        load_obj_mesh(device, queue, "quad.obj"),
        load_obj_mesh(device, queue, "gazerbo.obj"),
        load_obj_mesh(device, queue, "Dragon_80K.obj"),
    ];

    let bunny_mesh = 0;
    let quad_mesh = 1;
    let gazebo_mesh = 2;
    let dragon_80k_mesh = 3;

    let mut meshes = Vec::<lp::Mesh>::with_capacity(meshes_aabbs.len());
    let mut aabbs = Vec::<lp::Aabb>::with_capacity(meshes_aabbs.len());
    for mesh_aabb in meshes_aabbs
    {
        meshes.push(mesh_aabb.0);
        aabbs.push(mesh_aabb.1);
    }

    let mut materials = Vec::<lp::Material>::new();

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
        0.00001,                                // Roughness
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

    let reflective = push_asset(&mut materials, lp::Material::new(
        lp::MaterialType::Reflective,       // Mat type
        lp::Vec4::new(0.9, 0.2, 0.2, 1.0),  // Color
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.0,                                // Roughness
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
        lp::Vec4::new(2.0, 2.0, 2.0, 1.0),  // Color
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
        lp::Vec4::new(1.0, 1.0, 1.0, 1.0),  // Color
        lp::Vec4::new(5.0, 5.0, 5.0, 0.0),  // Emission
        lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
        0.05,                               // Roughness
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

    let materials_buf = lp::upload_storage_buffer(&device, &queue, lp::to_u8_slice(&materials));

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

    let instances = [
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: bunny_matte, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: -2.0, y: 0.0, z: 0.0 }, lp::angle_axis(lp::Vec3::RIGHT, 45.0 * 3.1415 / 180.0), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 1, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: -2.0, y: 0.0, z: -2.0 }, lp::angle_axis(lp::Vec3::RIGHT, 45.0 * 3.1415 / 180.0), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 7, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 2.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 })), mesh_idx: 0, mat_idx: 2, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 2.0, y: 0.0, z: -2.0 }, lp::Quat::default(), lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 })), mesh_idx: 0, mat_idx: 6, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 4.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 3, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 4.0, y: 0.0, z: -2.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 10, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 6.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 4, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 6.0, y: 0.0, z: -2.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 11, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 8.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 8, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 10.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 9, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 12.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: 0, mat_idx: 12, padding0: 0.0, padding1: 0.0 },
        // Floor
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: -0.01, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones() * 20.0)), mesh_idx: 1, mat_idx: brown_matte, padding0: 0.0, padding1: 0.0 },
        // Gazebo
        lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 30.0, y: 0.0, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones())), mesh_idx: gazebo_mesh, mat_idx: brown_matte, padding0: 0.0, padding1: 0.0 },
        // Dragon
        // lp::Instance { inv_transform: lp::mat4_inverse(lp::xform_to_matrix(lp::Vec3 { x: 0.0, y: 2.5, z: 0.0 }, lp::Quat::default(), lp::Vec3::ones() * 10.0)), mesh_idx: dragon_80k_mesh, mat_idx: transparent, padding0: 0.0, padding1: 0.0 },
    ];

    let instances_buf = lp::upload_storage_buffer(&device, &queue, lp::to_u8_slice(&instances));

    let tlas_buf = lp::build_tlas(&device, &queue, instances.as_slice(), &aabbs);

    let mut textures = Vec::<wgpu::Texture>::new();
    let white_tex = push_asset(&mut textures, lp::create_white_texture(device, queue));
    let bunny_color = push_asset(&mut textures, load_texture(device, queue, "bunny_color.png", false));
    //let (env_map_cpu, env_map_gpu) = load_hdr_texture_and_keep_cpu_copy(device, queue, "poly_haven_studio_1k.hdr");
    let (env_map_cpu, env_map_gpu) = load_hdr_texture_and_keep_cpu_copy(device, queue, "sky.hdr");
    let env_map = push_asset(&mut textures, env_map_gpu);

    let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let environment = lp::Environment {
        emission: lp::Vec3 { x: 1.0, y: 1.0, z: 1.0 },
        emission_tex_idx: env_map,
    };

    let env_buf = lp::upload_storage_buffer(device, queue, lp::to_u8_slice(&[environment]));

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

    return lp::SceneDesc {
        meshes: meshes,
        tlas_nodes: tlas_buf,
        instances:  instances_buf,
        materials:  materials_buf,

        textures: textures,
        samplers: vec![linear_sampler],
        environments: env_buf,

        // Lights
        lights: lp::build_lights(device, queue, &instances, &[environment], &[env_map_cpu]),
    };
}

pub fn load_obj_mesh(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> (lp::Mesh, lp::Aabb)
{
    let scene = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS);
    assert!(scene.is_ok());

    let (mut models, _materials) = scene.expect("Failed to load OBJ file");

    let mesh = &mut models[0].mesh;

    // Construct the buffer to send to GPU. Include an extra float
    // for 16-byte alignment of vectors.

    let mut aabb = lp::Aabb::neutral();
    let mut verts_pos = Vec::<f32>::with_capacity(mesh.positions.len() + mesh.positions.len() / 3);
    for i in (0..mesh.positions.len()).step_by(3)
    {
        let pos = lp::Vec3 { x: mesh.positions[i + 0], y: mesh.positions[i + 1], z: mesh.positions[i + 2] };
        verts_pos.push(mesh.positions[i + 0]);
        verts_pos.push(mesh.positions[i + 1]);
        verts_pos.push(mesh.positions[i + 2]);
        verts_pos.push(0.0);
        lp::grow_aabb_to_include_vert(&mut aabb, pos);
    }

    let bvh_buf = lp::build_bvh(&device, &queue, verts_pos.as_slice(), &mut mesh.indices);

    let verts_pos_buf = lp::upload_storage_buffer(&device, &queue, lp::to_u8_slice(&verts_pos));
    let indices_buf   = lp::upload_storage_buffer(&device, &queue, lp::to_u8_slice(&mesh.indices));
    let mut verts = Vec::<lp::Vertex>::with_capacity(mesh.positions.len() / 3);
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
            tex_coords.y = -mesh.texcoords[vert_idx*2+1];
        };

        let vert = lp::Vertex { normal: normal, padding0: 0.0, tex_coords: tex_coords, padding1: 0.0, padding2: 0.0 };

        verts.push(vert);
    }

    let verts_buf = lp::upload_storage_buffer(&device, &queue, lp::to_u8_slice(&verts));

    return (lp::Mesh {
        verts_pos: verts_pos_buf,
        verts: verts_buf,
        indices: indices_buf,
        bvh_nodes: bvh_buf,
    }, aabb);
}

pub fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &str, hdr: bool) -> wgpu::Texture
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

    return texture;
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

use half::*;

fn rgba32f_to_rgba16f(image_rgba32f: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>) -> Vec<f16>
{
    return image_rgba32f.pixels()
                        .flat_map(|p| p.0.iter().map(|&f| f16::from_f32(f)))
                        .collect();
}

fn push_asset<T>(vec: &mut Vec<T>, el: T) -> u32
{
    vec.push(el);
    return (vec.len() - 1) as u32;
}
