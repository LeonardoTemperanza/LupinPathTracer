
use lupin as lp;

use crate::base::*;

pub fn build_scene(device: &wgpu::Device, queue: &wgpu::Queue) -> lp::SceneDesc
{
    let scene = tobj::load_obj("stanford_bunny.obj", &tobj::GPU_LOAD_OPTIONS);

    assert!(scene.is_ok());
    let (mut models, _materials) = scene.expect("Failed to load OBJ file");

    let mesh = &mut models[0].mesh;

    // Construct the buffer to send to GPU. Include an extra float
    // for 16-byte alignment of vectors.

    let mut aabb = Aabb::neutral();
    let mut verts_pos = Vec::<f32>::with_capacity(mesh.positions.len() + mesh.positions.len() / 3);
    for i in (0..mesh.positions.len()).step_by(3)
    {
        let pos = Vec3 { x: mesh.positions[i + 0], y: mesh.positions[i + 1], z: mesh.positions[i + 2] };
        verts_pos.push(mesh.positions[i + 0]);
        verts_pos.push(mesh.positions[i + 1]);
        verts_pos.push(mesh.positions[i + 2]);
        verts_pos.push(0.0);
        grow_aabb_to_include_vert(&mut aabb, pos);
    }

    let bvh_buf = lp::build_bvh(&device, &queue, verts_pos.as_slice(), &mut mesh.indices);

    let verts_pos_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts_pos) });
    let indices_buf   = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&mesh.indices) });
    let mut verts = Vec::<lp::Vertex>::with_capacity(mesh.positions.len() / 3);
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
            // WGPU Convention is +=right,down and tipically it's +=right,up
            tex_coords.y = -mesh.texcoords[vert_idx*2+1];
        };

        let vert = lp::Vertex { normal: normal.into(), padding0: 0.0, tex_coords: tex_coords.into(), padding1: 0.0, padding2: 0.0 };

        verts.push(vert);
    }

    let verts_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&verts) });

    // Stress-test
    /*
    let mut instances = Vec::<lp::Instance>::default();
    for i in 0..100
    {
        for j in 0..100
        {
            let offset: f32 = 1.5;
            instances.push(lp::Instance {
                pos: Vec3 { x: offset * i as f32, y: 0.0, z: offset * j as f32 }.into(),
                mesh_idx: 0,
                mat_idx: 0,
                padding0: 0.0, padding1: 0.0, padding2: 0.0,
            });
        }
    }
    */

    let instances = [
        lp::Instance { inv_transform: lp::mat4_inverse(xform_to_matrix(Vec3 { x: 0.0, y: 0.0, z: 0.0 }.into(), Quat::default(), Vec3::ones()).into()), mesh_idx: 0, mat_idx: 0, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(xform_to_matrix(Vec3 { x: -2.0, y: 0.0, z: 0.0 }.into(), angle_axis(Vec3::RIGHT, 40.0 * 3.1415 / 180.0), Vec3::ones()).into()), mesh_idx: 0, mat_idx: 1, padding0: 0.0, padding1: 0.0 },
        lp::Instance { inv_transform: lp::mat4_inverse(xform_to_matrix(Vec3 { x: 2.0, y: 0.0, z: 0.0 }.into(), Quat::default(), Vec3 { x: 0.8, y: 1.3, z: 1.0 }).into()), mesh_idx: 0, mat_idx: 2, padding0: 0.0, padding1: 0.0 },
    ];

    let instances_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&instances) });

    let materials = [
        lp::Material::new(
            lp::MaterialType::Matte,            // Mat type
            lp::Vec4::new(1.0, 0.8, 0.8, 1.0),  // Color
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
            0.0,                                // Roughness
            0.0,                                // Metallic
            0.0,                                // ior
            0.0,                                // anisotropy
            0.0,                                // depth
            0,                                  // Color tex
            0,                                  // Emission tex
            0                                   // Roughness tex
        ),
        lp::Material::new(
            lp::MaterialType::Glossy,           // Mat type
            lp::Vec4::new(1.0, 1.0, 1.0, 1.0),  // Color
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
            0.0,                                // Roughness
            0.0,                                // Metallic
            0.0,                                // ior
            0.0,                                // anisotropy
            0.0,                                // depth
            0,                                  // Color tex
            0,                                  // Emission tex
            0                                   // Roughness tex
        ),
        lp::Material::new(
            lp::MaterialType::Reflective,       // Mat type
            lp::Vec4::new(1.0, 1.0, 1.0, 1.0),  // Color
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
            0.0,                                // Roughness
            0.0,                                // Metallic
            0.0,                                // ior
            0.0,                                // anisotropy
            0.0,                                // depth
            0,                                  // Color tex
            0,                                  // Emission tex
            0                                   // Roughness tex
        ),
        lp::Material::new(
            lp::MaterialType::Transparent,      // Mat type
            lp::Vec4::new(1.0, 1.0, 1.0, 1.0),  // Color
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Emission
            lp::Vec4::new(0.0, 0.0, 0.0, 0.0),  // Scattering
            0.0,                                // Roughness
            0.0,                                // Metallic
            0.0,                                // ior
            0.0,                                // anisotropy
            0.0,                                // depth
            0,                                  // Color tex
            0,                                  // Emission tex
            0                                   // Roughness tex
        ),
    ];

    let materials_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&materials) });

    let lp_aabb: lp::Aabb = aabb.into();
    let tlas_buf = lp::build_tlas(&device, &queue, instances.as_slice(), &[lp_aabb]);

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
        verts_pos: vec![verts_pos_buf],
        verts:     vec![verts_buf],
        indices:   vec![indices_buf],
        bvh_nodes: vec![bvh_buf],

        tlas_nodes: tlas_buf,
        instances:  instances_buf,
        materials:  materials_buf,

        textures: vec![load_texture(device, queue, "bunny_color.png", false)],
        samplers: vec![linear_sampler],
        env_map: load_texture(device, queue, "poly_haven_studio_1k.hdr", true),
        env_map_sampler: env_map_sampler,
    };
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
        let rgba = rgba32f_to_rgba16f(&img);

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All
            },
            unsafe { to_u8_slice(&rgba) },
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
            unsafe { to_u8_slice(&rgba.into_raw()) },
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

use half::*;

fn rgba32f_to_rgba16f(image: &image::DynamicImage) -> Vec<f16>
{
    let rgba32f = image.to_rgba32f();
    rgba32f.pixels()
        .flat_map(|p| p.0.iter().map(|&f| f16::from_f32(f)))
        .collect()
}
