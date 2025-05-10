
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
        // for 16-byte padding.

        verts_pos.reserve_exact(mesh.positions.len() + mesh.positions.len() / 3);
        for i in (0..mesh.positions.len()).step_by(3)
        {
            verts_pos.push(mesh.positions[i + 0]*5.0);  // NOTE NOTE NOTE remove this
            verts_pos.push(mesh.positions[i + 1]*5.0);
            verts_pos.push(mesh.positions[i + 2]*5.0);
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

        let instances = [
            lp::Instance { pos: Vec3 { x: 0.0, y: 0.0, z: 0.0 }.into(), mesh_idx: 0, texture_idx: 0, sampler_idx: 0, padding0: 0.0, padding1: 0.0 },
            lp::Instance { pos: Vec3 { x: 0.0, y: 1.5, z: 0.0 }.into(), mesh_idx: 0, texture_idx: 0, sampler_idx: 0, padding0: 0.0, padding1: 0.0 },
            lp::Instance { pos: Vec3 { x: 0.0, y: 3.0, z: 0.0 }.into(), mesh_idx: 0, texture_idx: 0, sampler_idx: 0, padding0: 0.0, padding1: 0.0 },
            lp::Instance { pos: Vec3 { x: 0.0, y: 4.5, z: 0.0 }.into(), mesh_idx: 0, texture_idx: 0, sampler_idx: 0, padding0: 0.0, padding1: 0.0 },
            lp::Instance { pos: Vec3 { x: 0.0, y: 7.0, z: 0.0 }.into(), mesh_idx: 0, texture_idx: 0, sampler_idx: 0, padding0: 0.0, padding1: 0.0 },
        ];

        let instances_buf = lp::upload_storage_buffer(&device, &queue, unsafe { to_u8_slice(&instances) });

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

        return lp::SceneDesc
        {
            verts_pos: vec![verts_pos_buf],
            verts:     vec![verts_buf],
            indices:   vec![indices_buf],
            bvh_nodes: vec![bvh_buf],

            tlas_nodes: lp::create_empty_storage_buffer(device),
            instances:  instances_buf,

            textures: vec![load_texture(device, queue, "bunny_texture.jpg")],
            //textures: vec![],
            samplers: vec![linear_sampler],
        };
    }

    return lp::SceneDesc
    {
        verts_pos: vec![],
        verts:     vec![],
        indices:   vec![],
        bvh_nodes: vec![],

        tlas_nodes: lp::create_empty_storage_buffer(device),
        instances:  lp::create_empty_storage_buffer(device),

        textures: vec![],
        samplers: vec![],
    };
}

pub fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &str) -> wgpu::Texture
{
    use image::GenericImageView;

    let img = image::open(path).expect("Failed to load image");
    let rgba = img.to_rgba8();
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
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[]
    });

    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All
        },
        &rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * dimensions.0),
            rows_per_image: Some(dimensions.1)
        },
        size
    );

    return texture;
}
