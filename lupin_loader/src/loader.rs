
use lupin_pt as lp;
use lupin_pt::wgpu as wgpu;

/// Builds an empty scene, which should show up as a black texture in a render.
/// Useful for testing of this library, more than anything else, really.
pub fn build_scene_empty(device: &wgpu::Device, queue: &wgpu::Queue) -> lp::Scene
{
    let scene_cpu = lp::SceneCPU::default();
    lp::validate_scene(&scene_cpu, 0, 0);
    return lp::build_accel_structures_and_upload(device, queue, &scene_cpu, vec![], vec![], vec![], &[], true);
}

pub fn build_scene_cornell_box(device: &wgpu::Device, queue: &wgpu::Queue, build_sw_and_hw: bool) -> (lp::Scene, Vec<SceneCamera>)
{
    // Values taken from YoctoGL.

    let mut scene = lp::SceneCPU::default();

    let white_mat = push_asset(&mut scene.materials, {
        let mut mat = lp::Material::default();
        mat.color = lp::Vec4 { x: 0.725, y: 0.71, z: 0.68, w: 1.0 };
        mat
    });
    let red_mat = push_asset(&mut scene.materials, {
        let mut mat = lp::Material::default();
        mat.color = lp::Vec4 { x: 0.63, y: 0.065, z: 0.05, w: 1.0 };
        mat
    });
    let green_mat = push_asset(&mut scene.materials, {
        let mut mat = lp::Material::default();
        mat.color = lp::Vec4 { x: 0.14, y: 0.45, z: 0.091, w: 1.0 };
        mat
    });
    let emissive_mat = push_asset(&mut scene.materials, {
        let mut mat = lp::Material::default();
        mat.emission = lp::Vec4 { x: 17.0, y: 12.0, z: 4.0, w: 0.0 };
        mat
    });

    // Floor
    {
        let floor_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(-1.0, 0.0,  1.0), lp::Vec4::new3( 1.0, 0.0,  1.0),
            lp::Vec4::new3( 1.0, 0.0, -1.0), lp::Vec4::new3(-1.0, 0.0, -1.0),
        ]);
        scene.indices_array.push(vec![0, 1, 2, 2, 3, 0]);

        let floor_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = floor_mesh;
            instance.mat_idx  = white_mat;
            instance
        });
    }

    // Ceiling
    {
        let ceiling_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(-1.0, 2.0,  1.0), lp::Vec4::new3(-1.0, 2.0, -1.0),
            lp::Vec4::new3( 1.0, 2.0, -1.0), lp::Vec4::new3( 1.0, 2.0,  1.0),
        ]);
        scene.indices_array.push(vec![0, 1, 2, 2, 3, 0]);

        let ceiling_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = ceiling_mesh;
            instance.mat_idx  = white_mat;
            instance
        });
    }

    // Backwall
    {
        let backwall_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(-1.0, 0.0, 1.0), lp::Vec4::new3( 1.0, 0.0, 1.0),
            lp::Vec4::new3( 1.0, 2.0, 1.0), lp::Vec4::new3(-1.0, 2.0, 1.0),
        ]);
        scene.indices_array.push(vec![0, 2, 1, 2, 0, 3]);

        let backwall_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = backwall_mesh;
            instance.mat_idx  = white_mat;
            instance
        });
    }

    // Rightwall
    {
        let rightwall_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(1.0, 0.0, -1.0), lp::Vec4::new3(1.0, 0.0,  1.0),
            lp::Vec4::new3(1.0, 2.0,  1.0), lp::Vec4::new3(1.0, 2.0, -1.0),
        ]);
        scene.indices_array.push(vec![0, 1, 2, 2, 3, 0]);

        let rightwall_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = rightwall_mesh;
            instance.mat_idx  = green_mat;
            instance
        });
    }

    // Leftwall
    {
        let leftwall_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(-1.0, 0.0,  1.0), lp::Vec4::new3(-1.0, 0.0, -1.0),
            lp::Vec4::new3(-1.0, 2.0, -1.0), lp::Vec4::new3(-1.0, 2.0,  1.0),
        ]);
        scene.indices_array.push(vec![0, 1, 2, 2, 3, 0]);

        let leftwall_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = leftwall_mesh;
            instance.mat_idx  = red_mat;
            instance
        });
    }

    // Shortbox
    {
        let shortbox_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(0.53, 0.6, -0.75), lp::Vec4::new3(0.7, 0.6, -0.17), lp::Vec4::new3(0.13, 0.6, -0.0),
            lp::Vec4::new3(-0.05, 0.6, -0.57), lp::Vec4::new3(-0.05, 0.0, -0.57), lp::Vec4::new3(-0.05, 0.6, -0.57),
            lp::Vec4::new3(0.13, 0.6, -0.0), lp::Vec4::new3(0.13, 0.0, -0.0), lp::Vec4::new3(0.53, 0.0, -0.75),
            lp::Vec4::new3(0.53, 0.6, -0.75), lp::Vec4::new3(-0.05, 0.6, -0.57), lp::Vec4::new3(-0.05, 0.0, -0.57),
            lp::Vec4::new3(0.7, 0.0, -0.17), lp::Vec4::new3(0.7, 0.6, -0.17), lp::Vec4::new3(0.53, 0.6, -0.75),
            lp::Vec4::new3(0.53, 0.0, -0.75), lp::Vec4::new3(0.13, 0.0, -0.0), lp::Vec4::new3(0.13, 0.6, -0.0),
            lp::Vec4::new3(0.7, 0.6, -0.17), lp::Vec4::new3(0.7, 0.0, -0.17), lp::Vec4::new3(0.53, 0.0, -0.75),
            lp::Vec4::new3(0.7, 0.0, -0.17), lp::Vec4::new3(0.13, 0.0, -0.0), lp::Vec4::new3(-0.05, 0.0, -0.57),
        ]);
        scene.indices_array.push(vec![0, 2, 1, 2, 0, 3, 4, 6, 5, 6, 4, 7,
                                      8, 10, 9, 10, 8, 11, 12, 14, 13, 14, 12, 15,
                                      16, 18, 17, 18, 16, 19, 20, 22, 21, 22, 20, 23]);

        let shortbox_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = shortbox_mesh;
            instance.mat_idx  = white_mat;
            instance
        });
    }

    // Tallbox
    {
        let tallbox_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(-0.53, 1.2, -0.09), lp::Vec4::new3(0.04, 1.2, 0.09), lp::Vec4::new3(-0.14, 1.2, 0.67),
            lp::Vec4::new3(-0.71, 1.2, 0.49), lp::Vec4::new3(-0.53, 0.0, -0.09), lp::Vec4::new3(-0.53, 1.2, -0.09),
            lp::Vec4::new3(-0.71, 1.2, 0.49), lp::Vec4::new3(-0.71, 0.0, 0.49), lp::Vec4::new3(-0.71, 0.0, 0.49),
            lp::Vec4::new3(-0.71, 1.2, 0.49), lp::Vec4::new3(-0.14, 1.2, 0.67), lp::Vec4::new3(-0.14, 0.0, 0.67),
            lp::Vec4::new3(-0.14, 0.0, 0.67), lp::Vec4::new3(-0.14, 1.2, 0.67), lp::Vec4::new3(0.04, 1.2, 0.09),
            lp::Vec4::new3(0.04, 0.0, 0.09), lp::Vec4::new3(0.04, 0.0, 0.09), lp::Vec4::new3(0.04, 1.2, 0.09),
            lp::Vec4::new3(-0.53, 1.2, -0.09), lp::Vec4::new3(-0.53, 0.0, -0.09), lp::Vec4::new3(-0.53, 0.0, -0.09),
            lp::Vec4::new3(0.04, 0.0, 0.09), lp::Vec4::new3(-0.14, 0.0, 0.67), lp::Vec4::new3(-0.71, 0.0, 0.49),
        ]);
        scene.indices_array.push(vec![0, 2, 1, 2, 0, 3, 4, 6, 5, 6, 4, 7,
                                      8, 10, 9, 10, 8, 11, 12, 14, 13, 14, 12, 15,
                                      16, 18, 17, 18, 16, 19, 20, 22, 21, 22, 20, 23]);

        let tallbox_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = tallbox_mesh;
            instance.mat_idx  = white_mat;
            instance
        });
    }

    // Light
    {
        let light_mesh = push_asset(&mut scene.mesh_infos, lp::MeshInfo::default());
        scene.verts_pos_array.push(vec![
            lp::Vec4::new3(-0.25, 1.99, -0.25), lp::Vec4::new3(-0.25, 1.99,  0.25),
            lp::Vec4::new3(0.25,  1.99, 0.25),  lp::Vec4::new3(0.25,  1.99, -0.25),
        ]);
        scene.indices_array.push(vec![0, 2, 1, 2, 0, 3]);

        let light_instance = push_asset(&mut scene.instances, {
            let mut instance = lp::Instance::default();
            instance.mesh_idx = light_mesh;
            instance.mat_idx  = emissive_mat;
            instance
        });
    }

    lp::validate_scene(&scene, 0, 0);

    let cameras = vec![SceneCamera {
        transform: lp::Mat3x4 { m: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, -3.9]] },
        params: lp::CameraParams {
            is_orthographic: false,
            lens: 0.035,
            aperture: 0.0,
            focus: 3.9,
            film: 0.024,
            aspect: 1.0,
        },
    }];
    return (lp::build_accel_structures_and_upload(device, queue, &scene, vec![], vec![], vec![], &[], build_sw_and_hw), cameras);
}

pub fn load_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &str, hdr: bool) -> Result<wgpu::Texture, image::ImageError>
{
    return load_texture_with_usage(device, queue, path, hdr, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
}

pub fn load_texture_with_usage(device: &wgpu::Device, queue: &wgpu::Queue, path: &str, hdr: bool, usage: wgpu::TextureUsages) -> Result<wgpu::Texture, image::ImageError>
{
    use image::GenericImageView;

    let img = image::open(path)?;
    let dimensions = img.dimensions();

    let size = wgpu::Extent3d {
        width: dimensions.0,
        height: dimensions.1,
        depth_or_array_layers: 1
    };

    let format = if hdr {
        wgpu::TextureFormat::Rgba16Float
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format,
        usage: usage | wgpu::TextureUsages::COPY_DST,
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

fn rgba32f_to_rgba16f(image_rgba32f: &image::ImageBuffer<image::Rgba<f32>, Vec<f32>>) -> Vec<half::f16>
{
    return image_rgba32f.pixels()
        .flat_map(|p| p.0.iter().map(|&f| half::f16::from_f32(f)))
        .collect();
}

fn push_asset<T>(vec: &mut Vec<T>, el: T) -> u32
{
    vec.push(el);
    return (vec.len() - 1) as u32;
}

/// Camera that has been placed into the scene.
/// A scene can contain many cameras.
#[derive(Default)]
pub struct SceneCamera
{
    pub transform: lp::Mat3x4,
    pub params: lp::CameraParams,
}

fn grow_vec<T: Default + Clone>(v: &mut Vec<T>, size: usize)
{
    if size <= v.len() { return; }
    v.resize(size, Default::default());
}

// Info saved from the first pass of json parsing.
// NOTE: This is needed because in the yoctogl format,
// the textures themselves don't contain information about
// the usage of said texture. Here we do need the context
// because we create a different texture for color vs normals,
// (also using different compression settings)
#[derive(Default, Clone, Debug)]
struct TextureLoadInfo
{
    path: String,
    used_for_color: bool,
    used_for_data: bool,
}

/// Load a scene in the format used by the Yocto/GL library, version 2.4.
pub fn load_scene_yoctogl_v24(path: &std::path::Path, device: &wgpu::Device, queue: &wgpu::Queue, build_both_bvhs: bool) -> Result<(lp::Scene, Vec<SceneCamera>), LoadError>
{
    let parent_dir = path.parent().unwrap_or(std::path::Path::new(""));

    let mut scene = lp::SceneCPU::default();
    let mut textures = Vec::<wgpu::Texture>::new();
    let mut texture_views = Vec::<wgpu::TextureView>::new();
    let mut samplers = Vec::<wgpu::Sampler>::new();

    let mut tex_load_infos = Vec::<TextureLoadInfo>::new();
    let mut num_parsed_textures = 0;

    let mut scene_cams = Vec::<SceneCamera>::new();

    // Conversion matrix to Lupin's coordinate system, which is left-handed.
    let mut conversion = lp::Mat3x4::IDENTITY;
    conversion.m[2][2] *= -1.0;
    let mut conversion_mat4 = lp::Mat4::IDENTITY;
    conversion_mat4.m[2][2] *= -1.0;

    // Parse json.
    let json = std::fs::read(&path)?;
    let mut p = Parser::new(&json[..]);
    p.expect_char('{');

    let mut dict_continue = true;
    while dict_continue
    {
        let strlit = p.next_strlit();
        match strlit
        {
            b"cameras" =>
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
                            b"name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            },
                            b"aspect" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.aspect = p.parse_f32();
                            },
                            b"focus" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.focus = p.parse_f32();
                            },
                            b"aperture" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.aperture = p.parse_f32();
                            },
                            b"frame" =>
                            {
                                p.expect_char(':');
                                scene_cam.transform = conversion * p.parse_mat3x4f() * conversion;
                            },
                            b"lens" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.lens = p.parse_f32();
                            },
                            b"orthographic" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.is_orthographic = p.parse_bool();
                            }
                            b"film" =>
                            {
                                p.expect_char(':');
                                scene_cam.params.film = p.parse_f32();
                            }
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
            b"environments" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut env = lp::Environment::default();
                env.transform = conversion_mat4 * lp::Mat4::IDENTITY;

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
                            b"name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            }
                            b"emission" =>
                            {
                                p.expect_char(':');
                                env.emission = p.parse_vec3f();
                            }
                            b"emission_tex" =>
                            {
                                p.expect_char(':');
                                env.emission_tex_idx = p.parse_u32();
                                if env.emission_tex_idx != lp::SENTINEL_IDX
                                {
                                    grow_vec(&mut tex_load_infos, env.emission_tex_idx as usize + 1);
                                    tex_load_infos[env.emission_tex_idx as usize].used_for_color = true;
                                }
                            }
                            b"frame" =>
                            {
                                p.expect_char(':');
                                env.transform = conversion_mat4 * p.parse_mat3x4f().to_mat4();
                            }
                            _ => {}
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');
                    scene.environments.push(env);

                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            b"textures" =>
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
                            b"uri" =>
                            {
                                p.expect_char(':');
                                let path_str = p.next_strlit();
                                if !path_str.is_empty()
                                {
                                    let path = std::path::Path::new(std::str::from_utf8(path_str).unwrap());
                                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                                    let is_hdr = matches!(ext.to_lowercase().as_str(), "hdr" | "exr");
                                    let full_path = parent_dir.join(path);

                                    grow_vec(&mut tex_load_infos, num_parsed_textures + 1);
                                    tex_load_infos[num_parsed_textures].path = String::from(full_path.to_str().unwrap());
                                }
                            }
                            b"name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            }
                            _ => {}
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');

                    num_parsed_textures += 1;
                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            b"materials" =>
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
                        let mat = parse_material_yocto_v24(&mut p, &mut tex_load_infos);
                        push_asset(&mut scene.materials, mat);

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');

                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            b"shapes" =>
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
                            b"uri" =>
                            {
                                p.expect_char(':');
                                let path_str = p.next_strlit();
                                if !path_str.is_empty()
                                {
                                    let path = std::path::Path::new(std::str::from_utf8(path_str).unwrap());
                                    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                                    let full_path = parent_dir.join(path);

                                    match ext
                                    {
                                        "ply" =>
                                        {
                                            let res = load_mesh_ply(&full_path, &mut scene);
                                            if let Err(err) = res {
                                                return Err(err);
                                            }
                                        },
                                        _ =>
                                        {
                                            assert!(false);
                                        }
                                    }
                                }
                            }
                            b"name" =>
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
            b"instances" =>
            {
                p.expect_char(':');
                p.expect_char('[');

                let mut list_continue = true;
                while list_continue
                {
                    let mut instance = lp::Instance::default();
                    let default_transform = conversion * lp::Mat3x4::IDENTITY;
                    instance.transpose_inverse_transform = default_transform.inverse().transpose();

                    p.expect_char('{');

                    let mut dict_continue = true;
                    while dict_continue
                    {
                        let strlit = p.next_strlit();
                        match strlit
                        {
                            b"name" =>
                            {
                                p.expect_char(':');
                                let name = p.next_strlit();
                            },
                            b"frame" =>
                            {
                                p.expect_char(':');
                                let transform = conversion * p.parse_mat3x4f();
                                instance.transpose_inverse_transform = transform.inverse().transpose();
                            },
                            b"material" =>
                            {
                                p.expect_char(':');
                                instance.mat_idx = p.parse_u32();
                            },
                            b"shape" =>
                            {
                                p.expect_char(':');
                                instance.mesh_idx = p.parse_u32();
                            },
                            _ => {},
                        }

                        dict_continue = p.next_list_el();
                    }

                    p.expect_char('}');
                    scene.instances.push(instance);

                    list_continue = p.next_list_el();
                }

                p.expect_char(']');
            },
            b"asset" =>
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
        if p.found_error { return Err(LoadError::InvalidJson(p.error_str)); }
    }

    p.expect_char('}');

    if p.found_error { return Err(LoadError::InvalidJson(p.error_str)); }

    // Load textures.
    for info in &tex_load_infos
    {
        // TODO: This is not needed anymore because we do the srgb->linear conversion in the shader.
        //let mut uses = 0;
        //if info.used_for_color { uses += 1; }
        //if info.used_for_data  { uses += 1; }

        let path = std::path::Path::new(&info.path);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_hdr = matches!(ext.to_lowercase().as_str(), "hdr" | "exr");
        let res = load_texture(device, queue, &info.path, is_hdr);
        if let Err(err) = res {
            return Err(err.into());
        }

        let view = res.as_ref().unwrap().create_view(&Default::default());
        textures.push(res.unwrap());
        texture_views.push(view);
        push_asset(&mut samplers, lp::create_linear_sampler(device));
    }

    // Fill in environment data.
    let mut environment_infos = Vec::<lp::EnvMapInfo>::with_capacity(scene.environments.len());
    for env in &scene.environments
    {
        if env.emission_tex_idx == lp::SENTINEL_IDX
        {
            // Functions always expect environment infos (TODO?)
            environment_infos.push(lp::EnvMapInfo {
                data: vec!(lp::Vec4::ones()),
                width: 1,
                height: 1,
            });
            continue;
        }

        let info = &tex_load_infos[env.emission_tex_idx as usize];

        //let mut uses = 0;
        //if info.used_for_color { uses += 1; }
        //if info.used_for_data  { uses += 1; }
        // This shouldn't happen but it does, I guess we'll just load in SRGB.
        // assert!(uses <= 1);

        let path = std::path::Path::new(&info.path);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let is_hdr = matches!(ext.to_lowercase().as_str(), "hdr" | "exr");
        let res = load_texture_cpu(&path);
        if let Err(err) = res {
            return Err(err.into());
        }

        if let Ok(tex_cpu) = res {
            environment_infos.push(lp::EnvMapInfo {
                data: tex_cpu.data,
                width: tex_cpu.width,
                height: tex_cpu.height,
            });
        }
    }

    lp::validate_scene(&scene, textures.len() as u32, samplers.len() as u32);

    let scene_gpu = lp::build_accel_structures_and_upload(device, queue, &scene, textures, texture_views, samplers, &environment_infos, build_both_bvhs);
    return Ok((scene_gpu, scene_cams));
}

fn parse_material_yocto_v24(p: &mut Parser, tex_load_infos: &mut Vec<TextureLoadInfo>) -> lp::Material
{
    let mut mat = lp::Material::default();

    let mut dict_continue = true;
    while dict_continue
    {
        let strlit = p.next_strlit();
        match strlit
        {
            b"name" =>
            {
                p.expect_char(':');
                let name = p.next_strlit();
            }
            b"color" =>
            {
                p.expect_char(':');

                let color = p.parse_vec3f();
                mat.color = lp::Vec4 { x: color.x, y: color.y, z: color.z, w: mat.color.w };
                mat.color.w = 1.0;
            },
            b"emission" =>
            {
                p.expect_char(':');

                let color = p.parse_vec3f();
                mat.emission = lp::Vec4 { x: color.x, y: color.y, z: color.z, w: mat.emission.w };
            },
            b"scattering" =>
            {
                p.expect_char(':');

                let color = p.parse_vec3f();
                mat.scattering = lp::Vec4 { x: color.x, y: color.y, z: color.z, w: mat.scattering.w }
            },
            b"roughness" =>
            {
                p.expect_char(':');
                mat.roughness = p.parse_f32();
            },
            b"metallic" =>
            {
                p.expect_char(':');
                mat.metallic = p.parse_f32();
            },
            b"ior" =>
            {
                p.expect_char(':');
                mat.ior = p.parse_f32();
            },
            b"scanisotropy" =>
            {
                p.expect_char(':');
                mat.sc_anisotropy = p.parse_f32();
            },
            b"trdepth" =>
            {
                p.expect_char(':');
                mat.tr_depth = p.parse_f32();
            },
            b"opacity" =>
            {
                p.expect_char(':');
                mat.color.w = p.parse_f32();
            },
            b"type" =>
            {
                p.expect_char(':');
                let mat_type_str = p.next_strlit();
                match mat_type_str
                {
                    b"matte" => mat.mat_type = lp::MaterialType::Matte,
                    b"glossy" => mat.mat_type = lp::MaterialType::Glossy,
                    b"reflective" => mat.mat_type = lp::MaterialType::Reflective,
                    b"transparent" => mat.mat_type = lp::MaterialType::Transparent,
                    b"refractive" => mat.mat_type = lp::MaterialType::Refractive,
                    b"subsurface" => mat.mat_type = lp::MaterialType::Subsurface,
                    b"volume" => mat.mat_type = lp::MaterialType::Volumetric,
                    b"gltfpbr" => mat.mat_type = lp::MaterialType::GltfPbr,
                    _ => {}
                }
            },
            b"color_tex" =>
            {
                p.expect_char(':');
                mat.color_tex_idx = p.parse_u32();
                if mat.color_tex_idx != lp::SENTINEL_IDX
                {
                    grow_vec(tex_load_infos, mat.color_tex_idx as usize + 1);
                    tex_load_infos[mat.color_tex_idx as usize].used_for_color = true;
                }
            },
            b"emission_tex" =>
            {
                p.expect_char(':');
                mat.emission_tex_idx = p.parse_u32();
                if mat.emission_tex_idx != lp::SENTINEL_IDX
                {
                    grow_vec(tex_load_infos, mat.emission_tex_idx as usize + 1);
                    tex_load_infos[mat.emission_tex_idx as usize].used_for_color = true;
                }
            },
            b"roughness_tex" =>
            {
                p.expect_char(':');
                mat.roughness_tex_idx = p.parse_u32();
                if mat.roughness_tex_idx != lp::SENTINEL_IDX
                {
                    grow_vec(tex_load_infos, mat.roughness_tex_idx as usize + 1);
                    tex_load_infos[mat.roughness_tex_idx as usize].used_for_data = true;
                }
            },
            b"scattering_tex" =>
            {
                p.expect_char(':');
                mat.scattering_tex_idx = p.parse_u32();
                if mat.scattering_tex_idx != lp::SENTINEL_IDX
                {
                    grow_vec(tex_load_infos, mat.scattering_tex_idx as usize + 1);
                    tex_load_infos[mat.scattering_tex_idx as usize].used_for_data = true;
                }
            },
            b"normal_tex" =>
            {
                p.expect_char(':');
                mat.normal_tex_idx = p.parse_u32();
                if mat.normal_tex_idx != lp::SENTINEL_IDX
                {
                    grow_vec(tex_load_infos, mat.normal_tex_idx as usize + 1);
                    tex_load_infos[mat.normal_tex_idx as usize].used_for_data = true;
                }
            },
            _ => {}
        }

        dict_continue = p.next_list_el();
    }

    return mat;
}

// Utility functions used for parsing any simple ASCII textual format.
struct Parser<'a>
{
    pub whole_file: &'a [u8],
    pub buf: &'a [u8],
    pub found_error: bool,
    pub error_str: String,
}

impl<'a> Parser<'a>
{
    fn new(buf: &'a [u8]) -> Self
    {
        return Self {
            buf,
            error_str: Default::default(),
            whole_file: buf,
            found_error: false
        };
    }

    // Returns strlit without "" chars.
    fn next_strlit(&mut self) -> &[u8]
    {
        self.eat_whitespace();

        self.expect_char('\"');

        if self.found_error { return &self.buf[0..0]; }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| b == b'"')
            .unwrap_or(self.buf.len());
        let strlit = &self.buf[0..trimmed_start];

        self.buf = &self.buf[trimmed_start..];
        self.expect_char('"');
        if self.found_error { return &self.buf[0..0]; }

        return strlit;
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
            self.parse_error(String::from("Expecting a new line, file ended prematurely."));
        }
    }

    fn next_ident(&mut self) -> &[u8]
    {
        self.eat_whitespace();

        if self.buf.is_empty() { return &self.buf[0..0]; }
        if !u8::is_ascii_alphabetic(&self.buf[0]) && self.buf[0] != b'_' { return &self.buf[0..0]; }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| !(u8::is_ascii_alphabetic(&b) || u8::is_ascii_digit(&b) || b == b'_'))
            .unwrap_or(self.buf.len());
        let token = &self.buf[0..trimmed_start];

        self.buf = &self.buf[trimmed_start..];
        if self.found_error { return &self.buf[0..0]; }

        return token;
    }

    fn expect_ident(&mut self, ident: &[u8])
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

        if token != ident { self.parse_error(format!("Expected {}, got {}.", String::from_utf8_lossy(ident), String::from_utf8_lossy(token))); }
    }

    fn peek_ident(&mut self) -> &[u8]
    {
        self.eat_whitespace();

        if self.buf.is_empty() { return &self.buf[0..0]; }
        if !u8::is_ascii_alphabetic(&self.buf[0]) && self.buf[0] != b'_' { return &self.buf[0..0]; }

        let trimmed_start = self.buf
            .iter()
            .position(|&b| !(u8::is_ascii_alphabetic(&b) || u8::is_ascii_digit(&b) || b == b'_'))
            .unwrap_or(self.buf.len());
        let token = &self.buf[0..trimmed_start];

        if self.found_error { return &self.buf[0..0]; }

        return token;
    }

    fn expect_char(&mut self, c: char)
    {
        self.eat_whitespace();
        if self.buf.len() == 0 { self.parse_error(format!("Expected {}, got EOF.", c)); return; }
        if !c.is_ascii()       { self.parse_error(format!("Expected {}, got unknown.", c)); return; }
        if self.buf[0] != c as u8 { self.parse_error(format!("Expected {}, got {}.", c, self.buf[0] as char)); return; }
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

    fn parse_mat3x4f(&mut self) -> lp::Mat3x4
    {
        let mut res = lp::Mat3x4::IDENTITY;

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

    fn parse_bool(&mut self) -> bool
    {
        self.eat_whitespace();

        let ident = self.next_ident();
        match ident
        {
            b"true" => { return true; }
            b"false" => { return false; }
            _ => { self.parse_error(String::from("Expected bool.")); return false; }
        }
    }

    fn parse_f32(&mut self) -> f32
    {
        self.eat_whitespace();

        if self.buf.is_empty()
        {
            self.parse_error(String::from("Expected f32, got end of file."));
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
            self.parse_error(String::from("Expected f32."));
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
                    self.parse_error(String::from("Expected f32."));
                    return 0.0
                }
            },
            Err(_) =>
            {
                self.parse_error(String::from("Expected f32."));
                return 0.0
            }
        };
    }

    fn parse_u32(&mut self) -> u32
    {
        self.eat_whitespace();

        if self.buf.is_empty()
        {
            self.parse_error(String::from("Expected u32, got end of file."));
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
            self.parse_error(String::from("Expected u32."));
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
                    self.parse_error(String::from("Expected u32."));
                    return 0
                }
            },
            Err(_) =>
            {
                self.parse_error(String::from("Expected u32."));
                return 0
            }
        };
    }

    pub fn parse_error(&mut self, msg: String)
    {
        if self.found_error { return; }
        self.found_error = true;

        let (up_to_error, _) = exclude_subslice(self.whole_file, self.buf);
        let line_num = up_to_error.iter().filter(|&&b| b == b'\n').count() + 1;
        self.error_str = format!("Error on line {}: {}", line_num, msg);

        fn exclude_subslice<'a>(a: &'a [u8], b: &'a [u8]) -> (&'a [u8], &'a [u8])
        {
            let a_ptr = a.as_ptr() as usize;
            let b_ptr = b.as_ptr() as usize;

            assert!(b_ptr >= a_ptr);
            assert!(b_ptr + b.len() <= a_ptr + a.len());

            let start = (b_ptr - a_ptr) as usize;
            let end = start + b.len();

            (&a[..start], &a[end..])
        }
    }
}

// .PLY Format

#[derive(Debug)]
pub enum LoadError
{
    Io(std::io::Error),
    ImageErr(image::ImageError),
    InvalidPly,
    InvalidJson(String),
}

impl std::fmt::Display for LoadError
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result
    {
        match self
        {
            LoadError::Io(err)             => write!(f, "I/O Error: {}", err),
            LoadError::ImageErr(err)       => write!(f, "Image Loading Error: {}", err),
            LoadError::InvalidPly          => write!(f, "Invalid Mesh Error."),
            LoadError::InvalidJson(string) => write!(f, "Invalid JSON: {}", string)
        }
    }
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

// NOTE: Not a complete ply parser, it parses/uses only the features
// that are expected to be used from the yoctogl 2.4 format.
fn load_mesh_ply(path: &std::path::Path, scene: &mut lp::SceneCPU) -> Result<u32, LoadError>
{
    assert_eq!(scene.verts_pos_array.len(), scene.mesh_infos.len());
    assert_eq!(scene.mesh_infos.len(), scene.indices_array.len());

    // Parse header
    let ply = std::fs::read(path)?;
    let mut p = Parser::new(&ply);
    let header_continue = true;

    let mut x_buf  = Buffer::default();
    let mut y_buf  = Buffer::default();
    let mut z_buf  = Buffer::default();
    let mut nx_buf = Buffer::default();
    let mut ny_buf = Buffer::default();
    let mut nz_buf = Buffer::default();
    let mut u_buf  = Buffer::default();
    let mut v_buf  = Buffer::default();
    let mut r_buf  = Buffer::default();
    let mut g_buf  = Buffer::default();
    let mut b_buf  = Buffer::default();
    let mut a_buf  = Buffer::default();
    let indices = Vec::<u32>::default();
    let mut num_verts = 0;
    let mut num_faces = 0;
    let mut vert_size = 0;
    let indices_offset = 0;

    p.expect_ident(b"ply");

    loop
    {
        let ident = p.next_ident();
        match ident
        {
            b"format" =>
            {
                p.expect_ident(b"binary_little_endian");
                let major_version = p.parse_u32();
                p.expect_char('.');
                let minor_version = p.parse_u32();

                if major_version != 1 || minor_version != 0
                {
                    assert!(false);
                }
            }
            b"comment" =>
            {
                p.go_to_next_line();
            }
            b"element" =>
            {
                let el_name = p.next_ident();
                if el_name == b"vertex"
                {
                    num_verts = p.parse_u32();

                    let mut offset = 0;

                    while p.peek_ident() == b"property"
                    {
                        p.next_ident();

                        let mut prop_size = 0;
                        let prop_type_str = p.next_ident();
                        if prop_type_str == b"float"
                        {
                            prop_size = 4;
                        }

                        let prop_name = p.next_ident();
                        match prop_name
                        {
                            b"x"  => { x_buf.present  = true; x_buf.offset  = offset; },
                            b"y"  => { y_buf.present  = true; y_buf.offset  = offset; },
                            b"z"  => { z_buf.present  = true; z_buf.offset  = offset; },
                            b"nx" => { nx_buf.present = true; nx_buf.offset = offset; },
                            b"ny" => { ny_buf.present = true; ny_buf.offset = offset; },
                            b"nz" => { nz_buf.present = true; nz_buf.offset = offset; },
                            b"u"  => { u_buf.present  = true; u_buf.offset  = offset; },
                            b"s"  => { u_buf.present  = true; u_buf.offset  = offset; },  // NOTE: Alternative name for "u"
                            b"v"  => { v_buf.present  = true; v_buf.offset  = offset; },
                            b"t"  => { v_buf.present  = true; v_buf.offset  = offset; },  // NOTE: Alternative name for "v"
                            b"red"   => { r_buf.present = true; r_buf.offset = offset; },
                            b"green" => { g_buf.present = true; g_buf.offset = offset; },
                            b"blue"  => { b_buf.present = true; b_buf.offset = offset; },
                            b"alpha" => { a_buf.present = true; a_buf.offset = offset; },
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
                    r_buf.stride = total_size;
                    g_buf.stride = total_size;
                    b_buf.stride = total_size;
                    a_buf.stride = total_size;
                    vert_size = total_size;
                }
                else if el_name == b"face"
                {
                    num_faces = p.parse_u32();
                    p.expect_ident(b"property");
                    p.expect_ident(b"list");
                    p.expect_ident(b"uchar");
                    let ident = p.next_ident();
                    if ident != b"uint" && ident != b"int" {
                        return Err(LoadError::InvalidPly);
                    }
                    p.expect_ident(b"vertex_indices");
                }
            }
            b"end_header" => { p.go_to_next_line(); break; }
            _ => { p.parse_error(String::from("Unknown .PLY field.")); }
        }

        if p.found_error {
            return Err(LoadError::InvalidPly);
        }
    }

    if p.found_error {
        return Err(LoadError::InvalidPly);
    }

    if !x_buf.present || !y_buf.present || !z_buf.present {
        return Err(LoadError::InvalidPly);
    }

    let mut mesh_info = lp::MeshInfo::default();

    let verts_pos = buf_extract_vec3_as_vec4(p.buf, &x_buf, &y_buf, &z_buf, num_verts);
    if nx_buf.present || ny_buf.present || nz_buf.present
    {
        assert!(nx_buf.present && ny_buf.present && nz_buf.present);
        let normals = buf_extract_vec3_as_vec4(p.buf, &nx_buf, &ny_buf, &nz_buf, num_verts);

        scene.verts_normal_array.push(normals);
        let normals_idx = scene.verts_normal_array.len() - 1;
        mesh_info.normals_buf_idx = normals_idx as u32;
    }
    if u_buf.present || v_buf.present
    {
        assert!(u_buf.present && v_buf.present);

        let mut texcoords = buf_extract_vec2(p.buf, &u_buf, &v_buf, num_verts);

        for texcoord in &mut texcoords
        {
            // WGPU Convention is +=right,down and tipically it's +=right,up
            texcoord.y = 1.0 - texcoord.y;
        }

        scene.verts_texcoord_array.push(texcoords);
        let texcoords_idx = scene.verts_texcoord_array.len() - 1;
        mesh_info.texcoords_buf_idx = texcoords_idx as u32;
    }
    if r_buf.present || g_buf.present || b_buf.present || a_buf.present
    {
        assert!(r_buf.present && g_buf.present && b_buf.present);
        let colors = buf_extract_vec4(p.buf, &r_buf, &g_buf, &b_buf, &a_buf, num_verts);
        scene.verts_color_array.push(colors);
        let colors_idx = scene.verts_color_array.len() - 1;
        mesh_info.colors_buf_idx = colors_idx as u32;
    }

    let indices = ply_extract_indices(&p.buf[(num_verts * vert_size as u32) as usize..], num_faces);

    // Check indices
    for idx in &indices
    {
        if *idx as usize >= verts_pos.len() {
            return Err(LoadError::InvalidPly);
        }
    }

    scene.mesh_infos.push(mesh_info);
    scene.verts_pos_array.push(verts_pos);
    scene.indices_array.push(indices.clone());

    return Ok((scene.mesh_infos.len() - 1) as u32);
}

// Extracts 3 elements from a buffer, and fills in 0 as the last element in the vec4.
fn buf_extract_vec3_as_vec4(buf: &[u8], x: &Buffer, y: &Buffer, z: &Buffer, num: u32) -> Vec<lp::Vec4>
{
    assert!(x.present && y.present && z.present);
    let mut res = vec![lp::Vec4::default(); num as usize];
    for i in 0..num as usize
    {
        let offset = x.offset + i * x.stride;
        res[i].x = extract_f32(buf, offset);
    }
    for i in 0..num as usize
    {
        let offset = y.offset + i * y.stride;
        res[i].y = extract_f32(buf, offset);
    }
    for i in 0..num as usize
    {
        let offset = z.offset + i * z.stride;
        res[i].z = extract_f32(buf, offset);
    }

    return res;
}

fn buf_extract_vec4(buf: &[u8], x: &Buffer, y: &Buffer, z: &Buffer, w: &Buffer, num: u32) -> Vec<lp::Vec4>
{
    assert!(x.present && y.present && z.present && w.present);
    let mut res = vec![lp::Vec4::default(); num as usize];
    for i in 0..num as usize
    {
        let offset = x.offset + i * x.stride;
        res[i].x = extract_f32(buf, offset);
    }
    for i in 0..num as usize
    {
        let offset = y.offset + i * y.stride;
        res[i].y = extract_f32(buf, offset);
    }
    for i in 0..num as usize
    {
        let offset = z.offset + i * z.stride;
        res[i].z = extract_f32(buf, offset);
    }
    for i in 0..num as usize
    {
        let offset = w.offset + i * w.stride;
        res[i].w = extract_f32(buf, offset);
    }

    return res;
}

fn buf_extract_vec2(buf: &[u8], x: &Buffer, y: &Buffer, num: u32) -> Vec<lp::Vec2>
{
    assert!(x.present && y.present);
    let mut res = vec![lp::Vec2::default(); num as usize];
    for i in 0..num as usize
    {
        let offset = x.offset + i * x.stride;
        res[i].x = extract_f32(buf, offset);
    }
    for i in 0..num as usize
    {
        let offset = y.offset + i * y.stride;
        res[i].y = extract_f32(buf, offset);
    }

    return res;
}

fn ply_extract_indices(buf: &[u8], num_faces: u32) -> Vec<u32>
{
    let mut indices = Vec::<u32>::new();

    let mut s = Serializer { buf: buf };
    for i in 0..num_faces
    {
        let num_indices = s.read_u8();
        assert!(num_indices >= 3);

        let     idx0 = s.read_u32();
        let mut idx1 = s.read_u32();
        let mut idx2 = s.read_u32();

        indices.push(idx0);
        indices.push(idx1);
        indices.push(idx2);

        for j in 0..num_indices-3
        {
            idx1 = idx2;
            idx2 = s.read_u32();
            indices.push(idx0);
            indices.push(idx1);
            indices.push(idx2);
        }
    }

    return indices;
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

/// Transfers texture from GPU to CPU, into a buffer of uniform type (RGBAF32).
/// Supports rgba8_unorm, rgba16f.
pub fn download_texture(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<lp::Vec4>
{
    let width = texture.size().width;
    let height = texture.size().height;
    let format = texture.format();

    // How many bytes per pixel?
    let bytes_per_pixel = match format {
        wgpu::TextureFormat::Rgba8Unorm |
        wgpu::TextureFormat::Rgba8UnormSrgb => 4,   // u8 * 4
        wgpu::TextureFormat::Rgba16Float => 8,      // f16 * 4
        wgpu::TextureFormat::Rgba32Float => 16,     // f32 * 4
        _ => panic!("Unsupported texture format: {:?}", format),
    };

    // Bytes per row must be padded to 256 bytes for GPU copy
    let unpadded_bytes_per_row = bytes_per_pixel * width as usize;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row = ((unpadded_bytes_per_row + align - 1) / align) * align;

    // Create readback buffer
    let buffer_size = padded_bytes_per_row as u64 * height as u64;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("download buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Encode copy from texture -> buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("download encoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row as u32),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));

    // Map and block until ready (synchronous style)
    {
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).expect("Device wait failed.");
        rx.recv().unwrap().unwrap();
    }

    // Read the data
    let view = buffer.slice(..).get_mapped_range();
    let mut pixels = Vec::with_capacity((width * height) as usize);

    for y in 0..height as usize {
        let row = &view[y * padded_bytes_per_row..y * padded_bytes_per_row + unpadded_bytes_per_row];
        match format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                for px in row.chunks_exact(4) {
                    let r = px[0] as f32 / 255.0;
                    let g = px[1] as f32 / 255.0;
                    let b = px[2] as f32 / 255.0;
                    let a = px[3] as f32 / 255.0;
                    pixels.push(lp::Vec4 { x: r, y: g, z: b, w: a });
                }
            }
            wgpu::TextureFormat::Rgba16Float => {
                for px in row.chunks_exact(8) {
                    let r = half::f16::from_bits(u16::from_le_bytes([px[0], px[1]])).to_f32();
                    let g = half::f16::from_bits(u16::from_le_bytes([px[2], px[3]])).to_f32();
                    let b = half::f16::from_bits(u16::from_le_bytes([px[4], px[5]])).to_f32();
                    let a = half::f16::from_bits(u16::from_le_bytes([px[6], px[7]])).to_f32();
                    pixels.push(lp::Vec4 { x: r, y: g, z: b, w: a });
                }
            }
            wgpu::TextureFormat::Rgba32Float => {
                for px in row.chunks_exact(16) {
                    let r = f32::from_le_bytes([px[0], px[1], px[2], px[3]]);
                    let g = f32::from_le_bytes([px[4], px[5], px[6], px[7]]);
                    let b = f32::from_le_bytes([px[8], px[9], px[10], px[11]]);
                    let a = f32::from_le_bytes([px[12], px[13], px[14], px[15]]);
                    pixels.push(lp::Vec4 { x: r, y: g, z: b, w: a });
                }
            }
            _ => unreachable!(),
        }
    }

    drop(view);
    buffer.unmap();

    return pixels;
}

pub struct TextureCPU
{
    pub data: Vec::<lp::Vec4>,
    pub width: u32,
    pub height: u32
}
pub fn load_texture_cpu(path: &std::path::Path) -> Result<TextureCPU, image::ImageError>
{
    use image::GenericImageView;

    let img = image::open(path).expect("Failed to load image");
    let dimensions = img.dimensions();

    let rgba_f32 = img.to_rgba32f();
    let rgba = rgba32f_to_rgba16f(&rgba_f32);

    let vec4_data: Vec<lp::Vec4> = rgba_f32
        .chunks_exact(4)
        .map(|chunk| lp::Vec4 {
            x: chunk[0],
            y: chunk[1],
            z: chunk[2],
            w: chunk[3],
        })
        .collect();

    return Ok(TextureCPU {
        data: vec4_data,
        width: rgba_f32.width(),
        height: rgba_f32.height(),
    });
}

/// Detects file format from its extension. Supports rgba8_unorm, rgba16f.
/// NOTE: Currently drops alpha.
pub fn save_texture(device: &wgpu::Device, queue: &wgpu::Queue, path: &std::path::Path, texture: &wgpu::Texture) -> Result<(), image::ImageError>
{
    let format = texture.format();
    assert!(format == wgpu::TextureFormat::Rgba8Unorm || format == wgpu::TextureFormat::Rgba16Float,
            "Unsupported texture format for save_texture!");

    let width = texture.size().width;
    let height = texture.size().height;

    let bytes_per_pixel = match format {
        wgpu::TextureFormat::Rgba8Unorm => 4,
        wgpu::TextureFormat::Rgba16Float => 8,
        _ => { panic!("unsupported texture format"); }
    };
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let row_size = align_up(width * bytes_per_pixel, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = (row_size * height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Copy texture into transfer buffer.
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Texture Copy Encoder"),
    });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &output_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(row_size),
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

    // Map synchronously.
    let buffer_slice = output_buffer.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::wait_indefinitely()).expect("Device wait failed.");

    let data = buffer_slice.get_mapped_range();

    match format
    {
        wgpu::TextureFormat::Rgba8Unorm =>
        {
            let mut data_no_alpha = Vec::with_capacity((width * height * 3) as usize);
            for row in 0..height
            {
                let start = (row * row_size) as usize;
                let end = start + (width * bytes_per_pixel) as usize;
                let row_data = &data[start..end];

                for px in row_data.chunks_exact(4)
                {
                    data_no_alpha.push(px[0]);
                    data_no_alpha.push(px[1]);
                    data_no_alpha.push(px[2]);
                }
            }

            let img = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(width, height, data_no_alpha).unwrap();
            let res = img.save(path);
            return res;
        },
        wgpu::TextureFormat::Rgba16Float =>
        {
            assert!(data.len() % 2 == 0, "Data must be aligned to 2 bytes!");
            let data_f16: &[half::f16] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const half::f16, data.len() / 2)
            };

            let mut data_f32_no_alpha = Vec::<f32>::with_capacity((width * height * 3) as usize);
            for row in 0..height
            {
                let start = (row * row_size) as usize;
                let end = start + (width * bytes_per_pixel) as usize;
                let row_data = &data_f16[start/2..end/2];  // Convert from number of bytes to number of f16.

                for px in row_data.chunks_exact(4)
                {
                    data_f32_no_alpha.push(px[0].into());
                    data_f32_no_alpha.push(px[1].into());
                    data_f32_no_alpha.push(px[2].into());
                }
            }
            let img = image::ImageBuffer::<image::Rgb<f32>, _>::from_raw(width, height, data_f32_no_alpha).unwrap();
            let res = img.save(path);
            return res;
        },
        _ => { panic!("Unsupported texture type."); }
    }
}

fn align_up(value: u32, align: u32) -> u32
{
    assert!(0 == (align & (align - 1)), "Must align to a power of two");
    return (value + (align - 1)) & !(align - 1);
}
