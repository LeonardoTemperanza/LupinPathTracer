
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

//use ::egui::FontDefinitions;

pub use lupin as lp;
pub use lupin_loader as lpl;
use lupin::wgpu as wgpu;

fn main()
{
    if std::fs::exists("scenes").is_err() {
        panic!("It appears that the \"scenes\" directory is missing (it's not directly visible from the current working directory).");
    }

    let output_dir_name = "output";
    if std::fs::exists(output_dir_name).is_err() {
        std::fs::create_dir("output").expect("Could not create \"output\" folder to put renders into.");
    }

    let scene_names = [
        "bathroom1",
        "bistroexterior",
        "bistrointerior",
        "car2",
        "classroom",
        "coffee",
        "ecosys",
        "hairball",
        "junkshop",
        "lonemonk",
        "sanmiguel"
    ];

    let (device, queue, adapter) = lp::init_default_wgpu_context_no_window();
    let tonemap_resources = lp::build_tonemap_resources(&device);

    let max_bounces = 8;
    let num_samples_per_pixel = 5;
    let pathtrace_resources = lp::build_pathtrace_resources(&device, &Default::default());

    for scene_name in scene_names
    {
        let mut path_json_buf = std::path::PathBuf::new();
        path_json_buf.push("scenes");
        path_json_buf.push(scene_name);
        path_json_buf.set_extension("json");

        let path_json = path_json_buf.as_path();

        if std::fs::exists(path_json).is_err()
        {
            eprintln!("Scene \"{}\" not found.", scene_name);
            continue;
        }

        let (scene, cameras) = lpl::load_scene_yoctogl_v24(path_json, &device, &queue).unwrap();
        let num_accums = 200;

        if cameras.len() <= 0
        {
            eprintln!("There are no cameras in scene \"{}\".", scene_name);
        }

        let camera = &cameras[0];

        let (width, height) = compute_dimensions_for_1080p(camera.params.aspect);

        let mut output_tex = DoubleBufferedTexture::create(&device, &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST |
                   wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[]
        });

        print!("Scene \"{}\": ", scene_name);
        for accum_idx in 0..num_accums
        {
            let desc = lp::PathtraceDesc {
                scene: &scene,
                render_target: output_tex.front(),
                resources: &pathtrace_resources,
                accum_params: &lp::AccumulationParams {
                    prev_frame: Some(output_tex.back()),
                    accum_counter: accum_idx,
                },
                tile_params: None,
                camera_params: camera.params,
                camera_transform: camera.transform,
            };
            lp::pathtrace_scene(&device, &queue, &desc, Default::default(), None);
            output_tex.flip();
        }
        output_tex.flip();  // Final image is now in front.
        println!("Done.");

        // Save image
        let tonemapped = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: width as u32, height: height as u32, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING |
                   wgpu::TextureUsages::TEXTURE_BINDING |
                   wgpu::TextureUsages::RENDER_ATTACHMENT |
                   wgpu::TextureUsages::COPY_SRC,
            view_formats: &[]
        });

        lp::tonemap_and_fit_aspect(&device, &queue, &lp::TonemapDesc {
            resources: &tonemap_resources,
            hdr_texture: output_tex.front(),
            render_target: &tonemapped,
            tonemap_params: &Default::default(),
        }, None);

        let mut output_path_buf = std::path::PathBuf::new();
        output_path_buf.push(output_dir_name);
        output_path_buf.push(scene_name);
        output_path_buf.set_extension("png");
        let output_path = output_path_buf.as_path();

        let res = lpl::save_texture(&device, &queue, output_path, &tonemapped);
    }
}

fn compute_dimensions_for_1080p(aspect: f32) -> (u32, u32)
{
    if aspect > 1.0 {  // Taller than wide
        return ((1920.0 / aspect) as u32, 1920);
    } else {  // Wider than tall
        return (1920, (1920.0 * aspect) as u32);
    }
}

// TODO: I should probably add this to the main library?
pub struct DoubleBufferedTexture
{
    pub textures: [wgpu::Texture; 2],
    pub front_idx: usize,
    pub back_idx: usize
}

impl<'a> DoubleBufferedTexture
{
    pub fn create(device: &wgpu::Device, desc: &wgpu::TextureDescriptor) -> DoubleBufferedTexture
    {
        return Self {
            textures: [
                device.create_texture(desc),
                device.create_texture(desc),
            ],
            front_idx: 0,
            back_idx: 1,
        }
    }

    pub fn front(&'a self) -> &'a wgpu::Texture
    {
        return &self.textures[self.front_idx];
    }

    pub fn back(&'a self) -> &'a wgpu::Texture
    {
        return &self.textures[self.back_idx];
    }

    pub fn copy_front_to_back(&self, device: &wgpu::Device, queue: &wgpu::Queue)
    {
        assert!(self.textures[0].format() == self.textures[1].format());

        let format = self.textures[0].format();
        let blitter = wgpu::util::TextureBlitter::new(device, format);
        let mut encoder = device.create_command_encoder(&Default::default());
        let src = self.textures[self.front_idx].create_view(&Default::default());
        let dst = self.textures[self.back_idx].create_view(&Default::default());
        blitter.copy(device, &mut encoder, &src, &dst);
        queue.submit(Some(encoder.finish()));
    }

    pub fn flip(&mut self)
    {
        let tmp = self.front_idx;
        self.front_idx = self.back_idx;
        self.back_idx = tmp;
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32)
    {
        resize_texture(device, &mut self.textures[0], width, height);
        resize_texture(device, &mut self.textures[1], width, height);
    }
}

fn resize_texture(device: &wgpu::Device, texture: &mut wgpu::Texture, new_width: u32, new_height: u32)
{
    let desc = wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: new_width, height: new_height, depth_or_array_layers: 1 },
        mip_level_count: texture.mip_level_count(),
        sample_count: texture.sample_count(),
        dimension: texture.dimension(),
        format: texture.format(),
        usage: texture.usage(),
        view_formats: &[]
    };

    texture.destroy();
    *texture = device.create_texture(&desc);
}
