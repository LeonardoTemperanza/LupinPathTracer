
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

//use ::egui::FontDefinitions;

pub use lupin as lp;
pub use lupin_loader as lpl;
use lupin::wgpu as wgpu;

pub struct Scene
{
    pub name: &'static str,
    pub samples: u32,
    pub max_radiance: f32,
    pub credits: &'static str,
}

pub struct RenderInfo
{
    pub output_file: String,
    pub scene_idx: usize,
    pub time: f32,
    pub res_x: u32,
    pub res_y: u32,
}

fn main()
{
    let scenes_dir = "lupin_scenes";
    if std::fs::exists(scenes_dir).is_err() {
        panic!("It appears that the \"{}\" directory is missing (it's not directly visible from the current working directory).", scenes_dir);
    }

    let output_dir_name = "output";
    let res = std::fs::exists(output_dir_name);
    if res.is_err() || !res.unwrap() {
        std::fs::create_dir(output_dir_name).expect("Could not create output folder to put renders into.");
    }

    let scenes = [
        Scene { name: "landscape", samples: 500, max_radiance: 100.0, credits: "Scene by Jan-Walter Schliep, Burak Kahraman, Timm Dapper: \\link{pbrt.org/scenes-v3.html}" },
        Scene { name: "lonemonk", samples: 2000, max_radiance: 100.0, credits: "Scene by Carlo Bergonzini: \\link{www.blender.org/download/demo-files/}" },
        Scene { name: "coffee", samples: 2000, max_radiance: 100.0, credits: "Scene by \"Cekuhnen\": \\link{benedikt-bitterli.me/resources}" },
        Scene { name: "classroom", samples: 2000, max_radiance: 100.0, credits: "Scene by Christophe Seux: \\link{www.blender.org/download/demo-files/}" },
        Scene { name: "bistroexterior", samples: 1000, max_radiance: 10.0, credits: "Scene by Amazon Lumberyard: \\link{casual-effects.com/data}" },
        Scene { name: "bistrointerior", samples: 1000, max_radiance: 100.0, credits: "Scene by Amazon Lumberyard: \\link{casual-effects.com/data}" },
        Scene { name: "junkshop", samples: 2000, max_radiance: 100.0, credits: "Model by Alex Treviño, concept by Anaïs Maamar: \\link{www.blender.org/download/demo-files/}" },
        Scene { name: "bathroom1", samples: 6000, max_radiance: 100.0, credits: "Scene by \"Mareck\": \\link{https://benedikt-bitterli.me/resources}" },
        //Scene { name: "car2", samples: 1000, max_radiance: 100.0, credits: "Scene by \"Thecali\": \\link{https://benedikt-bitterli.me/resources}" },
        //Scene { name: "ecosys", samples: 1000, max_radiance: 100.0, credits: "Scene by \"Deussen et al.\": \\link{pbrt.org/scenes-v3.html}" },
        Scene { name: "sanmiguel", samples: 1000, max_radiance: 20.0, credits: "Scene by Guillermo M. Leal Llaguno: \\link{https://benedikt-bitterli.me/resources}" },
    ];

    let (device, queue, adapter) = lp::init_default_wgpu_context_no_window();
    let tonemap_resources = lp::build_tonemap_resources(&device);

    let mut render_infos = Vec::new();
    let mut sw_render_times = Vec::new();
    let mut scene_stats = Vec::new();
    for hw_rt in [false, true]
    {
        let exposure = 0.0;

        let max_bounces = 8;
        let num_samples_per_pixel = if hw_rt { 5 } else { 1 };
        let pathtrace_resources = lp::build_pathtrace_resources(&device, &lp::BakedPathtraceParams {
            with_runtime_checks: false,
            max_bounces: 8,
            samples_per_pixel: num_samples_per_pixel,
        });

        if hw_rt { println!("HW RT:"); } else { println!("SW_RT:"); }

        for (scene_idx, scene) in scenes.iter().enumerate()
        {
            if !hw_rt && scene.name == "sanmiguel"
            {
                sw_render_times.push(0.0);
                sw_render_times.push(0.0);
                continue;
            }
            if !hw_rt && scene.name == "landscape"
            {
                sw_render_times.push(0.0);
                sw_render_times.push(0.0);
                sw_render_times.push(0.0);
                sw_render_times.push(0.0);
                continue;
            }

            let mut path_json_buf = std::path::PathBuf::new();
            path_json_buf.push(scenes_dir);
            path_json_buf.push(scene.name);
            path_json_buf.push(scene.name);
            path_json_buf.set_extension("json");

            let path_json = path_json_buf.as_path();

            let res = std::fs::exists(path_json);
            if res.is_err() || !res.unwrap()
            {
                eprintln!("Scene \"{}\" not found.", scene.name);
                continue;
            }

            let (lp_scene, cameras) = lpl::load_scene_yoctogl_v24(path_json, &device, &queue, !hw_rt).unwrap();
            scene_stats.push(lp::get_scene_stats(&lp_scene));

            if cameras.len() <= 0 {
                eprintln!("There are no cameras in scene \"{}\".", scene.name);
            }

            for (i, camera) in cameras.iter().enumerate()
            {
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

                assert!(scene.samples % num_samples_per_pixel == 0);
                let num_accums = scene.samples / num_samples_per_pixel;
                print!("Scene \"{}\": ", scene.name);
                use std::io::Write;
                std::io::stdout().flush().unwrap();

                let start_time = std::time::Instant::now();

                for accum_idx in 0..num_accums
                {
                    let desc = lp::PathtraceDesc {
                        scene: &lp_scene,
                        render_target: output_tex.front(),
                        resources: &pathtrace_resources,
                        accum_params: &lp::AccumulationParams {
                            prev_frame: Some(output_tex.back()),
                            accum_counter: accum_idx,
                        },
                        tile_params: None,
                        camera_params: camera.params,
                        camera_transform: camera.transform,
                        force_software_bvh: !hw_rt,
                        advanced: lp::AdvancedParams {
                            max_radiance: scene.max_radiance,
                        }
                    };
                    lp::pathtrace_scene(&device, &queue, &desc, Default::default(), None);
                    output_tex.flip();

                    // Avoid overloading the command buffer.
                    if accum_idx % 50 == 0 {
                        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    }
                }
                output_tex.flip();  // Final image is now in front.

                // Wait for GPU tasks to finish.
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let elapsed = start_time.elapsed();
                let elapsed_f32 = elapsed.as_millis() as f32 / 1000.0;

                println!("Done. ({}s)", elapsed_f32);

                if !hw_rt {
                    sw_render_times.push(elapsed_f32);
                }

                if hw_rt
                {
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
                        viewport: None,
                    }, exposure, true, true);

                    let mut output_name = String::from(scene.name);
                    use std::fmt::Write;
                    if cameras.len() > 1 { write!(output_name, "{}", i+1).unwrap(); }

                    let mut output_path_buf = std::path::PathBuf::new();
                    output_path_buf.push(output_dir_name);
                    output_path_buf.push(output_name.clone());
                    output_path_buf.set_extension("png");
                    let output_path = output_path_buf.as_path();

                    let res = lpl::save_texture(&device, &queue, output_path, &tonemapped);

                    render_infos.push(RenderInfo {
                        output_file: output_name.clone(),
                        scene_idx: scene_idx,
                        time: elapsed_f32,
                        res_x: width,
                        res_y: height
                    });
                }
            }
        }
    }

    // Print latex table for scene descriptions
    {
        println!("\\begin{{table}}[H]");
        println!("  \\centering");
        println!("  \\footnotesize");
        println!("  \\label{{tab:scene_stats}}");
        println!("  \\sisetup{{");
        println!("    group-separator={{,}}, % adds commas to large numbers");
        println!("    table-align-text-post=false,");
        println!("    detect-weight=true,");
        println!("    detect-family=true");
        println!("  }}");
        println!("  \\begin{{tabular}}{{");
        println!("    l % scene name");
        println!("    S % triangles (millions)");
        println!("    S % instances");
        println!("    S % materials");
        println!("    S % lights");
        println!("    S % textures");
        println!("  }}");
        println!("    \\toprule");
        println!("    {{Scene}} & {{Triangles}} & {{Instances}} & {{Materials}} & {{Lights}} & {{Textures}}\\\\");
        println!("    \\midrule");
        for i in 0..scenes.len()
        {
            let stats = &scene_stats[i];
            println!("    {} & {} & {} & {} & {} & {} \\\\", scenes[i].name, stats.total_tri_count, stats.instances, stats.materials, stats.lights, stats.textures);
        }
        println!("    \\bottomrule");
        println!("  \\end{{tabular}}");
        println!("\\end{{table}}");
    }

    // Print latex table for timings
    {
        let base_image_id = 15;

        println!("\\begin{{table}}[H]");
        println!("  \\centering");
        println!("  \\footnotesize");
        println!("  \\label{{tab:scene_stats}}");
        println!("  \\sisetup{{");
        println!("    group-separator={{,}}, % adds commas to large numbers");
        println!("    table-align-text-post=false,");
        println!("    detect-weight=true,");
        println!("    detect-family=true");
        println!("  }}");
        println!("  \\begin{{tabular}}{{");
        println!("    l % image reference");
        println!("    S % scene name");
        println!("    S % resolution");
        println!("    S % number of samples");
        println!("    S % time software bvh");
        println!("    S % time hardware bvh");
        println!("  }}");
        println!("    \\toprule");
        println!("    {{Render}} & {{Scene}} & {{Resolution}} & {{Samples}} & {{Time SW}} & {{Time HW}}\\\\");
        println!("    \\midrule");
        for (i, render_info) in render_infos.iter().enumerate()
        {
            let image_id = render_info.scene_idx + base_image_id;
            let scene = &scenes[render_info.scene_idx];
            let sw_time = sw_render_times[i];
            if sw_time == 0.0 {
                println!("    Image {} & {} & {}x{} & {} & {}s & {}s \\\\", image_id, scene.name, render_info.res_x, render_info.res_y, scene.samples, "/", render_info.time);
            } else {
                println!("    Image {} & {} & {}x{} & {} & {}s & {}s \\\\", image_id, scene.name, render_info.res_x, render_info.res_y, scene.samples, sw_time, render_info.time);
            }
        }
        println!("    \\bottomrule");
        println!("  \\end{{tabular}}");
        println!("\\end{{table}}");
        println!("");
    }

    // Print latex images
    {
        for render_info in &render_infos
        {
            let image_name = &render_info.output_file;
            let credits = scenes[render_info.scene_idx].credits;

            println!("\\begin{{sidewaysfigure}}");
            println!("  \\centering");
            println!("  \\includegraphics[width=1.0\\textwidth]{{images/renders/{image_name}.png}}");
            println!("  \\caption{{{credits}}}");
            println!("  \\label{{lst:caustics}}");
            println!("\\end{{sidewaysfigure}}");
            println!("");
        }
    }

    unsafe { std::arch::asm!("int3"); }
}

fn compute_dimensions_for_1080p(aspect: f32) -> (u32, u32)
{
    if aspect < 1.0 {  // Taller than wide
        return ((1920.0 * aspect) as u32, 1920);
    } else {  // Wider than tall
        return (1920, (1920.0 / aspect) as u32);
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
