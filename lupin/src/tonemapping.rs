
use crate::base::*;
use crate::wgpu_utils::*;

use wgpu::util::DeviceExt;  // For some extra device traits.

pub static TONEMAPPING_SRC: &str = include_str!("shaders/tonemapping.wgsl");

pub struct TonemapResources
{
    pub identity_pipeline: wgpu::RenderPipeline,
    pub aces_pipeline: wgpu::RenderPipeline,
    pub filmic_pipeline: wgpu::RenderPipeline,
    pub sampler: wgpu::Sampler,
}

pub fn build_tonemap_resources(device: &wgpu::Device) -> TonemapResources
{
    let shader_desc = wgpu::ShaderModuleDescriptor {
        label: Some("Lupin Tonemapping Shader"),
        source: wgpu::ShaderSource::Wgsl(TONEMAPPING_SRC.into())
    };

    let tonemap_shader = device.create_shader_module(shader_desc);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2
                },
                count: None
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None
            }
        ]
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lupin Tonemapping Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    fn tonemap_pipeline_descriptor<'a>(shader: &'a wgpu::ShaderModule, pipeline_layout: &'a wgpu::PipelineLayout, frag_main: &'static str) -> wgpu::RenderPipelineDescriptor<'a>
    {
        return wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vert_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(frag_main),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,  // TODO How to work with other formats?
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        }
    }

    let identity_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "no_tonemap_main"));
    let aces_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "aces_main"));
    let filmic_pipeline = device.create_render_pipeline(&tonemap_pipeline_descriptor(&tonemap_shader, &pipeline_layout, "filmic_main"));

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    return TonemapResources {
        identity_pipeline,
        aces_pipeline,
        filmic_pipeline,
        sampler,
    };
}

#[derive(Copy, Clone)]
pub struct ColorGradeParams
{
    pub exposure: f32,
    pub tint: Vec3,
    pub linear_contrast: f32,
    pub log_contrast: f32,
    pub linear_saturation: f32,
    pub filmic: bool,
    pub srgb: bool,
    pub contrast: f32,
    pub saturation: f32,
    pub shadows: f32,
    pub midtones: f32,
    pub highlights: f32,
    pub shadows_color: Vec3,
    pub midtones_color: Vec3,
    pub highlights_color: Vec3,
}

impl Default for ColorGradeParams
{
    fn default() -> Self
    {
        return Self {
            exposure: 0.0,
            tint: Vec3::ones(),
            linear_contrast: 0.5,
            log_contrast: 0.5,
            linear_saturation: 0.5,
            filmic: false,
            srgb: true,
            contrast: 0.5,
            saturation: 0.5,
            shadows: 0.5,
            midtones: 0.5,
            highlights: 0.5,
            shadows_color: Vec3::ones(),
            midtones_color: Vec3::ones(),
            highlights_color: Vec3::ones(),
        };
    }
}

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub enum TonemapOperator
{
    #[default] Aces,
    FilmicUC2,
    FilmicCustom { linear_white: f32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32 },
}

#[derive(Default, Clone, Copy, Debug)]
pub struct TonemapParams
{
    pub operator: TonemapOperator,
    pub exposure: f32,
}

pub struct TonemapDesc<'a>
{
    pub resources: &'a TonemapResources,
    pub hdr_texture: &'a wgpu::Texture,
    pub render_target: &'a wgpu::Texture,
    pub tonemap_params: &'a TonemapParams,
}

// NOTE: Coupled to the tonemapping shader
#[derive(Default)]
#[repr(C)]
struct TonemapUniforms
{
    scale: Vec2,
    exposure: f32,
    // For filmic tonemapping
    linear_white: f32,
    a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Viewport
{
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32
}

pub fn tonemap_and_fit_aspect(device: &wgpu::Device, queue: &wgpu::Queue, desc: &TonemapDesc, viewport: Option<Viewport>)
{
    let resources = desc.resources;
    let hdr_texture = desc.hdr_texture;
    let render_target = desc.render_target;
    let tonemap_params = desc.tonemap_params;

    let render_target_view = render_target.create_view(&Default::default());
    let hdr_texture_view = hdr_texture.create_view(&Default::default());

    let viewport = viewport.unwrap_or(Viewport {
        x: 0.0,
        y: 0.0,
        w: desc.render_target.size().width as f32,
        h: desc.render_target.size().height as f32
    });

    let src_aspect = desc.hdr_texture.size().width as f32 / desc.hdr_texture.size().height as f32;
    let dst_aspect = viewport.w / viewport.h;

    let mut params = TonemapUniforms::default();
    params.exposure = tonemap_params.exposure;
    params.scale = if src_aspect > dst_aspect {
        Vec2 { x: 1.0, y: dst_aspect / src_aspect }
    } else {
        Vec2 { x: src_aspect / dst_aspect, y: 1.0 }
    };

    let pipeline = match tonemap_params.operator
    {
        TonemapOperator::Aces      => &resources.aces_pipeline,
        TonemapOperator::FilmicUC2 =>
        {
            params.linear_white = 11.2;
            params.a = 0.22;
            params.b = 0.3;
            params.c = 0.1;
            params.d = 0.2;
            params.e = 0.01;
            params.f = 0.30;

            &resources.filmic_pipeline
        }
        TonemapOperator::FilmicCustom { linear_white, a, b, c, d, e, f } =>
        {
            params.linear_white = linear_white;
            params.a = a;
            params.b = b;
            params.c = c;
            params.d = d;
            params.e = e;
            params.f = f;

            &resources.filmic_pipeline
        }
    };

    let params_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: to_u8_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &resources.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_resource_nocheck(&params_uniform) },
        ]
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None
        });

        pass.set_viewport(viewport.x, viewport.y, viewport.w, viewport.h, 0.0, 1.0);
        pass.set_scissor_rect(viewport.x as u32, viewport.y as u32, viewport.w as u32, viewport.h as u32);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

pub fn blit_texture_and_fit_aspect(device: &wgpu::Device, queue: &wgpu::Queue, resources: &TonemapResources, input: &wgpu::Texture, blit_to: &wgpu::Texture, viewport: Option<Viewport>)
{
    let blit_to_view = blit_to.create_view(&Default::default());
    let input_view = input.create_view(&Default::default());

    let viewport = viewport.unwrap_or(Viewport {
        x: 0.0,
        y: 0.0,
        w: blit_to.size().width as f32,
        h: blit_to.size().height as f32
    });

    let src_aspect = input.size().width as f32 / input.size().height as f32;
    let dst_aspect = viewport.w / viewport.h;

    let mut params = TonemapUniforms::default();
    params.scale = if src_aspect > dst_aspect {
        Vec2 { x: 1.0, y: dst_aspect / src_aspect }
    } else {
        Vec2 { x: src_aspect / dst_aspect, y: 1.0 }
    };

    let params_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: to_u8_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &resources.aces_pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&input_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
            wgpu::BindGroupEntry { binding: 2, resource: buffer_resource_nocheck(&params_uniform) },
        ]
    });

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &blit_to_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None
        });

        pass.set_viewport(viewport.x, viewport.y, viewport.w, viewport.h, 0.0, 1.0);
        pass.set_scissor_rect(viewport.x as u32, viewport.y as u32, viewport.w as u32, viewport.h as u32);
        pass.set_pipeline(&resources.identity_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}

fn buffer_resource_nocheck(buffer: &wgpu::Buffer) -> wgpu::BindingResource
{
    return wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: buffer, offset: 0, size: None });
}
