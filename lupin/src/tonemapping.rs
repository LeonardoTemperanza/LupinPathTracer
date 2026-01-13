
use crate::base::*;

pub static TONEMAPPING_SRC: &str = include_str!("shaders/tonemapping.wgsl");

pub struct TonemapResources
{
    pub pipeline: wgpu::RenderPipeline,
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
        ]
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Lupin Tonemapping Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            range: 0..std::mem::size_of::<TonemapUniforms>() as u32,
        }],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &tonemap_shader,
            entry_point: Some("vert_main"),
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &tonemap_shader,
            entry_point: Some("main"),
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
    });

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
        pipeline,
        sampler,
    };
}

pub struct TonemapDesc
{
    /// If None, the whole image is tonemapped.
    pub viewport: Option<Viewport>,
    /// Applied as: color = color * 2^exposure
    pub exposure: f32,
    /// Whether or not to perform filmic tonemapping.
    pub filmic: bool,
    /// Whether or not to perform Linear -> SRGB conversion.
    pub srgb: bool,
    /// Whether or not to clear the texture's previous contents.
    pub clear: bool,
}

impl Default for TonemapDesc
{
    fn default() -> Self
    {
        return Self {
            viewport: None,
            exposure: 0.0,
            filmic: false,
            srgb: true,
            clear: true,
        };
    }
}

// NOTE: Coupled to the tonemapping shader
#[derive(Default)]
#[repr(C)]
struct TonemapUniforms
{
    scale: Vec2,
    exposure: f32,
    filmic: u32,  // bool
    srgb: u32,  // bool
}

#[derive(Default, Copy, Clone, Debug)]
pub struct Viewport
{
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32
}

/// `src` and `dst` are required to be different textures, because of underlying limitations of graphics APIs.
pub fn tonemap_and_fit_aspect(device: &wgpu::Device, queue: &wgpu::Queue, resources: &TonemapResources,
                              src: &wgpu::Texture, dst: &wgpu::Texture, desc: &TonemapDesc)
{
    let render_target_view = dst.create_view(&Default::default());
    let hdr_texture_view = src.create_view(&Default::default());

    let viewport = desc.viewport.unwrap_or(Viewport {
        x: 0.0,
        y: 0.0,
        w: dst.width() as f32,
        h: dst.height() as f32
    });

    let src_aspect = src.width() as f32 / src.height() as f32;
    let dst_aspect = viewport.w / viewport.h;
    let scale = if src_aspect > dst_aspect {
        Vec2 { x: 1.0, y: dst_aspect / src_aspect }
    } else {
        Vec2 { x: src_aspect / dst_aspect, y: 1.0 }
    };

    let params = TonemapUniforms {
        scale: scale,
        exposure: desc.exposure,
        filmic: if desc.filmic { 1 } else { 0 },
        srgb: if desc.srgb { 1 } else { 0 },
    };

    let pipeline = &resources.pipeline;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &resources.pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr_texture_view) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&resources.sampler) },
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
                    load: if desc.clear { wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }) } else { wgpu::LoadOp::Load },
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
        pass.set_push_constants(wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX, 0, to_u8_slice(&[params]));
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..6, 0..1);
    }

    queue.submit(Some(encoder.finish()));
}
