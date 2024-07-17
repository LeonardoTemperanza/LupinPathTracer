
@group(0) @binding(0) var to_show: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> res: vec2u;

struct VertexOutput
{
    @builtin(position) pos: vec4f,
    @location(0) tex_coords: vec2f
}

@vertex
fn vs_main(@builtin(vertex_index) vert_idx: u32)->VertexOutput
{
    var pos = array<vec2f, 6>
    (
        vec2f(-1.0, -1.0),
        vec2f(-1.0, 1.0),
        vec2f(1.0, -1.0),
        vec2f(1.0, -1.0),
        vec2f(-1.0, 1.0),
        vec2f(1.0, 1.0)
    );

    return VertexOutput(vec4f(pos[vert_idx], 0.0f, 1.0f), pos[vert_idx] * -0.5f + 0.5f);
}

@fragment
fn fs_main(@location(0) tex_coords: vec2f)->@location(0) vec4f
{
    var texture_size: vec2u = textureDimensions(to_show).xy;
    let image_aspect_ratio: f32 = f32(texture_size.x) / f32(texture_size.y);
    let screen_aspect_ratio: f32 = f32(res.x) / f32(res.y);

    // Scale UVs so that it fits the screen while keeping
    // original aspect ratio.
    var texture_scale  = vec2f(1.0f);
    var texture_offset = vec2f(0.0f);
    if screen_aspect_ratio > image_aspect_ratio
    {
        // Vertical bars
        texture_scale = vec2f(screen_aspect_ratio / image_aspect_ratio, 1.0f);
    }
    else
    {
        // Horizontal bars
        texture_scale = vec2f(1.0f, image_aspect_ratio / screen_aspect_ratio);
    }
    
    let texture_coord = (tex_coords - 0.5f) * texture_scale + 0.5f;

    if (texture_coord.x < 0.0 || texture_coord.x > 1.0 ||
        texture_coord.y < 0.0 || texture_coord.y > 1.0)
    {
        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    return textureSample(to_show, tex_sampler, texture_coord);
}
