
@group(0) @binding(0) var to_show: texture_2d<u32>;

@vertex
fn vs_main(@builtin(vertex_index) vert_idx: u32)->@builtin(position) vec4f
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

    return vec4f(pos[vert_idx], 0.0f, 0.0f);
}

@fragment
fn fs_main(@builtin(position) pos: vec4f)->@location(0) vec4f
{
    var texture_size: vec2u = textureDimensions(to_show).xy;
    let aspect_ratio: f32 = f32(texture_size.y) / f32(texture_size.x);
    let texture_scale = vec2f(1.0f / aspect_ratio, 1.0f);
    var texture_coord: vec2f = (pos.xy + 1.0f) * 0.5f;
    texture_coord = texture_scale * (texture_coord - 0.5f) + 0.5f;

    if (texture_coord.x < 0.0 || texture_coord.x > 1.0 ||
        texture_coord.y < 0.0 || texture_coord.y > 1.0)
    {
        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    return vec4f(1.0f, 0.0f, 0.0f, 1.0f);
}
