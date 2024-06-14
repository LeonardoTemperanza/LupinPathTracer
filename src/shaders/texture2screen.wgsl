
struct VertexOutput
{
    @builtin(position) pos: vec4f,
    @location(1) tex_coords: vec2f
}

@vertex
fn vert_main(@builtin(vertex_index) vert_idx: u32)->VertexOutput
{
    var positions = array<vec2f, 6>
    (
        vec2f(-1.0, -1.0), // bottom-left
        vec2f( 1.0, -1.0), // bottom-right
        vec2f(-1.0,  1.0), // top-left
        vec2f(-1.0,  1.0), // top-left
        vec2f( 1.0, -1.0), // bottom-right
        vec2f( 1.0,  1.0)  // top-right
    );

    var tex_coords = array<vec2f, 6>
    (
        vec2f(0.0, 1.0), // bottom-left
        vec2f(1.0, 1.0), // bottom-right
        vec2f(0.0, 0.0), // top-left
        vec2f(0.0, 0.0), // top-left
        vec2f(1.0, 1.0), // bottom-right
        vec2f(1.0, 0.0)  // top-right
    );

    var pos       = positions[vert_idx];
    var tex_coord = tex_coords[vert_idx];
    
    var output: VertexOutput;
    output.pos = vec4f(pos, 0.0f, 1.0f);
    output.tex_coords = tex_coord;
    return output;
}

@group(0) @binding(0) var texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@fragment
fn frag_main(input: VertexOutput)->@location(0) vec4f
{
    return textureSample(texture, tex_sampler, input.tex_coords);
}
