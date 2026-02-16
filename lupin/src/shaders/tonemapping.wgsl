
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

var<immediate> constants: TonemapParams;

struct TonemapParams
{
    scale: vec2f,
    exposure: f32,
    filmic: u32,  // bool
    srgb: u32,  // bool
}

struct VertexOutput
{
    @builtin(position) clip_position: vec4f,
    @location(0) tex_coords: vec2f,
}

@vertex
fn vert_main(@builtin(vertex_index) vertex_index : u32) -> VertexOutput
{
    // NOTE: Assuming front face is counter-clockwise
    const pos = array(
        vec2(-1.0,  1.0),
        vec2( 1.0, -1.0),
        vec2( 1.0,  1.0),
        vec2(-1.0,  1.0),
        vec2(-1.0, -1.0),
        vec2( 1.0, -1.0),
    );

    const tex_coords = array(
        vec2(0.0, 0.0),
        vec2(1.0, 1.0),
        vec2(1.0, 0.0),
        vec2(0.0, 0.0),
        vec2(0.0, 1.0),
        vec2(1.0, 1.0),
    );

    return VertexOutput(
        vec4f(pos[vertex_index] * constants.scale, 0.0f, 1.0f),
        tex_coords[vertex_index],
    );
}

@fragment
fn main(input: VertexOutput) -> @location(0) vec4f
{
    if any(input.tex_coords > vec2f(1.0f)) || any(input.tex_coords < vec2f(0.0f)) {
        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    var color = max(textureSample(source_texture, source_sampler, input.tex_coords).rgb, vec3f(0.0f));
    if constants.exposure != 0.0f { color *= exp2(constants.exposure); }
    if constants.filmic != 0 { color = tonemap_filmic(color); }
    if constants.srgb != 0 { color = linear_to_srgb(color); }

    return vec4f(color, 1.0f);
}

fn tonemap_filmic(color: vec3f) -> vec3f
{
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    let hdr = color * 0.6f;  // Brings it back to ACES range.
    let ldr = (hdr * hdr * 2.51f + hdr * 0.03f) /
              (hdr * hdr * 2.43f + hdr * 0.59f + 0.14f);
    return max(ldr, vec3f(0.0f));
}

fn linear_to_srgb(linear_color: vec3f) -> vec3f
{
    let cutoff = vec3f(linear_color <= vec3f(0.0031308f));
    let higher = vec3f(1.055f) * pow(linear_color, vec3f(1.0f/2.4f)) - vec3f(0.055f);
    let lower  = linear_color * vec3f(12.92f);
    return mix(higher, lower, cutoff);
}
