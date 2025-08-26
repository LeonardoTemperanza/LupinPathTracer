
@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@group(0) @binding(2) var<uniform> params: TonemapParams;

struct TonemapParams
{
    scale: vec2f,
    exposure: f32,
    // For filmic tonemapping
    linear_white: f32,
    a: f32, b: f32, c: f32, d: f32, e: f32, f: f32,
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
        vec4f(pos[vertex_index] * params.scale, 0.0f, 1.0f),
        tex_coords[vertex_index],
    );
}

@fragment
fn filmic_main(input: VertexOutput) -> @location(0) vec4f
{
    if any(input.tex_coords > vec2f(1.0f)) || any(input.tex_coords < vec2f(0.0f)) {
        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    var color = max(textureSample(source_texture, source_sampler, input.tex_coords).rgb, vec3f(0.0f));
    color *= pow(2.0f, params.exposure);
    color = tonemap_filmic_uc2(color, params.linear_white,
                               params.a, params.b,
                               params.c, params.d,
                               params.e, params.f);
    return vec4f(linear_to_srgb(color), 1.0f);
}

@fragment
fn aces_main(input: VertexOutput) -> @location(0) vec4f
{
    if any(input.tex_coords > vec2f(1.0f)) || any(input.tex_coords < vec2f(0.0f)) {
        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    var color = max(textureSample(source_texture, source_sampler, input.tex_coords).rgb, vec3f(0.0f));
    color *= pow(2.0f, params.exposure);
    return vec4(linear_to_srgb(tonemap_aces(color)), 1.0f);
}

@fragment
fn no_tonemap_main(input: VertexOutput) -> @location(0) vec4f
{
    if any(input.tex_coords > vec2f(1.0f)) || any(input.tex_coords < vec2f(0.0f)) {
        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);
    }

    let color = clamp(textureSample(source_texture, source_sampler, input.tex_coords).rgb, vec3f(0.0f), vec3f(1.0f));
    return vec4(color, 1.0f);
}

// Tonemapping functions from: https://gist.github.com/SpineyPete/ebf9619f009318536c6da48209894fed

fn tonemap_filmic_uc2(linear_color: vec3f, linear_white: f32, A: f32, B: f32, C: f32, D: f32, E: f32, F: f32) -> vec3f
{
    // Uncharted II configurable tonemapper.

    // A = shoulder strength
    // B = linear strength
    // C = linear angle
    // D = toe strength
    // E = toe numerator
    // F = toe denominator
    // Note: E / F = toe angle
    // linear_white = linear white point value

    var x: vec3f = linear_color;
    let color: vec3f = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

    x = vec3f(linear_white);
    let white: vec3f = ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;

    return color / white;
}

fn tonemap_filmic_uc2_default(color: vec3f) -> vec3f
{
    // Uncharted II fixed tonemapping formula.
	// Gives a warm and gritty image, saturated shadows and bleached highlights.

    return tonemap_filmic_uc2(color, 11.2, 0.22, 0.3, 0.1, 0.2, 0.01, 0.30);
}

fn tonemap_aces(color: vec3f) -> vec3f
{
    return color;

    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    /*
    let hdr = color * 0.6f;  // brings it back to ACES range
    let ldr = (hdr * hdr * 2.51f + hdr * 0.03f) /
              (hdr * hdr * 2.43f + hdr * 0.59f + vec3f(0.14f));
    return max(vec3f(0.0f), ldr);
    */

    /*

    // https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
    // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
    let ACESInputMat = transpose(mat3x3f(
        0.59719f, 0.35458f, 0.04823f,
        0.07600f, 0.90834f, 0.01566f,
        0.02840f, 0.13383f, 0.83777f,
    ));
    // ODT_SAT => XYZ => D60_2_D65 => sRGB
    let ACESOutputMat = transpose(mat3x3f(
        1.60475f, -0.53108f, -0.07367f,
        -0.10208f, 1.10813f, -0.00605f,
        -0.00327f, -0.07276f, 1.07602f,
    ));

    let srgb = ACESInputMat * color;

    // RRT => ODT
    let odt = (srgb * srgb + srgb * 0.0245786f - 0.000090537f) /
             (srgb * srgb * 0.983729f + srgb * 0.4329510f + 0.238081f);

    let ldr = ACESOutputMat * odt;
    return max(vec3f(0, 0, 0), ldr);


    */

    // OLD ACES IMPL

    /*
    // ACES filmic tonemapper with highlight desaturation ("crosstalk").
    // Based on the curve fit by Krzysztof Narkowicz.
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/

    const slope: f32 = 12.0f; // higher values = slower rise.

    // Store grayscale as an extra channel.
    let x = vec4f(
        // RGB
        color.r, color.g, color.b,
        // Luminosity
        (color.r * 0.299) + (color.g * 0.587) + (color.b * 0.114)
    );

    // ACES Tonemapper
    const a: f32 = 2.51f;
    const b: f32 = 0.03f;
    const c: f32 = 2.43f;
    const d: f32 = 0.59f;
    const e: f32 = 0.14f;

    let tonemap: vec4f = clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec4f(0.0f), vec4f(1.0f));
    var t: f32 = x.a;

    t = t * t / (slope + t);

    // Return after desaturation step.
    return mix(tonemap.rgb, tonemap.aaa, t);
    */
}

fn linear_to_srgb(linear_color: vec3f) -> vec3f
{
    let cutoff = vec3f(linear_color <= vec3f(0.0031308f));
    let higher = vec3f(1.055f) * pow(linear_color, vec3f(1.0f/2.4f)) - vec3f(0.055f);
    let lower  = linear_color * vec3f(12.92f);
    return mix(higher, lower, cutoff);
}
