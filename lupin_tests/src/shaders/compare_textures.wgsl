
@group(0) @binding(0) var<storage, read_write> error: atomic<u32>;  // Actually a bool
@group(0) @binding(1) var output: texture_storage_2d<rgba16float, read>;
@group(0) @binding(2) var expected_output: texture_storage_2d<rgba16float, read>;

struct PushConstants { epsilon: f32 }
var<immediate> constants: PushConstants;

const WORKGROUP_SIZE_X: u32 = 4;
const WORKGROUP_SIZE_Y: u32 = 4;

@compute
@workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3u)
{
    let dim = textureDimensions(output).xy;
    if any(global_id.xy >= dim) { return; }

    let diff = textureLoad(output, global_id.xy).rgb - textureLoad(expected_output, global_id.xy).rgb;
    if length(diff) > constants.epsilon
    {
        if atomicLoad(&error) == 0 {
            atomicStore(&error, 1);
        }
    }
}
