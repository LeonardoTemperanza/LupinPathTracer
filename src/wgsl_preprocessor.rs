
pub struct PreprocessorParams
{
    // Pertaining to compute shaders only
    // (these can be ignored otherwise)

    desired_workgroup_size_x: u32,
    desired_workgroup_size_y: u32,
    desired_workgroup_size_z: u32,

    // Device dependent limits
    max_workgroup_size_x: u32,
    max_workgroup_size_y: u32,
    max_workgroup_size_z: u32,
    max_workgroup_invocations: u32
}

pub fn preprocess_shader(shader_src: &str, params: PreprocessorParams)->String
{

}

pub fn eat_all_whitespace(string: &[char])->&[char]
{

}
