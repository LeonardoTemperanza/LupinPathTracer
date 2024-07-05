
#[derive(Default, Clone, Copy)]
pub struct PreprocessorParams
{
    // These are needed for "#include"
    pub all_shader_names: &'static [&'static str],
    pub all_shader_contents: &'static [&'static str],

    // Pertaining to compute shaders only
    // (these can be ignored otherwise)

    pub desired_workgroup_size_x: u32,
    pub desired_workgroup_size_y: u32,
    pub desired_workgroup_size_z: u32,

    // Device dependent limits
    pub max_workgroup_size_x: u32,
    pub max_workgroup_size_y: u32,
    pub max_workgroup_size_z: u32,
    pub max_workgroup_invocations: u32
}

impl PreprocessorParams
{
    pub fn set_compute_limits(&self, device: &wgpu::Device)
    {

    }
}

pub struct SourceTransformation
{
    pub at: i32,  // Original line number where the transformation takes place
    pub kind: SourceTransformKind,
}

pub enum SourceTransformKind
{
    IncludeFile
    {
        included: &'static str
    },
    AddLines
    {
        num_lines: i32,
    },
    RemoveLines
    {
        num_lines: i32,
    }
}

// Maps the processed file to the original one and viceversa
pub struct SourceMap
{
    transforms: Vec<SourceTransformation>
}

pub struct Tokenizer
{
    line_num: i32,
    file: &'static str,
    at: usize
}

pub fn preprocess_shader(shader_name: &str, default_shader: &str, params: PreprocessorParams)->(String, SourceMap)
{
    assert!(params.all_shader_names.len() == params.all_shader_contents.len());

    let mut res = String::default();
    let source_map = SourceMap { transforms: vec![] };

    // Find the shader name in the list of shaders
    let mut shader_content: &str = "";
    let mut found = false;
    let num_shaders = params.all_shader_names.len();
    for i in 0..num_shaders
    {
        if shader_name.eq(params.all_shader_names[i])
        {
            shader_content = params.all_shader_contents[i];
            found = true;
            break;
        }
    }

    if !found
    {
        eprintln!("Preprocessor error: shader '{}' not found!", shader_name);
        return (default_shader.to_string(), source_map);
    }

    /*
    let mut tokenizer = Tokenizer { line_num: 0, file: shader_content, at: 0 };
    while tokenizer.at < shader_content.len()
    {
        eat_all_whitespace(&mut tokenizer);

        if shader_content.chars().nth(tokenizer.at) == Some('#')  // Could be a custom pragma directive
        {
            tokenizer.at += 1;
            if tokenizer.at >= shader_content.len() { break; }

            if shader_content[tokenizer.at..].starts_with("include")
            {
                println!("Include!");
                tokenizer.at += "include".len();
            }
            else if shader_content[tokenizer.at..].starts_with("assert")
            {
                println!("Assert");
                tokenizer.at += "assert".len();
            }
            else
            {
                tokenizer.at += 1;
            }
        }
        else
        {
            tokenizer.at += 1;
        }
    }
    */

    res = String::from(shader_content);
    return (res, source_map);
}

pub fn translate_and_show_error_message(source_map: SourceMap)
{

}

pub fn is_newline(c: char)->bool
{
    return c == '\n'
}

pub fn is_whitespace(c: char)->bool
{
    return is_newline(c) || c == ' ' || c == '\t' || c == '\r';
}

pub fn eat_all_whitespace(tokenizer: &mut Tokenizer)
{
    for i in tokenizer.at..tokenizer.file.len()
    {
        let content: &str = tokenizer.file;
        let c = content.chars().nth(i).unwrap();
        if !is_whitespace(c) { break; }

        if is_newline(c)
        {
            tokenizer.line_num += 1;
        }

        tokenizer.at += 1;
    }
}
