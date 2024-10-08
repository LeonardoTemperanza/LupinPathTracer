
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

fn main()
{
    println!("cargo::rerun-if-changed=src/renderer/renderer_wgpu/shaders/*");

    let rust_file_dest_path = "src/renderer/renderer_wgpu/shaders.rs";
    let mut file = File::create(&rust_file_dest_path).unwrap();

    let shader_directory = "src/renderer/renderer_wgpu/shaders/";

    writeln!(file, "").unwrap();
    writeln!(file, "////////").unwrap();
    writeln!(file, "// This file was automatically generated by the 'build.rs' program.").unwrap();
    writeln!(file, "// It contains the names and contents of the shader files from the '{}' directory.", shader_directory).unwrap();
    writeln!(file, "").unwrap();

    let mut file_names = Vec::new();
    let mut file_contents = Vec::new();

    for entry in fs::read_dir(shader_directory).unwrap()
    {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file()
        {
            let content = fs::read_to_string(&path).unwrap();
            let file_name = path.strip_prefix(shader_directory).unwrap().file_name().unwrap().to_str().unwrap().to_string();

            file_names.push(file_name);
            file_contents.push(content);
        }
    }

    writeln!(file, "static SHADER_NAMES: [&str; {}] = {:?};", file_names.len(), file_names).unwrap();
    writeln!(file, "").unwrap();
    writeln!(file, "static SHADER_CONTENTS: [&str; {}] = {:?};", file_contents.len(), file_contents).unwrap();
}