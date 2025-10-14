
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

// Don't spawn a terminal window on windows
#![windows_subsystem = "windows"]

use std::time::Instant;

pub use lupin as lp;

fn main()
{
    let (instance, device, queue, adapter) = lp::init_default_wgpu_context_no_window();
    let pathtrace_resources = lp::build_pathtrace_resources(&device, &Default::default());
    let tonemap_resources = lp::build_tonemap_resources(&device);
}
