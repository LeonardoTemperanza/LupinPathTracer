
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(unexpected_cfgs)]

// Don't spawn a terminal window on windows
#![windows_subsystem = "windows"]

use std::time::Instant;

pub use lupin_pt as lp;

fn main()
{
    let (device, queue, adapter) = lp::init_default_wgpu_context_no_window();
    let tonemap_resources = lp::build_tonemap_resources(&device);
    let pathtrace_resources = lp::build_pathtrace_resources(&device, &Default::default());
}
