
////////
// This file was automatically generated by the 'build.rs' program.
// It contains the names and contents of the shader files from the 'src/renderer/renderer_wgpu/shaders/' directory.

static SHADER_NAMES: [&str; 3] = ["base.wgsl", "pathtracer.wgsl", "show_texture.wgsl"];

static SHADER_CONTENTS: [&str; 3] = ["\r\n// Useful constants\r\nconst f32_max: f32 = 0x1.fffffep+127;\r\n\r\nfn transform_point(p: vec3f, transform: mat4x4f)->vec3f\r\n{\r\n    let p_vec4 = vec4f(p, 1.0f);\r\n    let transformed = transform * p_vec4;\r\n    return (transformed / transformed.w).xyz;\r\n}\r\n\r\nfn transform_dir(dir: vec3f, transform: mat4x4f)->vec3f\r\n{\r\n    let dir_vec4 = vec4f(dir, 0.0f);\r\n    return (transform * dir_vec4).xyz;\r\n}\r\n\r\nstruct Ray\r\n{\r\n    ori: vec3f,\r\n    dir: vec3f,\r\n    // Precomputed inverse of the ray direction. Since\r\n    // floating point division is more expensive than\r\n    // floating point multiplication, it is (in the context of raytracing)\r\n    // often faster to precompute the inverse of the ray ahead of time.\r\n    // On the other hand this is duplicated state that has to be managed\r\n    inv_dir: vec3f\r\n}\r\n\r\n// From: https://tavianator.com/2011/ray_box.html\r\n// For misses, t = f32_max\r\nfn ray_aabb_dst(ray: Ray, aabb_min: vec3f, aabb_max: vec3f)->f32\r\n{\r\n    let t_min: vec3f = (aabb_min - 0.001 - ray.ori) * ray.inv_dir;\r\n    let t_max: vec3f = (aabb_max + 0.001 - ray.ori) * ray.inv_dir;\r\n    let t1: vec3f = min(t_min, t_max);\r\n    let t2: vec3f = max(t_min, t_max);\r\n    let dst_far: f32  = min(min(t2.x, t2.y), t2.z);\r\n    let dst_near: f32 = max(max(t1.x, t1.y), t1.z);\r\n\r\n    let did_hit: bool = dst_far >= dst_near && dst_far > 0.0f;\r\n    return select(f32_max, dst_near, did_hit);\r\n}\r\n\r\n// From: https://www.shadertoy.com/view/MlGcDz\r\n// Triangle intersection. Returns { t, u, v }\r\n// For misses, t = f32_max\r\nfn ray_tri_dst(ray: Ray, v0: vec3f, v1: vec3f, v2: vec3f)->vec3f\r\n{\r\n    let v1v0 = v1 - v0;\r\n    let v2v0 = v2 - v0;\r\n    let rov0 = ray.ori - v0;\r\n\r\n    // Cramer's rule for solving p(t) = ro+t·rd = p(u,v) = vo + u·(v1-v0) + v·(v2-v1).\r\n    // The four determinants above have lots of terms in common. Knowing the changing\r\n    // the order of the columns/rows doesn't change the volume/determinant, and that\r\n    // the volume is dot(cross(a,b,c)), we can precompute some common terms and reduce\r\n    // it all to:\r\n    let n = cross(v1v0, v2v0);\r\n    let q = cross(rov0, ray.dir);\r\n    let d = 1.0 / dot(ray.dir, n);\r\n    let u = d * dot(-q, v2v0);\r\n    let v = d * dot(q, v1v0);\r\n    var t = d * dot(-n, rov0);\r\n\r\n    if min(u, v) < 0.0 || (u+v) > 1.0 { t = f32_max; }\r\n    return vec3f(t, u, v);\r\n}\r\n", "\r\n//#include \"base.wgsl\"\r\n\r\n// NOTE: Early returns are heavily discouraged here because it\r\n// will lead to thread divergence, and that in turn will cause\r\n// any proceeding __syncthreads() call to invoke UB, as some threads\r\n// will never reach that point because they early returned. In general\r\n// execution between multiple threads should be as predictable as possible\r\n\r\n// Scene representation\r\n//@group(0) @binding(0) var models: storage\r\n\r\n@group(0) @binding(0) var output_texture: texture_storage_2d<rgba8unorm, write>;\r\n@group(0) @binding(1) var<storage, read> verts_pos: array<vec3f>;\r\n@group(0) @binding(2) var<storage, read> indices: array<u32>;\r\n@group(0) @binding(3) var<storage, read> bvh_nodes: array<BvhNode>;\r\n@group(0) @binding(4) var<storage, read> verts: array<Vertex>;\r\n@group(0) @binding(5) var<uniform> camera_transform: mat4x4f;\r\n\r\n//@group(1) @binding(0) var atlas_1_channel: texture_2d<r8unorm, read>;\r\n// Like base color\r\n//@group(1) @binding(2) var atlas_3_channels: texture_2d<rgb8unorm, read>;\r\n// Like environment maps\r\n//@group(1) @binding(4) var atlas_hdr_3_channels: texture_2d<rgbf32, read>;\r\n\r\n// Should add a per_frame here that i can use to send camera transform\r\n\r\nconst f32_max: f32 = 0x1.fffffep+127;\r\n\r\n// This doesn't include positions, as that\r\n// is stored in a separate buffer for locality\r\nstruct Vertex\r\n{\r\n    normal: vec3f,\r\n    tex_coords: vec2f\r\n}\r\n\r\nstruct PerFrame\r\n{\r\n    camera_transform: mat4x3f\r\n}\r\n\r\nstruct Material\r\n{\r\n    color_scale: vec3f,\r\n    alpha_scale: f32,\r\n    roughness_scale: f32,\r\n    emission_scale: f32,\r\n\r\n    // Textures (base coordinates for atlas lookup)\r\n    // An out of bounds index will yield the neutral value\r\n    // for the corresponding texture type\r\n    color_texture: vec2u,      // 3 channels\r\n    alpha_texture: vec2u,      // 1 channel\r\n    roughness_texture: vec2u,  // 1 channel\r\n    emission_texture: vec2u    // hdr 3 channels\r\n\r\n    // We also need some texture wrapping info, i guess.\r\n    // Stuff like: repeat, clamp, etc.\r\n}\r\n\r\nstruct Ray\r\n{\r\n    ori: vec3f,\r\n    dir: vec3f,\r\n    inv_dir: vec3f  // Precomputed inverse of the ray direction, for performance\r\n}\r\n\r\nfn transform_point(p: vec3f, transform: mat4x4f)->vec3f\r\n{\r\n    let p_vec4 = vec4f(p, 1.0f);\r\n    let transformed = transform * p_vec4;\r\n    return (transformed / transformed.w).xyz;\r\n}\r\n\r\nfn transform_dir(dir: vec3f, transform: mat4x4f)->vec3f\r\n{\r\n    let dir_vec4 = vec4f(dir, 0.0f);\r\n    return (transform * dir_vec4).xyz;\r\n}\r\n\r\n// NOTE: The odd ordering of the fields\r\n// ensures that the struct is 32 bytes wide,\r\n// given that vec3f has 16-byte padding\r\nstruct BvhNode\r\n{\r\n    aabb_min:  vec3f,\r\n    // If tri_count is 0, this is first_child\r\n    // otherwise this is tri_begin\r\n    tri_begin_or_first_child: u32,\r\n    aabb_max:  vec3f,\r\n    tri_count: u32\r\n    \r\n    // Second child is at index first_child + 1\r\n}\r\n\r\nstruct TlasNode\r\n{\r\n    aabb_min: vec3f,\r\n    aabb_max: vec3f,\r\n    // If is_leaf is true, this is mesh_instance,\r\n    // otherwise this is first_child\r\n    mesh_instance_or_first_child: u32,\r\n    is_leaf:   u32,  // 32-bit bool\r\n\r\n    // Second child is at index first_child + 1\r\n}\r\n\r\nstruct MeshInstance\r\n{\r\n    // Transform from world space to model space\r\n    inverse_transform: mat4x3f,\r\n    material: Material,\r\n    bvh_root: u32\r\n}\r\n\r\nstruct Scene\r\n{\r\n    env_map: vec2u,\r\n}\r\n\r\n// From: https://tavianator.com/2011/ray_box.html\r\n// For misses, t = f32_max\r\nfn ray_aabb_dst(ray: Ray, aabb_min: vec3f, aabb_max: vec3f)->f32\r\n{\r\n    let t_min: vec3f = (aabb_min - 0.001 - ray.ori) * ray.inv_dir;\r\n    let t_max: vec3f = (aabb_max + 0.001 - ray.ori) * ray.inv_dir;\r\n    let t1: vec3f = min(t_min, t_max);\r\n    let t2: vec3f = max(t_min, t_max);\r\n    let dst_far: f32  = min(min(t2.x, t2.y), t2.z);\r\n    let dst_near: f32 = max(max(t1.x, t1.y), t1.z);\r\n\r\n    let did_hit: bool = dst_far >= dst_near && dst_far > 0.0f;\r\n    return select(f32_max, dst_near, did_hit);\r\n}\r\n\r\n// From: https://www.shadertoy.com/view/MlGcDz\r\n// Triangle intersection. Returns { t, u, v }\r\n// For misses, t = f32_max\r\nfn ray_tri_dst(ray: Ray, v0: vec3f, v1: vec3f, v2: vec3f)->vec3f\r\n{\r\n    let v1v0 = v1 - v0;\r\n    let v2v0 = v2 - v0;\r\n    let rov0 = ray.ori - v0;\r\n\r\n    // Cramer's rule for solving p(t) = ro+t·rd = p(u,v) = vo + u·(v1-v0) + v·(v2-v1).\r\n    // The four determinants above have lots of terms in common. Knowing the changing\r\n    // the order of the columns/rows doesn't change the volume/determinant, and that\r\n    // the volume is dot(cross(a,b,c)), we can precompute some common terms and reduce\r\n    // it all to:\r\n    let n = cross(v1v0, v2v0);\r\n    let q = cross(rov0, ray.dir);\r\n    let d = 1.0 / dot(ray.dir, n);\r\n    let u = d * dot(-q, v2v0);\r\n    let v = d * dot(q, v1v0);\r\n    var t = d * dot(-n, rov0);\r\n\r\n    if min(u, v) < 0.0 || (u+v) > 1.0 { t = f32_max; }\r\n    return vec3f(t, u, v);\r\n}\r\n\r\nstruct HitInfo\r\n{\r\n    dst: f32,\r\n    normal: vec3f,\r\n    tex_coords: vec2f\r\n}\r\n\r\nconst MAX_BVH_DEPTH: u32 = 25;\r\nconst STACK_SIZE: u32 = (MAX_BVH_DEPTH + 1) * 8 * 8;\r\n// NOTE: First value in the stack is fictitious and\r\n// can be used to write any value.\r\n//var stack: array<u32, 26>;\r\nvar<workgroup> stack: array<u32, STACK_SIZE>;\r\n\r\nfn ray_scene_intersection(local_id: vec3u, ray: Ray)->HitInfo\r\n{\r\n    // Comment/Uncomment to test the performance of shared memory\r\n    // vs local array (registers or global memory)\r\n    //let offset: u32 = 0u;\r\n    let offset = (local_id.y * 8 + local_id.x) * (MAX_BVH_DEPTH + 1);\r\n\r\n    //var stack: array<u32, 26>;\r\n    var stack_idx: u32 = 2;\r\n    stack[0 + offset] = 0u;\r\n    stack[1 + offset] = 0u;\r\n\r\n    var num_boxes_hit: i32 = 0;\r\n\r\n    // t, u, v\r\n    var min_hit = vec3f(f32_max, 0.0f, 0.0f);\r\n    var tri_idx: u32 = 0;\r\n    while stack_idx > 1\r\n    {\r\n        stack_idx--;\r\n        let node = bvh_nodes[stack[stack_idx + offset]];\r\n\r\n        if node.tri_count > 0u  // Leaf node\r\n        {\r\n            let tri_begin = node.tri_begin_or_first_child;\r\n            let tri_count = node.tri_count;\r\n            for(var i: u32 = tri_begin; i < tri_begin + tri_count; i++)\r\n            {\r\n                let v0: vec3f = verts_pos[indices[i*3 + 0]];\r\n                let v1: vec3f = verts_pos[indices[i*3 + 1]];\r\n                let v2: vec3f = verts_pos[indices[i*3 + 2]];\r\n                let hit: vec3f = ray_tri_dst(ray, v0, v1, v2);\r\n                if hit.x < min_hit.x\r\n                {\r\n                    min_hit = hit;\r\n                    tri_idx = i;\r\n                }\r\n            }\r\n        }\r\n        else  // Non-leaf node\r\n        {\r\n            let left_child  = node.tri_begin_or_first_child;\r\n            let right_child = left_child + 1;\r\n            let left_child_node  = bvh_nodes[left_child];\r\n            let right_child_node = bvh_nodes[right_child];\r\n\r\n            let left_dst  = ray_aabb_dst(ray, left_child_node.aabb_min,  left_child_node.aabb_max);\r\n            let right_dst = ray_aabb_dst(ray, right_child_node.aabb_min, right_child_node.aabb_max);\r\n\r\n            // Push children onto the stack\r\n            // The closest child should be looked at\r\n            // first. This order is chosen so that it's more\r\n            // likely that the second child will never need\r\n            // to be queried\r\n\r\n            let left_first: bool = left_dst <= right_dst;\r\n            let push_left:  bool = left_dst < min_hit.x;\r\n            let push_right: bool = right_dst < min_hit.x;\r\n\r\n            if left_first\r\n            {\r\n                if push_right\r\n                {\r\n                    stack[stack_idx + offset] = right_child;\r\n                    stack_idx++;\r\n                }\r\n                \r\n                if push_left\r\n                {\r\n                    stack[stack_idx + offset] = left_child;\r\n                    stack_idx++;\r\n                }\r\n            }\r\n            else\r\n            {\r\n                if push_left\r\n                {\r\n                    stack[stack_idx + offset] = left_child;\r\n                    stack_idx++;\r\n                }\r\n\r\n                if push_right\r\n                {\r\n                    stack[stack_idx + offset] = right_child;\r\n                    stack_idx++;\r\n                }\r\n            }\r\n        }\r\n    }\r\n\r\n    var hit_info: HitInfo = HitInfo(min_hit.x, vec3f(0.0f), vec2f(0.0f));\r\n    if hit_info.dst != f32_max\r\n    {\r\n        let vert0: Vertex = verts[indices[tri_idx*3 + 0]];\r\n        let vert1: Vertex = verts[indices[tri_idx*3 + 1]];\r\n        let vert2: Vertex = verts[indices[tri_idx*3 + 2]];\r\n        let u = min_hit.y;\r\n        let v = min_hit.z;\r\n        let w = 1.0 - u - v;\r\n\r\n        hit_info.normal = vert0.normal*w + vert1.normal*u + vert2.normal*v;\r\n        hit_info.tex_coords = vert0.tex_coords*w + vert1.tex_coords*u + vert2.tex_coords*v;\r\n    }\r\n\r\n    return hit_info;\r\n}\r\n\r\n@compute\r\n@workgroup_size(8, 8, 1)\r\nfn cs_main(@builtin(local_invocation_id) local_id: vec3u, @builtin(global_invocation_id) global_id: vec3<u32>)\r\n{\r\n    var frag_coord = vec2f(global_id.xy) + 0.5f;\r\n    var output_dim = textureDimensions(output_texture).xy;\r\n    var resolution = vec2f(output_dim.xy);\r\n\r\n    var uv = frag_coord / resolution;\r\n    var coord = 2.0f * uv - 1.0f;\r\n    coord.y *= -resolution.y / resolution.x;\r\n    \r\n    var camera_look_at = normalize(vec3(coord, 1.0f));\r\n\r\n    var camera_ray = Ray(vec3f(0.0f, 0.0f, 0.0f), camera_look_at, 1.0f / camera_look_at);\r\n    camera_ray.ori = transform_point(camera_ray.ori, camera_transform);\r\n    camera_ray.dir = transform_dir(camera_ray.dir, camera_transform);\r\n    camera_ray.inv_dir = 1.0f / camera_ray.dir;\r\n\r\n    var hit = ray_scene_intersection(local_id, camera_ray);\r\n\r\n    var color = vec4f(select(vec3f(0.0f), hit.normal, hit.dst != f32_max), 1.0f);\r\n\r\n    if global_id.x < output_dim.x && global_id.y < output_dim.y\r\n    {\r\n        textureStore(output_texture, global_id.xy, color);\r\n    }\r\n}\r\n", "\r\n@group(0) @binding(0) var to_show: texture_2d<f32>;\r\n@group(0) @binding(1) var tex_sampler: sampler;\r\n\r\nstruct VertexOutput\r\n{\r\n    @builtin(position) pos: vec4f,\r\n    @location(0) tex_coords: vec2f\r\n}\r\n\r\n@vertex\r\nfn vs_main(@builtin(vertex_index) vert_idx: u32)->VertexOutput\r\n{\r\n    var pos = array<vec2f, 6>\r\n    (\r\n        vec2f(-1.0, -1.0),\r\n        vec2f(-1.0, 1.0),\r\n        vec2f(1.0, -1.0),\r\n        vec2f(1.0, -1.0),\r\n        vec2f(-1.0, 1.0),\r\n        vec2f(1.0, 1.0)\r\n    );\r\n\r\n    return VertexOutput(vec4f(pos[vert_idx], 0.0f, 1.0f), pos[vert_idx] * -0.5f + 0.5f);\r\n}\r\n\r\n@fragment\r\nfn fs_main(@location(0) tex_coords: vec2f)->@location(0) vec4f\r\n{\r\n    var texture_size: vec2u = textureDimensions(to_show).xy;\r\n    let aspect_ratio: f32 = f32(texture_size.y) / f32(texture_size.x);\r\n    let texture_scale = vec2f(1.0f / aspect_ratio, 1.0f);\r\n    //let texture_coord = texture_scale * (tex_coords - 0.5f) + 0.5f;\r\n    let texture_coord = tex_coords;\r\n\r\n    if (texture_coord.x < 0.0 || texture_coord.x > 1.0 ||\r\n        texture_coord.y < 0.0 || texture_coord.y > 1.0)\r\n    {\r\n        return vec4f(0.0f, 0.0f, 0.0f, 1.0f);\r\n    }\r\n\r\n    return textureSample(to_show, tex_sampler, tex_coords);\r\n}\r\n"];
