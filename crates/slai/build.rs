use minislang::{shader_slang::CompileTarget, SlangCompiler};
use std::path::PathBuf;
use std::str::FromStr;

pub fn main() {
    let mut slang = SlangCompiler::new(vec![PathBuf::from_str("./shaders").unwrap()]);
    stensor::register_shaders(&mut slang);

    let targets = [
        (CompileTarget::Wgsl, "wgsl"),
        (CompileTarget::Metal, "metal"),
        (CompileTarget::CudaSource, "cu"),
    ];

    for (target, ext) in targets {
        std::fs::create_dir_all(format!("./src/autogen/{ext}")).unwrap();
        slang.compile_all(target, "./shaders", "./src/autogen", &[]);
    }
}
