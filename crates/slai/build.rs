#[cfg(feature = "comptime_checks")]
use {
    minislang::{shader_slang::CompileTarget, SlangCompiler},
    std::env,
    std::path::{Path, PathBuf},
    std::str::FromStr,
};

#[cfg(not(feature = "comptime_checks"))]
pub fn main() {}

#[cfg(feature = "comptime_checks")]
pub fn main() {
    let mut slang = SlangCompiler::new(vec![PathBuf::from_str("./shaders").unwrap()]);
    stensor::register_shaders(&mut slang);

    let targets = [
        (CompileTarget::Wgsl, "wgsl"),
        (CompileTarget::Metal, "metal"),
        (CompileTarget::CudaSource, "cu"),
    ];

    let out_dir = env::var("OUT_DIR").expect("Couldn't determine output directory.");
    let out_dir = Path::new(&out_dir);
    let autogen_target = out_dir.join("autogen");

    for (target, ext) in targets {
        std::fs::create_dir_all(&autogen_target).unwrap();
        slang.compile_all(target, "./shaders", &autogen_target, &[]);
    }
}
