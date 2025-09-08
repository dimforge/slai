#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::result_large_err)]

use slang_hal::re_exports::minislang::SlangCompiler;

pub mod context;
pub mod gguf;
pub mod models;
pub mod ops;
pub mod quantization;
pub mod quantized_matrix;
pub mod tensor_cache;

pub const SLANG_SRC_DIR: include_dir::Dir<'_> =
    include_dir::include_dir!("$CARGO_MANIFEST_DIR/shaders");
pub fn register_shaders(compiler: &mut SlangCompiler) {
    stensor::register_shaders(compiler);
    compiler.add_dir(SLANG_SRC_DIR.clone());
}

pub mod re_exports {
    pub use slang_hal;
    pub use stensor;
}
