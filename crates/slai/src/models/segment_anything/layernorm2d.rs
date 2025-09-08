use crate::context::LlmContext;
use crate::tensor_cache::CachedTensor;
use slang_hal::backend::Backend;
use stensor::tensor::GpuTensor;

pub fn sam_layernorm_2d<B: Backend>(
    ctxt: &mut LlmContext<B>,
    layer: &GpuTensor<f32, B>,
    n_channels: u32,
    w: &GpuTensor<f32, B>,
    b: &GpuTensor<f32, B>,
    eps: f32,
) -> Result<CachedTensor<f32, B>, B::Error> {
    // LayerNorm2d
    // normalize along channel dimension
    let layer = ctxt.contiguous(layer.permute_ggml([1, 2, 0, 3]))?;
    let layer = ctxt.layernorm(&layer, eps)?;
    let layer = layer.permute_ggml([2, 0, 1, 3]);

    let w = ctxt.repeat(w.reshape_ggml([1, 1, n_channels, 1]), layer)?;
    let b = ctxt.repeat(b.reshape_ggml([1, 1, n_channels, 1]), layer)?;
    let w_layer = ctxt.mul(&w, layer)?;
    ctxt.add(&w_layer, &b)
}
