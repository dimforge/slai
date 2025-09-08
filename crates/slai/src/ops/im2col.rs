use crate::context::{GGML_0, GGML_1, GGML_2, GGML_3};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::tensor::{GpuScalar, GpuTensorView};

#[derive(Copy, Clone, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable, Debug, Default)]
#[repr(C)]
pub struct Im2ColConfig {
    pub batch_offset: u32,
    pub offset_delta: u32,
    pub ic: u32,
    pub iw: u32,
    pub ih: u32,
    pub ow: u32,
    pub oh: u32,
    pub kw: u32,
    pub kh: u32,
    pub pelements: u32,
    pub chw: u32,
    pub s0: u32,
    pub s1: u32,
    pub p0: u32,
    pub p1: u32,
    pub d0: u32,
    pub d1: u32,
    pub padding: [u32; 3],
}

#[derive(Shader)]
#[shader(module = "slai::im2col")]
pub struct Im2Col<B: Backend> {
    pub im2col: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct Im2ColArgs<'a, B: Backend> {
    params: &'a B::Buffer<Im2ColConfig>,
    in_tensor: B::BufferSlice<'a, f32>,
    out_tensor: B::BufferSlice<'a, f32>,
}

impl<B: Backend> Im2Col<B> {
    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    // kernel: [OC，IC, KH, KW]
    // input: [N, IC, IH, IW]
    // result: [N, OH, OW, IC*KH*KW]
    pub fn launch<'b>(
        &self,
        backend: &B,
        pass: &mut B::Pass,
        params: &mut GpuScalar<Im2ColConfig, B>,
        result: impl Into<GpuTensorView<'b, f32, B>>,
        kernel: impl Into<GpuTensorView<'b, f32, B>>,
        input: impl Into<GpuTensorView<'b, f32, B>>,
        s0: u32,
        s1: u32,
        p0: u32,
        p1: u32,
        d0: u32,
        d1: u32,
        is_2d: bool,
    ) -> Result<(), B::Error> {
        let result = result.into();
        let kernel = kernel.into();
        let input = input.into();

        let ishape = input.shape();
        let kshape = kernel.shape();
        let rshape = result.shape();

        let ic = ishape.size[if is_2d { GGML_2 } else { GGML_1 }];
        let ih = if is_2d { ishape.size[GGML_1] } else { 1 };
        let iw = ishape.size[GGML_0];

        let kh = if is_2d { kshape.size[GGML_1] } else { 1 };
        let kw = kshape.size[GGML_0];

        let oh = if is_2d { rshape.size[GGML_2] } else { 1 };
        let ow = rshape.size[GGML_1];

        let offset_delta = ishape.stride[if is_2d { GGML_2 } else { GGML_1 }];
        let batch_offset = ishape.stride[if is_2d { GGML_3 } else { GGML_2 }];

        let pelements = ow * kw * kh;
        let chw = ic * kh * kw;

        let config = Im2ColConfig {
            batch_offset,
            offset_delta,
            ic,
            iw,
            ih,
            ow,
            oh,
            kw,
            kh,
            pelements,
            chw,
            s0,
            s1,
            p0,
            p1,
            d0,
            d1,
            padding: [0; 3],
        };

        backend.write_buffer(params.buffer_mut(), &[config])?;

        let args = Im2ColArgs {
            params: params.buffer(),
            in_tensor: input.buffer(),
            out_tensor: result.buffer(),
        };

        let batch = ishape.size[if is_2d { 3 } else { 2 }];
        let grid = [(ow * kw * kh).div_ceil(32), oh, batch * ic];

        self.im2col.launch_grid(backend, pass, &args, grid)?;

        Ok(())
    }
}
