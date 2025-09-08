use nalgebra::DVector;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::GpuTensorView;

#[derive(Shader)]
#[shader(module = "slai::layernorm")]
/// Shader implementing the layer normalization kernel.
pub struct LayerNorm<B: Backend> {
    pub layernorm_cols: GpuFunction<B>,
    pub layernorm_rows: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct LayerNormArgs<'a, B: Backend> {
    in_shape: &'a B::Buffer<ViewShape>,
    out_shape: &'a B::Buffer<ViewShape>,
    input: B::BufferSlice<'a, f32>,
    output: B::BufferSlice<'a, f32>,
}

impl<B: Backend> LayerNorm<B> {
    pub fn launch_cols<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        output: impl Into<GpuTensorView<'b, f32, B>>,
        input: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let input = input.into();
        let output = output.into();
        let shape_input = input.shape();
        let shape_output = output.shape();
        assert_eq!(
            shape_input.size, shape_output.size,
            "LayerNorm: dimension mismatch."
        );

        let grid = [
            shape_input.size[1],
            shape_input.size[2],
            shape_input.size[3],
        ];

        shapes.insert(backend, shape_input)?;
        shapes.insert(backend, shape_output)?;
        let in_shape = shapes.get(shape_input).unwrap();
        let out_shape = shapes.get(shape_output).unwrap();

        let args = LayerNormArgs {
            in_shape,
            out_shape,
            input: input.buffer(),
            output: output.buffer(),
        };
        self.layernorm_cols.launch(backend, pass, &args, grid)
    }

    pub fn launch_rows<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        output: impl Into<GpuTensorView<'b, f32, B>>,
        input: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let input = input.into();
        let output = output.into();
        let shape_input = input.shape();
        let shape_output = output.shape();

        assert_eq!(
            shape_input.size, shape_output.size,
            "LayerNorm: dimension mismatch."
        );

        let grid = [
            shape_input.size[0],
            shape_input.size[2],
            shape_input.size[3],
        ];

        shapes.insert(backend, shape_input)?;
        shapes.insert(backend, shape_output)?;
        let in_shape = shapes.get(shape_input).unwrap();
        let out_shape = shapes.get(shape_output).unwrap();

        let args = LayerNormArgs {
            in_shape,
            out_shape,
            input: input.buffer(),
            output: output.buffer(),
        };
        self.layernorm_rows.launch_grid(backend, pass, &args, grid)
    }

    /// The layernorm function.
    ///
    /// See <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html> for details on the
    /// math.
    pub fn run_cpu(res: &mut DVector<f32>, v: &DVector<f32>) {
        const NUDGE_FACTOR: f32 = 1.0e-5;
        let mean = v.mean();
        res.zip_apply(v, |y, v| *y = v - mean);
        let variance = res.norm_squared() / (res.len() as f32);
        let scale = 1.0 / (variance + NUDGE_FACTOR).sqrt();
        *res *= scale;
    }
}

#[cfg(test)]
mod test {
    use crate::ops::LayerNorm;
    use nalgebra::DVector;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use slang_hal::re_exports::minislang::SlangCompiler;
    use slang_hal::Shader;
    use stensor::shapes::ViewShapeBuffers;
    use stensor::tensor::GpuTensor;
    use wgpu::{BufferUsages, Features, Limits};

    #[cfg(feature = "cuda")]
    use slang_hal::cuda::Cuda;

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_layernorm_cuda() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = false;
        gpu_layernorm_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_layernorm_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_layernorm_generic(backend).await;
    }

    async fn gpu_layernorm_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);
        let layernorm = super::LayerNorm::from_backend(&backend, &compiler).unwrap();
        let mut shapes = ViewShapeBuffers::new(&backend);

        const LEN: u32 = 1757;

        let v0 = DVector::new_random(LEN as usize);
        let out = DVector::new_random(LEN as usize);
        let mut out_read = DVector::zeros(LEN as usize);
        let gpu_v0 = GpuTensor::vector(
            &backend,
            &v0,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let gpu_out = GpuTensor::vector(
            &backend,
            &v0,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        layernorm
            .launch_cols(&backend, &mut shapes, &mut pass, &gpu_v0, &gpu_out)
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_out.buffer(), out_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = out;
        LayerNorm::<WebGpu>::run_cpu(&mut cpu_result, &v0);

        approx::assert_relative_eq!(out_read, cpu_result, epsilon = 1.0e-5);
    }
}
