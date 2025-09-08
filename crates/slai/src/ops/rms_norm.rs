use nalgebra::{DVector, Dyn, Storage, Vector};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::{GpuScalar, GpuTensorView};

#[derive(Shader)]
#[shader(module = "slai::rms_norm")]
/// Shader implementing the RMS norm kernel.
pub struct RmsNorm<B: Backend> {
    pub rms_norm: GpuFunction<B>,
}

#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct RmsNormConfig {
    pub nudge_factor: f32,
    // NOTE: due to slang’s alignment.
    pub _padding: [u32; 3],
}

#[derive(ShaderArgs)]
struct RmsNormArgs<'a, B: Backend> {
    shape_v: &'a B::Buffer<ViewShape>,
    shape_w: &'a B::Buffer<ViewShape>,
    shape_out: &'a B::Buffer<ViewShape>,
    v: B::BufferSlice<'a, f32>,
    w: B::BufferSlice<'a, f32>,
    out: B::BufferSlice<'a, f32>,
    config: &'a B::Buffer<RmsNormConfig>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn launch<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        config: &GpuScalar<RmsNormConfig, B>,
        result: impl Into<GpuTensorView<'b, f32, B>>,
        value: impl Into<GpuTensorView<'b, f32, B>>,
        weight: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let value = value.into();
        let weight = weight.into();
        let result = result.into();

        shapes.insert(backend, value.shape())?;
        shapes.insert(backend, weight.shape())?;
        shapes.insert(backend, result.shape())?;
        let shape_v = shapes.get(value.shape()).unwrap();
        let shape_w = shapes.get(weight.shape()).unwrap();
        let shape_out = shapes.get(result.shape()).unwrap();

        let args = RmsNormArgs {
            shape_v,
            shape_w,
            shape_out,
            v: value.buffer(),
            w: weight.buffer(),
            out: result.buffer(),
            config: config.buffer(),
        };
        self.rms_norm.launch(backend, pass, &args, [1; 3])
    }

    pub fn run_cpu<SW: Storage<f32, Dyn>>(
        out: &mut DVector<f32>,
        a: &DVector<f32>,
        w: &Vector<f32, Dyn, SW>,
    ) {
        const NUDGE_FACTOR: f32 = 1.0e-5;
        let rms = 1.0 / (a.norm_squared() / (a.nrows() as f32) + NUDGE_FACTOR).sqrt();
        out.zip_zip_apply(a, w, |o, a, w| *o = (a * rms) * w);
    }
}

#[cfg(test)]
mod test {
    use crate::ops::{RmsNorm, RmsNormConfig};
    use nalgebra::DVector;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    #[cfg(feature = "cuda")]
    use slang_hal::cuda::Cuda;
    use slang_hal::re_exports::minislang::SlangCompiler;
    use slang_hal::Shader;
    use stensor::shapes::ViewShapeBuffers;
    use stensor::tensor::GpuTensor;
    use wgpu::{BufferUsages, Features, Limits};

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rms_norm_cuda() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = false;
        gpu_rms_norm_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rms_norm_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_rms_norm_generic(backend).await;
    }

    async fn gpu_rms_norm_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);
        let rmsnorm = super::RmsNorm::from_backend(&backend, &compiler).unwrap();
        let mut shapes = ViewShapeBuffers::new(&backend);

        const LEN: u32 = 1757;

        let result = DVector::new_random(LEN as usize);
        let value = DVector::new_random(LEN as usize);
        let weight = DVector::new_random(LEN as usize);
        let mut gpu_result_read = DVector::zeros(LEN as usize);

        let gpu_result = GpuTensor::vector(
            &backend,
            &result,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let gpu_value = GpuTensor::vector(&backend, &value, BufferUsages::STORAGE).unwrap();
        let gpu_weight = GpuTensor::vector(&backend, &weight, BufferUsages::STORAGE).unwrap();
        let config = GpuTensor::scalar(
            &backend,
            RmsNormConfig {
                nudge_factor: 1.0e-6,
                _padding: [0; 3],
            },
            BufferUsages::UNIFORM,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        rmsnorm
            .launch(
                &backend,
                &mut shapes,
                &mut pass,
                &config,
                &gpu_value,
                &gpu_weight,
                &gpu_result,
            )
            .unwrap();
        drop(pass);
        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_result.buffer(), gpu_result_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = result;
        RmsNorm::<WebGpu>::run_cpu(&mut cpu_result, &value, &weight);

        approx::assert_relative_eq!(gpu_result_read, cpu_result, epsilon = 1.0e-4);
    }
}
