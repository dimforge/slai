use nalgebra::DVector;
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::GpuTensorView;

#[derive(Shader)]
#[shader(module = "slai::silu")]
/// Shader implementing the Silu activation function.
pub struct Silu<B: Backend> {
    pub silu: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct SiluArgs<'a, B: Backend> {
    shape_a: &'a B::Buffer<ViewShape>,
    shape_b: &'a B::Buffer<ViewShape>,
    in_out_a: B::BufferSlice<'a, f32>,
    in_b: B::BufferSlice<'a, f32>,
}

impl<B: Backend> Silu<B> {
    pub fn launch<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        in_out_h1: impl Into<GpuTensorView<'b, f32, B>>,
        in_h2: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let h1 = in_out_h1.into();
        let h2 = in_h2.into();
        shapes.insert(backend, h1.shape())?;
        shapes.insert(backend, h2.shape())?;
        let shape_a = shapes.get(h1.shape()).unwrap();
        let shape_b = shapes.get(h2.shape()).unwrap();

        let args = SiluArgs {
            shape_a,
            shape_b,
            in_out_a: h1.buffer(),
            in_b: h2.buffer(),
        };

        self.silu
            .launch(backend, pass, &args, [h1.len() as u32, 1, 1])
    }

    pub fn run_cpu(h1: &mut DVector<f32>, h2: &DVector<f32>) {
        // SwiGLU non-linearity.
        fn swish(x: f32, beta: f32) -> f32 {
            // This is the swish function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
            x / (1.0 + (-beta * x).exp())
        }

        h1.zip_apply(h2, |h, h2| *h = h2 * swish(*h, 1.0));
    }
}

#[cfg(test)]
mod test {
    use nalgebra::DVector;
    #[cfg(feature = "cuda")]
    use slang_hal::backend::Cuda;
    use slang_hal::backend::WebGpu;
    use slang_hal::backend::{Backend, Encoder};
    use slang_hal::re_exports::minislang::SlangCompiler;
    use slang_hal::Shader;
    use stensor::shapes::ViewShapeBuffers;
    use stensor::tensor::GpuTensor;
    use wgpu::{BufferUsages, Features, Limits};

    #[cfg(feature = "cuda")]
    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_silu_cuda() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = false;
        gpu_silu_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_silu_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_silu_generic(backend).await;
    }

    async fn gpu_silu_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        let silu = super::Silu::from_backend(&backend, &compiler).unwrap();
        let mut shapes = ViewShapeBuffers::new(&backend);

        const LEN: u32 = 1757;

        let h1 = DVector::new_random(LEN as usize);
        let h2 = DVector::new_random(LEN as usize);
        let mut h1_read = DVector::zeros(LEN as usize);

        let gpu_h1 = GpuTensor::vector(
            &backend,
            &h1,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();
        let gpu_h2 = GpuTensor::vector(&backend, &h2, BufferUsages::STORAGE).unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        silu.launch(&backend, &mut shapes, &mut pass, &gpu_h1, &gpu_h2)
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_h1.buffer(), h1_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = h1;
        super::Silu::<WebGpu>::run_cpu(&mut cpu_result, &h2);

        approx::assert_relative_eq!(h1_read, cpu_result, epsilon = 1.0e-5);
    }
}
