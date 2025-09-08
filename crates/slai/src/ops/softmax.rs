use nalgebra::{Dyn, StorageMut, Vector};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::GpuTensorView;

#[derive(Shader)]
#[shader(module = "slai::softmax")]
/// Shader implementing the softmax kernel.
pub struct SoftMax<B: Backend> {
    pub softmax_cols: GpuFunction<B>,
}

#[derive(ShaderArgs)]
struct SoftMaxArgs<'a, B: Backend> {
    shape: &'a B::Buffer<ViewShape>,
    in_out_mat: B::BufferSlice<'a, f32>,
}

impl<B: Backend> SoftMax<B> {
    pub fn launch_cols<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        in_out_mat: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let in_out_mat = in_out_mat.into();
        shapes.insert(backend, in_out_mat.shape())?;
        let shape_buf = shapes.get(in_out_mat.shape()).unwrap();

        let args = SoftMaxArgs {
            shape: shape_buf,
            in_out_mat: in_out_mat.buffer(),
        };
        let size = in_out_mat.shape().size;
        self.softmax_cols
            .launch_grid(backend, pass, &args, [size[1], size[2], size[3]])
    }

    /// The softmax function.
    ///
    /// Converts a set of real number into a probability distribution.
    /// See <https://fr.wikipedia.org/wiki/Fonction_softmax>
    pub fn run_cpu<S: StorageMut<f32, Dyn>>(vals: &mut Vector<f32, Dyn, S>) {
        // Note that llama2.c also introduces a bias based on the max value
        // to improve numerical stability. So it is effectively computing:
        // softmax(z) = (e^z - max) / (e^z - max).sum()
        let max_val = vals.max();
        let mut sum = 0.0;

        vals.apply(|x| {
            *x = (*x - max_val).exp();
            sum += *x;
        });

        *vals /= sum;
    }
}

#[cfg(test)]
mod test {
    use crate::ops::SoftMax;
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
    async fn gpu_softmax_cuda() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = false;
        gpu_softmax_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_softmax_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_softmax_generic(backend).await;
    }

    async fn gpu_softmax_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        let softmax = super::SoftMax::from_backend(&backend, &compiler).unwrap();
        let mut shapes = ViewShapeBuffers::new(&backend);

        const LEN: u32 = 1757;

        let v0 = DVector::new_random(LEN as usize);
        let mut gpu_v0_read = DVector::zeros(LEN as usize);
        let gpu_v0 = GpuTensor::vector(
            &backend,
            &v0,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        )
        .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        softmax
            .launch_cols(&backend, &mut shapes, &mut pass, gpu_v0.as_view())
            .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_v0.buffer(), gpu_v0_read.as_mut_slice())
            .await
            .unwrap();

        let mut cpu_result = v0;
        SoftMax::<WebGpu>::run_cpu(&mut cpu_result);

        approx::assert_relative_eq!(gpu_v0_read, cpu_result, epsilon = 1.0e-7);
    }
}
