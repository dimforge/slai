use nalgebra::{vector, DVector, DVectorViewMut, Rotation2};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::{GpuScalar, GpuTensorView};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RoPEVariant {
    // The original version of RoPE, where the rotated entries are adjacent.
    Original,
    // A variant of RoPE where the rotated entries are separated by `head_size / 2` elements.
    Neox,
}

#[derive(Shader)]
#[shader(module = "slai::rope")]
/// Shader implementing the Rotary Positional Encoding kernel.
pub struct RoPE<B: Backend> {
    pub rope_neox: GpuFunction<B>,
    pub rope: GpuFunction<B>,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone)]
/// Parameters needed to run the [`RoPE`] kernel. Matches the layout of the
/// corresponding WGSL struct.
pub struct RoPEConfig {
    pub head_size: u32,
    pub kv_dim: u32,
    pub pos: u32,
    pub base_freq: f32,
}

#[derive(ShaderArgs)]
struct RoPEArgs<'a, B: Backend> {
    shape_q: &'a B::Buffer<ViewShape>,
    shape_k: &'a B::Buffer<ViewShape>,
    config: &'a B::Buffer<RoPEConfig>,
    in_out_q: B::BufferSlice<'a, f32>,
    in_out_k: B::BufferSlice<'a, f32>,
}

impl<B: Backend> RoPE<B> {
    pub fn launch<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        variant: RoPEVariant,
        config: &GpuScalar<RoPEConfig, B>,
        in_out_q: impl Into<GpuTensorView<'b, f32, B>>,
        in_out_k: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let in_out_q = in_out_q.into();
        let in_out_k = in_out_k.into();

        assert_eq!(in_out_q.len() % 2, 0);
        assert_eq!(in_out_k.len() % 2, 0);
        assert!(
            in_out_q.len() >= in_out_k.len(),
            "The Query vector must be larger than, or as large as, the Key vector."
        );

        shapes.insert(backend, in_out_q.shape()).unwrap();
        shapes.insert(backend, in_out_k.shape()).unwrap();
        let shape_q = shapes.get(in_out_q.shape()).unwrap();
        let shape_k = shapes.get(in_out_k.shape()).unwrap();

        let pipeline = match variant {
            RoPEVariant::Original => &self.rope,
            RoPEVariant::Neox => &self.rope_neox,
        };

        let args = RoPEArgs {
            shape_q,
            shape_k,
            config: config.buffer(),
            in_out_k: in_out_k.buffer(),
            in_out_q: in_out_q.buffer(),
        };

        // Use `q` as the reference for the workgroup count since it is a bigger vector.
        pipeline.launch(backend, pass, &args, [in_out_q.len() as u32 / 2, 1, 1])
    }

    // Rotary Positional Encoding (RoPE): complex-valued rotate q and k in each head.
    pub fn run_cpu(
        q: &mut DVector<f32>,
        k: &mut DVectorViewMut<f32>,
        head_size: usize,
        dim: usize,
        kv_dim: usize,
        pos: usize,
    ) {
        for i in (0..dim).step_by(2) {
            // For RoPE, we have one rotation matrix like https://youtu.be/Mn_9W1nCFLo?si=GLIXuFLGVG8q6v2u&t=1963
            // for each head. So we need to transform `i` into the corresponding index within
            // the head.
            let head_dim = (i % head_size) as f32;
            // Not that the formulae from the video linked above would be:
            //     10000.0.powf(-2.0 * ((i / 2) as f32 - 1.0) / dim as f32)
            // Although in the paper shown in the video, their index is 1-based which his why thy
            // have to subtract 1.0 whereas we don’t need to.The `i / 2` and multiplication by 2.0
            // are both accounted for by stepping only on even values for `i`.
            // Therefore, the formulae below is equivalent to the RoPE paper’s formulae.
            let theta = 10000.0_f32.powf(-head_dim / head_size as f32);
            let m_theta = pos as f32 * theta;
            let rot = Rotation2::new(m_theta);

            let qi = vector![q[i], q[i + 1]];
            let mut out_q = q.fixed_rows_mut::<2>(i);
            out_q.copy_from(&(rot * qi));

            // When i >= kv_dim, we are done rotating all the elements from the keys. That’s
            // because there are less key heads than query heads, but each key head sub-vector has
            // the same dimension as the query head (they loose dimension when multiplied with the
            // key weight matrices).
            if i < kv_dim {
                let ki = vector![k[i], k[i + 1]];
                let mut out_k = k.fixed_rows_mut::<2>(i);
                out_k.copy_from(&(rot * ki));
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::RoPEConfig;
    use crate::ops::{RoPE, RoPEVariant};
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
    async fn gpu_rope_cuda() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = false;
        gpu_rope_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_rope_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_rope_generic(backend).await;
    }

    async fn gpu_rope_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        let rope = super::RoPE::from_backend(&backend, &compiler).unwrap();
        let mut shapes = ViewShapeBuffers::new(&backend);

        const HEAD_SIZE: u32 = 128;
        const LEN_Q: u32 = 13 * HEAD_SIZE;
        const LEN_K: u32 = 9 * HEAD_SIZE;

        let rope_indices = RoPEConfig {
            head_size: HEAD_SIZE,
            kv_dim: LEN_K,
            pos: 10,
            base_freq: 1.0e4,
        };

        let mut q = DVector::new_random(LEN_Q as usize);
        let mut k = DVector::new_random(LEN_K as usize);
        let mut result_q = DVector::zeros(LEN_Q as usize);
        let mut result_k = DVector::zeros(LEN_K as usize);

        let gpu_indices = GpuTensor::scalar(&backend, rope_indices, BufferUsages::UNIFORM).unwrap();
        let gpu_q = GpuTensor::vector(&backend, &q, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .unwrap();
        let gpu_k = GpuTensor::vector(&backend, &k, BufferUsages::STORAGE | BufferUsages::COPY_SRC)
            .unwrap();

        let mut encoder = backend.begin_encoding();
        let mut pass = encoder.begin_pass();
        rope.launch(
            &backend,
            &mut shapes,
            &mut pass,
            RoPEVariant::Original,
            &gpu_indices,
            &gpu_q,
            &gpu_k,
        )
        .unwrap();
        drop(pass);

        backend.submit(encoder).unwrap();
        backend.synchronize().unwrap();

        backend
            .slow_read_buffer(gpu_q.buffer(), result_q.as_mut_slice())
            .await
            .unwrap();
        backend
            .slow_read_buffer(gpu_k.buffer(), result_k.as_mut_slice())
            .await
            .unwrap();

        RoPE::<WebGpu>::run_cpu(
            &mut q,
            &mut k.rows_mut(0, LEN_K as usize),
            rope_indices.head_size as usize,
            LEN_Q as usize,
            rope_indices.kv_dim as usize,
            rope_indices.pos as usize,
        );

        // TODO: why is the epsilon so high? Is it a difference in sin/cos implementations?
        approx::assert_relative_eq!(result_q, q, epsilon = 1.0e-5);
        approx::assert_relative_eq!(result_k, k, epsilon = 1.0e-5);
    }
}
