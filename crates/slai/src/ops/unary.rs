use nalgebra::{Dyn, StorageMut, Vector, Vector4};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{ViewShape, ViewShapeBuffers};
use stensor::tensor::{GpuScalar, GpuTensorView};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[non_exhaustive]
/// Listing of all unary operations that can be applied by the [`Unary`] kernel.
pub enum UnaryOp {
    Abs,
    Sgn,
    Neg,
    Step,
    Elu,
    Gelu,
    GeluQuick,
    Silu,
    Tanh,
    Sin,
    Cos,
    Relu,
    Sigmoid,
    HardSigmoid,
    // HardSwish,
    Sqr,
    Sqrt,
    Log,
    // Unary ops with extra args.
    LeakyRelu,
    Clamp,
    Scale,
    AddScalar, // Named GGML_OP_ADD1 in ggml.
}

impl UnaryOp {
    const fn has_args(self) -> bool {
        match self {
            Self::Abs
            | Self::Sgn
            | Self::Neg
            | Self::Step
            | Self::Elu
            | Self::Gelu
            | Self::GeluQuick
            | Self::Silu
            | Self::Tanh
            | Self::Relu
            | Self::Sigmoid
            | Self::HardSigmoid
            // | Self::HardSwish
            | Self::Sqr
            | Self::Sqrt
            | Self::Log
            | Self::Sin
            | Self::Cos => false,
            Self::LeakyRelu | Self::Clamp | Self::Scale | Self::AddScalar => true,
        }
    }

    pub fn eval(self, x: f32, args: Vector4<f32>) -> f32 {
        match self {
            Self::Abs => x.abs(),
            Self::Sgn => x.signum(),
            Self::Neg => -x,
            Self::Step => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Elu => {
                if x > 0.0 {
                    x
                } else {
                    x.exp() - 1.0
                }
            }
            Self::Gelu => {
                const GELU_COEF_A: f32 = 0.044715;
                const SQRT_2_OVER_PI: f32 = 0.7978846;
                0.5 * x * (1.0 + (SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x)).tanh())
            }
            Self::GeluQuick => {
                const GELU_QUICK_COEF: f32 = -1.702;
                x * (1.0 / (1.0 + (GELU_QUICK_COEF * x).exp()))
            }
            Self::Silu => x / (1.0 + (-x).exp()),
            Self::Tanh => x.tanh(),
            Self::Relu => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Self::HardSigmoid => 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            // Self::HardSwish => x * 1.0f32.min(0.0f32.max((x + 3.0) / 6.0)),
            Self::Sqr => x * x,
            Self::Sqrt => x.sqrt(),
            Self::Sin => x.sin(),
            Self::Cos => x.cos(),
            Self::Log => x.ln(),
            Self::LeakyRelu => x.max(0.0) + x.min(0.0) * args.x,
            Self::Clamp => x.clamp(args.x, args.y),
            Self::Scale => x * args.x,
            Self::AddScalar => x + args.x,
        }
    }
}

/// Shader implementing various unary operations selected with [`UnaryOp`].
#[derive(Shader)]
#[shader(module = "slai::unary")]
pub struct Unary<B: Backend> {
    pub abs_op: GpuFunction<B>,
    pub abs_inplace: GpuFunction<B>,
    pub sgn_op: GpuFunction<B>,
    pub sgn_inplace: GpuFunction<B>,
    pub neg_op: GpuFunction<B>,
    pub neg_inplace: GpuFunction<B>,
    pub step_op: GpuFunction<B>,
    pub step_inplace: GpuFunction<B>,
    pub elu_op: GpuFunction<B>,
    pub elu_inplace: GpuFunction<B>,
    pub gelu_op: GpuFunction<B>,
    pub gelu_inplace: GpuFunction<B>,
    pub gelu_quick_op: GpuFunction<B>,
    pub gelu_quick_inplace: GpuFunction<B>,
    pub silu_op: GpuFunction<B>,
    pub silu_inplace: GpuFunction<B>,
    pub tanh_op: GpuFunction<B>,
    pub tanh_inplace: GpuFunction<B>,
    pub relu_op: GpuFunction<B>,
    pub relu_inplace: GpuFunction<B>,
    pub sigmoid_op: GpuFunction<B>,
    pub sigmoid_inplace: GpuFunction<B>,
    pub hard_sigmoid_op: GpuFunction<B>,
    pub hard_sigmoid_inplace: GpuFunction<B>,
    // pub hard_swish_op: GpuFunction<B>,
    // pub hard_swish_inplace: GpuFunction<B>,
    pub sqr_op: GpuFunction<B>,
    pub sqr_inplace: GpuFunction<B>,
    pub sqrt_op: GpuFunction<B>,
    pub sqrt_inplace: GpuFunction<B>,
    pub log_op: GpuFunction<B>,
    pub log_inplace: GpuFunction<B>,
    pub leaky_relu_op: GpuFunction<B>,
    pub leaky_relu_inplace: GpuFunction<B>,
    pub clamp_op: GpuFunction<B>,
    pub clamp_inplace: GpuFunction<B>,
    pub scale_op: GpuFunction<B>,
    pub scale_inplace: GpuFunction<B>,
    pub add_scalar_op: GpuFunction<B>,
    pub add_scalar_inplace: GpuFunction<B>,
    pub sin_op: GpuFunction<B>,
    pub sin_inplace: GpuFunction<B>,
    pub cos_op: GpuFunction<B>,
    pub cos_inplace: GpuFunction<B>,
}

// TODO: have ShaderArgs support Option<&'a B::Buffer> so we don't have to
//       use two structs heer.
#[derive(ShaderArgs)]
struct UnaryArgs<'a, B: Backend> {
    shape_src: &'a B::Buffer<ViewShape>,
    src: B::BufferSlice<'a, f32>,
    shape_dst: Option<&'a B::Buffer<ViewShape>>,
    dst: Option<B::BufferSlice<'a, f32>>,
    args: Option<&'a B::Buffer<Vector4<f32>>>,
}

impl<B: Backend> Unary<B> {
    pub fn launch_inplace<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        op: UnaryOp,
        src: impl Into<GpuTensorView<'b, f32, B>>,
        args: Option<&'b GpuScalar<Vector4<f32>, B>>,
    ) -> Result<(), B::Error> {
        let src = src.into();
        shapes.insert(backend, src.shape())?;
        let shape_src = shapes.get(src.shape()).unwrap();

        assert_eq!(
            op.has_args(),
            args.is_some(),
            "Unary ops argument mismatch."
        );

        let pipeline = match op {
            UnaryOp::Abs => &self.abs_inplace,
            UnaryOp::Sgn => &self.sgn_inplace,
            UnaryOp::Neg => &self.neg_inplace,
            UnaryOp::Step => &self.step_inplace,
            UnaryOp::Elu => &self.elu_inplace,
            UnaryOp::Gelu => &self.gelu_inplace,
            UnaryOp::GeluQuick => &self.gelu_quick_inplace,
            UnaryOp::Silu => &self.silu_inplace,
            UnaryOp::Tanh => &self.tanh_inplace,
            UnaryOp::Relu => &self.relu_inplace,
            UnaryOp::Sigmoid => &self.sigmoid_inplace,
            UnaryOp::HardSigmoid => &self.hard_sigmoid_inplace,
            // UnaryOp::HardSwish => &self.hard_swish_inplace,
            UnaryOp::Sqr => &self.sqr_inplace,
            UnaryOp::Sqrt => &self.sqrt_inplace,
            UnaryOp::Sin => &self.sin_inplace,
            UnaryOp::Cos => &self.cos_inplace,
            UnaryOp::Log => &self.log_inplace,
            UnaryOp::LeakyRelu => &self.leaky_relu_inplace,
            UnaryOp::Clamp => &self.clamp_inplace,
            UnaryOp::Scale => &self.scale_inplace,
            UnaryOp::AddScalar => &self.add_scalar_inplace,
        };

        let args = UnaryArgs {
            shape_src,
            src: src.buffer(),
            shape_dst: None,
            dst: None,
            args: args.map(|a| a.buffer()),
        };
        pipeline.launch_capped(backend, pass, &args, src.len() as u32)
    }

    pub fn launch<'a, 'b>(
        &'a self,
        backend: &B,
        shapes: &mut ViewShapeBuffers<B>,
        pass: &mut B::Pass,
        op: UnaryOp,
        dest: impl Into<GpuTensorView<'b, f32, B>>,
        src: impl Into<GpuTensorView<'b, f32, B>>,
        args: Option<&'b GpuScalar<Vector4<f32>, B>>,
    ) -> Result<(), B::Error> {
        let dest = dest.into();
        let src = src.into();
        shapes.insert(backend, dest.shape())?;
        shapes.insert(backend, src.shape())?;
        let shape_dest = shapes.get(dest.shape()).unwrap();
        let shape_src = shapes.get(src.shape()).unwrap();

        assert_eq!(
            op.has_args(),
            args.is_some(),
            "Unary ops argument mismatch."
        );

        let pipeline = match op {
            UnaryOp::Abs => &self.abs_op,
            UnaryOp::Sgn => &self.sgn_op,
            UnaryOp::Neg => &self.neg_op,
            UnaryOp::Step => &self.step_op,
            UnaryOp::Elu => &self.elu_op,
            UnaryOp::Gelu => &self.gelu_op,
            UnaryOp::GeluQuick => &self.gelu_quick_op,
            UnaryOp::Silu => &self.silu_op,
            UnaryOp::Tanh => &self.tanh_op,
            UnaryOp::Relu => &self.relu_op,
            UnaryOp::Sigmoid => &self.sigmoid_op,
            UnaryOp::HardSigmoid => &self.hard_sigmoid_op,
            // UnaryOp::HardSwish => &self.hard_swish_op,
            UnaryOp::Sqr => &self.sqr_op,
            UnaryOp::Sqrt => &self.sqrt_op,
            UnaryOp::Sin => &self.sin_op,
            UnaryOp::Cos => &self.cos_op,
            UnaryOp::Log => &self.log_op,
            UnaryOp::LeakyRelu => &self.leaky_relu_op,
            UnaryOp::Clamp => &self.clamp_op,
            UnaryOp::Scale => &self.scale_op,
            UnaryOp::AddScalar => &self.add_scalar_op,
        };

        let args = UnaryArgs {
            shape_src,
            src: src.buffer(),
            shape_dst: Some(shape_dest),
            dst: Some(dest.buffer()),
            args: args.map(|a| a.buffer()),
        };
        pipeline.launch_capped(backend, pass, &args, dest.len() as u32)
    }

    pub fn run_cpu<S: StorageMut<f32, Dyn>>(
        &self,
        op: UnaryOp,
        vals: &mut Vector<f32, Dyn, S>,
        args: Vector4<f32>,
    ) {
        vals.apply(|x| *x = op.eval(*x, args));
    }
}

#[cfg(test)]
mod test {
    use crate::ops::UnaryOp;
    use nalgebra::{DVector, Vector4};
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
    async fn gpu_unary_ops_cuda() {
        let mut backend = Cuda::new().unwrap();
        backend.cublas_enabled = false;
        gpu_unary_ops_generic(backend).await;
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_unary_ops_webgpu() {
        let backend = WebGpu::new(Features::default(), Limits::default())
            .await
            .unwrap();
        gpu_unary_ops_generic(backend).await;
    }

    async fn gpu_unary_ops_generic(backend: impl Backend) {
        let mut compiler = SlangCompiler::new(vec![]);
        crate::register_shaders(&mut compiler);

        let unop = super::Unary::from_backend(&backend, &compiler).unwrap();

        let ops = [
            UnaryOp::Abs,
            UnaryOp::Sgn,
            UnaryOp::Neg,
            UnaryOp::Step,
            UnaryOp::Elu,
            UnaryOp::Gelu,
            UnaryOp::GeluQuick,
            UnaryOp::Silu,
            UnaryOp::Tanh,
            UnaryOp::Relu,
            UnaryOp::Sigmoid,
            UnaryOp::HardSigmoid,
            // UnaryOp::HardSwish,
            UnaryOp::Sqr,
            UnaryOp::Sqrt,
            UnaryOp::Sin,
            UnaryOp::Cos,
            UnaryOp::Log,
            UnaryOp::LeakyRelu,
            UnaryOp::Clamp,
            UnaryOp::Scale,
            UnaryOp::AddScalar,
        ];
        let mut shapes = ViewShapeBuffers::new(&backend);

        for op in ops {
            println!("Checking {:?}", op);

            const LEN: u32 = 1757;

            let src = DVector::new_random(LEN as usize);
            let dst = DVector::zeros(LEN as usize);
            let mut dst_read = DVector::zeros(LEN as usize);
            let mut args = Vector4::new_random();
            if args[1] < args[0] {
                args.swap_rows(0, 1); // Ensure min <= max for clamp.
            }
            let gpu_args = op
                .has_args()
                .then(|| GpuTensor::scalar(&backend, args, BufferUsages::UNIFORM).unwrap());
            let gpu_src = GpuTensor::vector(&backend, &src, BufferUsages::STORAGE).unwrap();
            let gpu_dst = GpuTensor::vector(
                &backend,
                &dst,
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            )
            .unwrap();

            let mut encoder = backend.begin_encoding();
            let mut pass = encoder.begin_pass();
            unop.launch(
                &backend,
                &mut shapes,
                &mut pass,
                op,
                &gpu_dst,
                &gpu_src,
                gpu_args.as_ref(),
            )
            .unwrap();
            drop(pass);

            backend.submit(encoder).unwrap();
            backend.synchronize().unwrap();

            backend
                .slow_read_buffer(gpu_dst.buffer(), dst_read.as_mut_slice())
                .await
                .unwrap();

            let mut cpu_result = src;
            unop.run_cpu(op, &mut cpu_result, args);

            approx::assert_relative_eq!(dst_read, cpu_result, epsilon = 1.0e-5);
        }
    }
}
