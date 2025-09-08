use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, ConvTranspose2d,
    ConvTranspose2dConfig, GemvQuant, GetRelPos, Im2Col, Im2ColConfig, LayerNorm, RmsNorm,
    RmsNormConfig, RoPE, RoPEConfig, RoPEVariant, Silu, SoftMax, Unary, UnaryOp, WinPart,
};
use crate::quantized_matrix::GpuQuantMatrix;
use crate::tensor_cache::{CachedTensor, TensorCache, TensorKey};
use bytemuck::Pod;
use nalgebra::Vector4;
use slang_hal::backend::{Backend, DeviceValue, Encoder};
use slang_hal::re_exports::minislang::SlangCompiler;
use slang_hal::{Shader, ShaderArgs};
use stensor::shapes::{MatrixOrdering, ViewShapeBuffers};
use stensor::tensor::{GpuMatrix, GpuScalar, GpuTensorView, TensorBuilder};
use stensor::{BinOpOffsets, Contiguous, MatrixMode, OpAssign, OpAssignVariant, Repeat, N, T};
use wgpu::BufferUsages;
/*
 * TODO:
 * - [ ] layernorm_inplace
 * - [ ] add_rel_pos_assign
 * - [ ] conv_transpose_2d_p0
 */

pub const GGML_0: usize = 1;
pub const GGML_1: usize = 0;
pub const GGML_2: usize = 2;
pub const GGML_3: usize = 3;

pub struct LlmOps<B: Backend> {
    pub rms_norm: RmsNorm<B>,
    pub rope: RoPE<B>,
    pub silu: Silu<B>,
    pub matmul: GemvQuant<B>,
    pub soft_max: SoftMax<B>,
    pub op_assign: OpAssign<B>,
    pub attn: BatchedMultiqueryAttention<B>,
    pub layernorm: LayerNorm<B>,
    pub contiguous: Contiguous<B>,
    pub im2col: Im2Col<B>,
    pub unop: Unary<B>,
    pub repeat: Repeat<B>,
    pub win_part: WinPart<B>,
    pub get_rel_pos: GetRelPos<B>,
    pub conv_transpose2d: ConvTranspose2d<B>,
}

impl<B: Backend> LlmOps<B> {
    pub fn new(backend: &B, compiler: &SlangCompiler) -> Result<Self, B::Error> {
        Ok(Self {
            rms_norm: RmsNorm::from_backend(backend, compiler)?,
            rope: RoPE::from_backend(backend, compiler)?,
            silu: Silu::from_backend(backend, compiler)?,
            matmul: GemvQuant::from_backend(backend, compiler)?,
            soft_max: SoftMax::from_backend(backend, compiler)?,
            op_assign: OpAssign::from_backend(backend, compiler)?,
            attn: BatchedMultiqueryAttention::from_backend(backend, compiler)?,
            layernorm: LayerNorm::from_backend(backend, compiler)?,
            contiguous: Contiguous::from_backend(backend, compiler)?,
            im2col: Im2Col::from_backend(backend, compiler)?,
            unop: Unary::from_backend(backend, compiler)?,
            repeat: Repeat::from_backend(backend, compiler)?,
            win_part: WinPart::from_backend(backend, compiler)?,
            get_rel_pos: GetRelPos::from_backend(backend, compiler)?,
            conv_transpose2d: ConvTranspose2d::from_backend(backend, compiler)?,
        })
    }
}

pub struct LlmContext<'a, B: Backend> {
    pub backend: &'a B,
    pub cache: &'a mut TensorCache<B>,
    pub shapes: &'a mut ViewShapeBuffers<B>,
    pub pass: Option<B::Pass>,
    pub encoder: Option<B::Encoder>,
    pub ops: &'a LlmOps<B>,
}

impl<'a, B: Backend> LlmContext<'a, B> {
    pub fn begin_submission(&mut self) {
        if self.encoder.is_some() {
            self.submit();
        }

        let mut encoder = self.backend.begin_encoding();
        let pass = encoder.begin_pass();
        self.pass = Some(pass);
        self.encoder = Some(encoder);
    }

    pub fn submit(&mut self) {
        drop(self.pass.take());
        if let Some(encoder) = self.encoder.take() {
            self.backend.submit(encoder).unwrap()
        }
    }
}

// TODO: code quality checklist:
//       - Take GpuTensorViewMut for out tensors.
//       - Have all ops return Result.

impl<'a, B: Backend> LlmContext<'a, B> {
    pub fn tensor_uninit<const DIM: usize>(
        &mut self,
        size: [u32; DIM],
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        // TODO add a cache.
        unsafe {
            assert!(DIM <= 4);
            let mut full_size = [1; 4];
            full_size[..DIM].copy_from_slice(&size);
            let usage = BufferUsages::STORAGE;
            let ordering = MatrixOrdering::RowMajor;
            let key = TensorKey::with_type::<f32>(full_size, ordering, usage);

            self.cache.get_or_insert(key, || {
                TensorBuilder::tensor(full_size, usage)
                    .ordering(ordering)
                    .build_uninit::<f32, _>(self.backend)
            })
        }
    }

    pub fn uniform<T: DeviceValue + Pod>(
        &mut self,
        data: T,
    ) -> Result<CachedTensor<T, B>, B::Error> {
        self.tensor_with_usage(
            [1; 4],
            &[data],
            BufferUsages::STORAGE | BufferUsages::UNIFORM,
        )
    }

    pub fn tensor<T: DeviceValue + Pod, const DIM: usize>(
        &mut self,
        size: [u32; DIM],
        data: &[T],
    ) -> Result<CachedTensor<T, B>, B::Error> {
        self.tensor_with_usage(size, data, BufferUsages::STORAGE)
    }

    pub fn tensor_with_usage<T: DeviceValue + Pod, const DIM: usize>(
        &mut self,
        size: [u32; DIM],
        data: &[T],
        usage: BufferUsages,
    ) -> Result<CachedTensor<T, B>, B::Error> {
        // TODO add a cache.
        assert!(
            DIM <= 4,
            "tensors of dimensions higher than 4 are not supported."
        );
        // COPY_DST is required by the cache system for initialized tensors
        // so we can call `write_buffer` when recycling the tensor.
        // Note sure if this actually has any performance impact.
        let usage = usage | BufferUsages::COPY_DST;
        let mut full_size = [1; 4];
        full_size[..DIM].copy_from_slice(&size);
        let ordering = MatrixOrdering::RowMajor; // TODO: make this configurable?
        let key = TensorKey::with_type::<f32>(full_size, ordering, usage);

        if let Some(mut tensor) = self.cache.get(key) {
            self.backend.write_buffer(tensor.buffer_mut(), data)?;
            Ok(tensor)
        } else {
            let tensor = TensorBuilder::tensor(full_size, usage)
                .ordering(ordering)
                .build_init(self.backend, data)?;
            Ok(self.cache.enroll(tensor, usage))
        }
    }

    pub fn clone<'b>(
        &mut self,
        t: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        self.contiguous(t)
    }

    pub fn contiguous_assign<'b>(
        &mut self,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        value: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.contiguous.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            value,
        )
    }

    pub fn contiguous<'b>(
        &mut self,
        value: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let value = value.into();
        let result = self.tensor_uninit(value.shape().size)?;
        self.contiguous_assign(&result, value)?;
        Ok(result)
    }

    pub fn layernorm_assign<'b>(
        &mut self,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        value: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.layernorm.launch_rows(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            value,
        )
    }

    pub fn layernorm<'b>(
        &mut self,
        value: impl Into<GpuTensorView<'b, f32, B>>,
        _eps: f32, // TODO: take this into account somehow
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let value = value.into();
        let out = self.tensor_uninit(value.shape().size)?;
        self.layernorm_assign(&out, value)?;
        Ok(out)
    }

    pub fn rms_norm_assign<'b>(
        &mut self,
        config: &GpuScalar<RmsNormConfig, B>,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        value: impl Into<GpuTensorView<'b, f32, B>>,
        weight: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.rms_norm.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            config,
            out,
            value,
            weight,
        )
    }

    pub fn rms_norm<'b>(
        &mut self,
        config: &GpuScalar<RmsNormConfig, B>,
        value: impl Into<GpuTensorView<'b, f32, B>>,
        weight: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let value = value.into();
        let weight = weight.into();
        let out = self.tensor_uninit(value.shape().size)?;
        self.rms_norm_assign(config, &out, value, weight)?;
        Ok(out)
    }

    /// Runs `matmul` following ggml’s unconventional behavior.
    ///
    /// All inputs are expected to be row-major.
    pub fn matmul_assign_ggml<'b>(
        &mut self,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let out = out.into();
        let a = a.into();
        let b = b.into();
        assert_eq!(out.ordering(), Some(MatrixOrdering::RowMajor));
        assert_eq!(a.ordering(), Some(MatrixOrdering::RowMajor));
        assert_eq!(b.ordering(), Some(MatrixOrdering::RowMajor));

        // NOTE: this will resolve as a GemvTrFast operation on (a, b).
        //       Capitalized variables are column-major representations of the input.
        // out = b * tr(a)
        // OUT = a * tr(b) // Make `out` column-major by transposing the equation.
        // OUT = tr(A) * B // Make `a` and `b` column-major.
        // OUT = gemm_tr_fast(A, B)
        self.matmul_assign(out, b, a, N, T)
    }

    pub fn matmul_ggml<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let b = b.into();
        assert_eq!(a.ordering(), Some(MatrixOrdering::RowMajor));
        assert_eq!(b.ordering(), Some(MatrixOrdering::RowMajor));
        self.matmul(b, a, N, T)
    }

    pub fn matmul_assign<'b>(
        &mut self,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
        a_mode: MatrixMode,
        b_mode: MatrixMode,
    ) -> Result<(), B::Error> {
        self.ops.matmul.gemv_f32.dispatch_generic(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            a,
            b,
            a_mode,
            b_mode,
        )?;
        Ok(())
    }

    pub fn matmul<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
        a_mode: MatrixMode,
        b_mode: MatrixMode,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let b = b.into();

        let [vrows, vcols, vmats, vcubes] = b.shape().size;
        let [mrows, mcols, mmats, mcubes] = a.shape().size;

        // TODO: fix calculation of output shape to account for broadcasting properly.

        let out_shape = match (a_mode, b_mode) {
            (MatrixMode::Normal, MatrixMode::Normal) => {
                [mrows, vcols, vmats.max(mmats), vcubes.max(mcubes)]
            }
            (MatrixMode::Normal, MatrixMode::Transposed) => {
                [mrows, vrows, vmats.max(mmats), vcubes.max(mcubes)]
            }
            (MatrixMode::Transposed, MatrixMode::Normal) => {
                [mcols, vcols, vmats.max(mmats), vcubes.max(mcubes)]
            }
            (MatrixMode::Transposed, MatrixMode::Transposed) => {
                [mcols, vrows, vmats.max(mmats), vcubes.max(mcubes)]
            }
        };

        let out = self.tensor_uninit(out_shape)?;

        self.matmul_assign(&out, a, b, a_mode, b_mode)?;
        Ok(out)
    }

    pub fn matmul_quant_assign<'b>(
        &mut self,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        a: &GpuQuantMatrix<B>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.matmul.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            out,
            a,
            b,
        )
    }

    pub fn matmul_quant<'b>(
        &mut self,
        a: &GpuQuantMatrix<B>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let b = b.into();
        let [vrows, vcols, vmats, vcubes] = b.shape().size;
        let [mrows, mcols, mmats, mcubes] = a.shape().size;
        assert_eq!(mcols, vrows, "matmul dimension mismatch");
        assert_eq!(mmats, 1, "not supported");
        assert_ne!(mcubes, 1, "not supported");
        let out_shape = [mrows, vcols, vmats, vcubes];
        let out = self.tensor_uninit(out_shape)?;

        self.matmul_quant_assign(&out, a, b)?;
        Ok(out)
    }

    /// Returns a tensor with the shape of `b` and a content equal to `a` repeated as many times
    /// as necessary to fill it.
    ///
    /// Panics if the shape of `b` isn’t an integer multiple of the shape of `a`.
    pub fn repeat<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        // TODO: seems like ggml also keeps the same stride as `b`.
        let a = a.into();
        let b = b.into();
        for k in 0..4 {
            assert_eq!(b.size(k) % a.size(k), 0);
        }
        let out = self.tensor_uninit(b.shape().size)?;
        self.ops.repeat.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &out,
            a,
        )?;
        Ok(out)
    }

    pub fn add<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let b = b.into();
        let out = self.tensor_uninit(a.shape().size)?;
        self.copy(a, &out)?;
        self.add_assign(&out, b)?;
        Ok(out)
    }

    pub fn add_assign<'b>(
        &mut self,
        in_out_a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Add,
            in_out_a,
            b,
        )
    }

    pub fn mul<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let b = b.into();
        let out = self.tensor_uninit(a.shape().size)?;
        self.copy(a, &out)?;
        self.mul_assign(&out, b)?;
        Ok(out)
    }

    pub fn mul_assign<'b>(
        &mut self,
        in_out_a: impl Into<GpuTensorView<'b, f32, B>>,
        b: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Mul,
            in_out_a,
            b,
        )
    }

    pub fn rope<'b>(
        &mut self,
        variant: RoPEVariant,
        config: &GpuScalar<RoPEConfig, B>,
        in_out_q: impl Into<GpuTensorView<'b, f32, B>>,
        in_out_k: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.rope.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            variant,
            config,
            in_out_q,
            in_out_k,
        )
    }

    pub fn silu<'b>(
        &mut self,
        in_out_h1: impl Into<GpuTensorView<'b, f32, B>>,
        in_h2: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.silu.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_h1,
            in_h2,
        )
    }

    pub fn cos_inplace<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.unop_inplace(UnaryOp::Cos, x)
    }

    pub fn cos<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        self.unop(UnaryOp::Cos, x)
    }

    pub fn sin_inplace<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.unop_inplace(UnaryOp::Sin, x)
    }

    pub fn sin<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        self.unop(UnaryOp::Sin, x)
    }

    pub fn unop_inplace<'b>(
        &mut self,
        op: UnaryOp,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.unop.launch_inplace(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            op,
            x,
            None,
        )
    }

    pub fn unop<'b>(
        &mut self,
        op: UnaryOp,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let x = x.into();
        let out = self.tensor_uninit(x.shape().size)?;
        self.ops.unop.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            op,
            &out,
            x,
            None,
        )?;
        Ok(out)
    }

    pub fn gelu_inplace<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.unop_inplace(UnaryOp::Gelu, x)
    }

    pub fn gelu<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        self.unop(UnaryOp::Gelu, x)
    }

    pub fn relu_inplace<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.unop_inplace(UnaryOp::Relu, x)
    }

    pub fn relu<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        self.unop(UnaryOp::Relu, x)
    }

    pub fn scale_assign<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
        scale: f32,
    ) -> Result<(), B::Error> {
        let args = self.uniform(Vector4::ith(0, scale))?;
        self.ops.unop.launch_inplace(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            UnaryOp::Scale,
            x.into(),
            Some(&args),
        )
    }

    pub fn scale<'b>(
        &mut self,
        x: impl Into<GpuTensorView<'b, f32, B>>,
        scale: f32,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let args = self.uniform(Vector4::ith(0, scale))?;
        let x = x.into();
        let out = self.tensor_uninit(x.shape().size)?;
        self.ops.unop.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            UnaryOp::Scale,
            &out,
            x,
            Some(&args),
        )?;
        Ok(out)
    }

    /// Like [`Self::copy`] but with a custom buffer offset.
    ///
    /// This is to work around cases where the desired offset isn’t
    /// aligned to the hardware requirements (e.g. when targetting WebGpu).
    // TODO: find a more systematic way of dealing with this issue (this can happen
    //       for any operation, not just copying).
    pub fn copy_with_offsets<'b>(
        &mut self,
        in_: impl Into<GpuTensorView<'b, f32, B>>,
        in_offset: u32,
        out: impl Into<GpuTensorView<'b, f32, B>>,
        out_offset: u32,
    ) -> Result<(), B::Error> {
        let offsets = BinOpOffsets {
            a: out_offset,
            b: in_offset,
            padding: [0; 2],
        };
        let offsets = self.uniform(offsets)?;
        let out = out.into();
        let in_ = in_.into();
        self.ops.op_assign.launch_copy_with_offsets(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &offsets,
            out,
            in_,
        )
    }

    pub fn copy<'b>(
        &mut self,
        in_: impl Into<GpuTensorView<'b, f32, B>>,
        out: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.op_assign.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            OpAssignVariant::Copy,
            out,
            in_,
        )
    }

    pub fn win_part<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        window_size: u32,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let w = window_size;
        assert_eq!(a.size(3), 1);

        let px = (w - a.size_ggml(1) % w) % w;
        let py = (w - a.size_ggml(2) % w) % w;

        let npx = (px + a.size_ggml(1)) / w;
        let npy = (py + a.size_ggml(2)) / w;
        let np = npx * npy;

        let res_size = [w, a.size_ggml(0), w, np];

        let result = self.tensor_uninit(res_size)?;

        self.ops.win_part.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &result,
            a,
        )?;
        Ok(result)
    }

    pub fn win_unpart<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        w0: u32,
        h0: u32,
        window_size: u32,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let w = self.uniform(window_size)?;
        let result = self.tensor_uninit([w0, a.size_ggml(0), h0, 1])?;
        self.ops.win_part.launch_unpart(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &w,
            &result,
            a,
        )?;
        Ok(result)
    }

    pub fn get_rel_pos<'b>(
        &mut self,
        a: impl Into<GpuTensorView<'b, f32, B>>,
        qh: u32,
        kh: u32,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let a = a.into();
        let result = self.tensor_uninit([kh, a.size_ggml(0), qh, 1])?;
        self.ops.get_rel_pos.launch(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            &result,
            a,
        )?;
        Ok(result)
    }

    pub fn add_rel_pos_assign<'b>(
        &mut self,
        dst: impl Into<GpuTensorView<'b, f32, B>>,
        src1: impl Into<GpuTensorView<'b, f32, B>>,
        src2: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.get_rel_pos.launch_add_rel_pos(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            dst,
            src1,
            src2,
        )
    }

    pub fn attn_mask(
        &mut self,
        params: &BatchedMultiqueryAttentionParams,
        params_gpu: &GpuScalar<BatchedMultiqueryAttentionParams, B>,
        attn: &GpuMatrix<f32, B>,
    ) -> Result<(), B::Error> {
        #[derive(ShaderArgs)]
        struct MultMaskArgs<'a, B: Backend> {
            params: &'a B::Buffer<BatchedMultiqueryAttentionParams>,
            attn: &'a B::Buffer<f32>,
        }

        let rounded_pos = (params.pos + 1).div_ceil(4) * 4;
        let args = MultMaskArgs {
            params: params_gpu.buffer(),
            attn: attn.buffer(),
        };

        self.ops.attn.mult_mask_attn.launch(
            self.backend,
            self.pass.as_mut().unwrap(),
            &args,
            [params.n_heads * rounded_pos, 1, 1],
        )?;
        Ok(())
    }

    pub fn softmax_cols<'b>(
        &mut self,
        in_out_mat: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        self.ops.soft_max.launch_cols(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_mat,
        )
    }

    pub fn softmax_rows<'b>(
        &mut self,
        in_out_mat: impl Into<GpuTensorView<'b, f32, B>>,
    ) -> Result<(), B::Error> {
        let in_out_mat = in_out_mat.into();
        let in_out_mat = in_out_mat.transposed();
        self.ops.soft_max.launch_cols(
            self.backend,
            self.shapes,
            self.pass.as_mut().unwrap(),
            in_out_mat,
        )
    }

    pub fn im2col_assign<'b>(
        &mut self,
        params: &mut GpuScalar<Im2ColConfig, B>,
        result: impl Into<GpuTensorView<'b, f32, B>>,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
        s0: u32,                                      // stride dimension 0
        s1: u32,                                      // stride dimension 1
        p0: u32,                                      // padding dimension 0
        p1: u32,                                      // padding dimension 1
        d0: u32,                                      // dilation dimension 0
        d1: u32,                                      // dilation dimension 1
        is_2d: bool, // indicates if this is a 2D convolution instead of a 1D convolution.
    ) -> Result<(), B::Error> {
        self.ops.im2col.launch(
            self.backend,
            self.pass.as_mut().unwrap(),
            params,
            result,
            kernel,
            input,
            s0,
            s1,
            p0,
            p1,
            d0,
            d1,
            is_2d,
        )
    }

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    // kernel: [OC，IC, KH, KW]
    // input: [N, IC, IH, IW]
    // result: [N, OH, OW, IC*KH*KW]
    pub fn im2col<'b>(
        &mut self,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
        s0: u32,                                      // stride dimension 0
        s1: u32,                                      // stride dimension 1
        p0: u32,                                      // padding dimension 0
        p1: u32,                                      // padding dimension 1
        d0: u32,                                      // dilation dimension 0
        d1: u32,                                      // dilation dimension 1
        is_2d: bool, // indicates if this is a 2D convolution instead of a 1D convolution.
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let kernel = kernel.into();
        let input = input.into();
        let ksz = kernel.shape().size;
        let isz = input.shape().size;

        if is_2d {
            assert_eq!(ksz[GGML_2], isz[GGML_2]);
        } else {
            assert_eq!(ksz[GGML_1], isz[GGML_1]);
            assert_eq!(isz[GGML_3], 1);
        }

        fn conv_output_size(ins: u32, ks: u32, s: u32, p: u32, d: u32) -> u32 {
            (ins + 2 * p - d * (ks - 1) - 1) / s + 1
        }

        let oh = if is_2d {
            conv_output_size(isz[GGML_1], ksz[GGML_1], s1, p1, d1)
        } else {
            0
        };
        let ow = conv_output_size(isz[GGML_0], ksz[GGML_0], s0, p0, d0);

        assert!(is_2d || oh > 0, "input too small compared to kernel");
        assert!(ow > 0, "input too small compared to kernel");

        let out_sz = [
            ow,
            if is_2d {
                ksz[GGML_0] * ksz[GGML_1] * ksz[GGML_2]
            } else {
                ksz[GGML_0] * ksz[GGML_1]
            },
            if is_2d { oh } else { isz[GGML_2] },
            if is_2d { isz[GGML_3] } else { 1 },
        ];

        // NOTE: the `params` content is set in `Im2Col::launch`.
        let params_usage = BufferUsages::STORAGE | BufferUsages::UNIFORM | BufferUsages::COPY_DST;
        let mut params =
            self.tensor_with_usage([1; 4], &[Im2ColConfig::default()], params_usage)?;

        let result = self.tensor_uninit(out_sz)?;
        self.im2col_assign(
            &mut params,
            &result,
            kernel,
            input,
            s0,
            s1,
            p0,
            p1,
            d0,
            d1,
            is_2d,
        )?;
        Ok(result)
    }

    // pub fn conv_1d<'b>(
    //     &mut self,
    //     result: impl Into<GpuTensorView<'b, f32, B>>,
    //     kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
    //     input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    //     s0: u32,                                      // stride dimension
    //     p0: u32,                                      // padding dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), B::Error> {
    //     todo!()
    //     // let result = result.into();
    //     // self.im2col(params, result, kernel, input, s0, 0, p0, 0, d0, 0, false);
    //     //
    //     // let rsize = result.shape().size;
    //     // let ksize = kernel.shape().size;
    //     // let lhs = result.reshape([rsize[0], rsize[1] * rsize[2]]); // [N, OL, IC * K] => [N*OL, IC * K]
    //     // let rhs = kernel.reshape([ksize[0] * ksize[1], ksize[2]]); // [OC，IC, K] => [OC, IC * K]
    //     // self.gemm(gemm_res, lhs, rhs);
    //     // gemm_res.reshape_inplace([rsize[1], ksize[2], rsize[2]]); // [N, OC, OL]
    //     // gemm_res
    // }
    //
    // // conv_1d with padding = half
    // // alias for conv_1d(a, b, s, a->ne[0]/2, d)
    // pub fn conv_1d_ph<'b>(
    //     &mut self,
    //     result: impl Into<GpuTensorView<'b, f32, B>>,
    //     kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
    //     input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    //     s0: u32,                                      // stride dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), B::Error> {
    //     let kernel = kernel.into();
    //     self.conv_1d(result, kernel, input, s0, kernel.shape().size[0] / 2, d0)
    // }
    //
    // // depthwise
    // // TODO: this is very likely wrong for some cases! - needs more testing
    // pub fn conv_1d_dw<'b>(
    //     &mut self,
    //     result: impl Into<GpuTensorView<'b, f32, B>>,
    //     kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
    //     input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    //     s0: u32,                                      // stride dimension
    //     p0: u32,                                      // padding dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), B::Error> {
    //     todo!()
    // }
    //
    // pub fn conv_1d_dw_ph<'b>(
    //     &mut self,
    //     result: impl Into<GpuTensorView<'b, f32, B>>,
    //     kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
    //     input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    //     s0: u32,                                      // stride dimension
    //     d0: u32,                                      // padding dimension
    // ) -> Result<(), B::Error> {
    //     let kernel = kernel.into();
    //     self.conv_1d_dw(result, kernel, input, s0, kernel.shape().size[0] / 2, d0)
    // }
    //
    // pub fn conv_transpose_1d<'b>(
    //     &mut self,
    //     result: impl Into<GpuTensorView<'b, f32, B>>,
    //     kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
    //     input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    //     s0: u32,                                      // stride dimension
    //     p0: u32,                                      // padding dimension
    //     d0: u32,                                      // dilation dimension
    // ) -> Result<(), B::Error> {
    //     todo!()
    // }

    pub fn im2col_sk_p0<'b>(
        &mut self,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let kernel = kernel.into();
        let ksize = kernel.shape().size;
        self.im2col(
            kernel,
            input,
            ksize[GGML_0],
            ksize[GGML_1],
            0,
            0,
            1,
            1,
            true,
        )
    }

    pub fn conv_2d<'b>(
        &mut self,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
        s0: u32,                                      // stride dimension 0
        s1: u32,                                      // stride dimension 1
        p0: u32,                                      // padding dimension 0
        p1: u32,                                      // padding dimension 1
        d0: u32,                                      // dilation dimension 0
        d1: u32,                                      // dilation dimension 1
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let kernel = kernel.into();
        let input = input.into();
        let im2col = self.im2col(kernel, input, s0, s1, p0, p1, d0, d1, true)?; // [N, OH, OW, IC * KH * KW]

        // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
        let reshaped_im2col = im2col.reshape_ggml([
            im2col.size_ggml(0),
            im2col.size_ggml(3) * im2col.size_ggml(2) * im2col.size_ggml(1),
            1,
            1,
        ]);
        // [OC，IC, KH, KW] => [OC, IC * KH * KW]
        let reshaped_kernel = kernel.reshape_ggml([
            kernel.size_ggml(0) * kernel.size_ggml(1) * kernel.size_ggml(2),
            kernel.size_ggml(3),
            1,
            1,
        ]);
        // let reshaped_im2col = self.contiguous(reshaped_im2col)?;
        // let reshaped_kernel = self.contiguous(reshaped_kernel)?;

        let result = self.matmul_ggml(reshaped_im2col, reshaped_kernel)?;

        // reshape => [OC, N, OH, OW]
        // permute => [N, OC, OH, OW]
        self.contiguous(
            result
                .reshape_ggml([
                    im2col.size_ggml(1),
                    im2col.size_ggml(2),
                    im2col.size_ggml(3),
                    kernel.size_ggml(3),
                ])
                .permute_ggml([0, 1, 3, 2]),
        )
    }

    // kernel size is a->ne[0] x a->ne[1]
    // stride is equal to kernel size
    // padding is zero
    // example:
    // a:     16   16    3  768
    // b:   1024 1024    3    1
    // res:   64   64  768    1
    // used in sam
    pub fn conv_2d_sk_p0<'b>(
        &mut self,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let kernel = kernel.into();
        let ksize = kernel.shape().size;
        self.conv_2d(kernel, input, ksize[GGML_0], ksize[GGML_1], 0, 0, 1, 1)
    }

    // kernel size is a->ne[0] x a->ne[1]
    // stride is 1
    // padding is half
    // example:
    // a:      3    3    256  256
    // b:     64   64    256    1
    // res:   64   64    256    1
    // used in sam
    pub fn conv_2d_s1_ph<'b>(
        &mut self,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let kernel = kernel.into();
        let ksize = kernel.shape().size;
        self.conv_2d(
            kernel,
            input,
            1,
            1,
            ksize[GGML_0] / 2,
            ksize[GGML_1] / 2,
            1,
            1,
        )
    }

    // // depthwise
    // pub fn conv_2d_dw<'b>(
    //     &mut self,
    //     kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
    //     input: impl Into<GpuTensorView<'b, f32, B>>,  // data
    //     s0: u32,                                      // stride dimension 0
    //     s1: u32,                                      // stride dimension 1
    //     p0: u32,                                      // padding dimension 0
    //     p1: u32,                                      // padding dimension 1
    //     d0: u32,                                      // dilation dimension 0
    //     d1: u32,                                      // dilation dimension 1
    // ) -> Result<CachedTensor<f32, B>, B::Error> {
    //     todo!()
    // }

    pub fn conv_transpose_2d_p0<'b>(
        &mut self,
        kernel: impl Into<GpuTensorView<'b, f32, B>>, // convolution kernel
        input: impl Into<GpuTensorView<'b, f32, B>>,  // data
        stride: u32,
    ) -> Result<CachedTensor<f32, B>, B::Error> {
        let kernel = kernel.into(); // a
        let input = input.into(); // b
        assert_eq!(kernel.size_ggml(3), input.size_ggml(2));

        fn conv_transpose_output_size(ins: u32, ks: u32, s: u32, p: u32) -> u32 {
            (ins - 1) * s - 2 * p + ks
        }

        let output_sz = [
            conv_transpose_output_size(input.size_ggml(1), kernel.size_ggml(1), stride, 0),
            conv_transpose_output_size(input.size_ggml(0), kernel.size_ggml(0), stride, 0),
            kernel.size_ggml(2),
            input.size_ggml(3),
        ];
        let output = self.tensor_uninit(output_sz)?;
        let wdata = self.tensor_uninit([(input.len() + kernel.len()) as u32, 1, 1, 1])?;
        let params = self.uniform(ConvTranspose2dConfig { stride })?;

        self.ops.conv_transpose2d.launch_ref(
            self.backend,
            self.pass.as_mut().unwrap(),
            self.shapes,
            &params,
            &output,
            kernel,
            input,
            &wdata,
        )?;
        Ok(output)
    }
}
