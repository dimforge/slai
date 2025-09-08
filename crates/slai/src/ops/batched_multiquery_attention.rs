use crate::context::LlmContext;
use crate::models::llama2::cpu::softmax;
use nalgebra::{DMatrix, DVector};
use slang_hal::backend::Backend;
use slang_hal::function::GpuFunction;
use slang_hal::Shader;
use stensor::tensor::{GpuMatrix, GpuScalar, GpuVector};
use stensor::{N, T};

#[derive(Shader)]
#[shader(module = "slai::batched_multiquery_attention")]
/// Shader implementing batched multi-query attention.
pub struct BatchedMultiqueryAttention<B: Backend> {
    pub mult_mask_attn: GpuFunction<B>,
}

#[repr(C)]
#[derive(bytemuck::Pod, bytemuck::Zeroable, Copy, Clone, PartialEq, Eq, Debug)]
/// Parameters needed to run the [`BatchedMultiqueryAttention`] kernel. Matches the layout of the
/// corresponding WGSL struct.
pub struct BatchedMultiqueryAttentionParams {
    pub seq_len: u32,
    pub kv_dim: u32,
    pub kv_mul: u32,
    pub n_heads: u32,
    pub head_size: u32,
    pub pos: u32,
    pub _padding: [u32; 2],
}

impl<B: Backend> BatchedMultiqueryAttention<B> {
    pub fn launch(
        &self,
        ctxt: &mut LlmContext<B>,
        params: &BatchedMultiqueryAttentionParams,
        params_gpu: &GpuScalar<BatchedMultiqueryAttentionParams, B>,
        q: &GpuVector<f32, B>,
        key_cache: &GpuMatrix<f32, B>,
        value_cache: &GpuMatrix<f32, B>,
        attn: &GpuMatrix<f32, B>,
        xb: &GpuVector<f32, B>,
    ) -> Result<(), B::Error> {
        let n_q_heads = params.n_heads;
        let n_kv_heads = n_q_heads / params.kv_mul;
        // Pos rounded to a multiple of 4 to match the matmul element alignment.
        let rounded_pos = (params.pos + 1).div_ceil(4) * 4;
        // [head_size, pos + 1, n_kv_heads] -> [128, ..., 2] -> (transposed for gemv_tr: ) [..., 128, 2]
        let k = key_cache.view(
            0,
            [params.head_size, rounded_pos, n_kv_heads],
            [
                None,
                Some(params.head_size * n_kv_heads),
                Some(params.head_size),
            ],
        );
        // [head_size, kv_mul, n_kv_heads] -> [128, 6, 2]
        let q = q.view(0, [params.head_size, params.kv_mul, n_kv_heads], [None; 3]);
        // [pos + 1, kv_mul, n_kv_heads] -> [..., 6, 2]
        let att = attn.view(0, [rounded_pos, params.kv_mul, n_kv_heads], [None; 3]);
        // [pos + 1, n_q_heads, 1] -> [..., 12, 1]
        let att_softmax = attn
            .view(
                0,
                [params.pos + 1, n_q_heads, 1],
                [None, Some(rounded_pos), None],
            )
            .matrix(0);
        // [head_size, pos + 1, n_kv_heads] -> [128, ..., 2]
        let v = value_cache.view(
            0,
            [params.head_size, rounded_pos, n_kv_heads],
            [
                None,
                Some(params.head_size * n_kv_heads),
                Some(params.head_size),
            ],
        );
        // [head_size, kv_mul, n_kv_heads] -> [128, 6, 2]
        let xb = xb.view(0, [params.head_size, params.kv_mul, n_kv_heads], [None; 3]);

        // gemv.queue_tr(queue, att, k, q);
        // PERF: because we are taking a shapes depending on `pos` we will be
        //       creating a new Buffer for the shape at each forward.
        //       The shape cache should have a mechanism for updating some existing
        //       buffers in-place? Or switch to a LRU cache?

        ctxt.shapes.put_tmp(ctxt.backend, att_softmax.shape())?;
        ctxt.shapes.put_tmp(ctxt.backend, v.shape())?;
        ctxt.shapes
            .put_tmp(ctxt.backend, att.shape().f32_to_vec4())?;
        ctxt.shapes.put_tmp(ctxt.backend, k.shape().f32_to_vec4())?;
        ctxt.shapes.put_tmp(ctxt.backend, q.shape().f32_to_vec4())?;

        ctxt.matmul_assign(att, k, q, T, N)?;
        ctxt.attn_mask(params, params_gpu, attn)?;
        ctxt.softmax_cols(att_softmax)?;
        ctxt.matmul_assign(xb, v, att, N, N)?;
        Ok(())
    }

    pub fn run_cpu(
        params: &BatchedMultiqueryAttentionParams,
        q: &DVector<f32>,
        key_cache: &DMatrix<f32>,
        value_cache: &DMatrix<f32>,
        attn: &mut DMatrix<f32>,
        xb: &mut DVector<f32>,
    ) {
        // The number of embedding vector elements associated to each query head.
        let head_size = params.head_size as usize;
        // The number of query head associated to one key/value head.
        let kv_mul = params.kv_mul as usize;

        // Multihead attention. Iterate over all head.
        // TODO: in llama2.c, each head is iterated on in parallel.
        for h in 0..params.n_heads as usize {
            // Get the query vector for this head.
            let q = q.rows(h * head_size, head_size);
            // Attention scores for this head.
            let mut att = attn.column_mut(h);

            // Iterate over all timesteps (tokens in the sequence), including the current one, but
            // not past the current one due to causality.
            // See the KV cache explanation there: https://youtu.be/Mn_9W1nCFLo?si=3n4GH9f2OzMb5Np0&t=2940
            // -> This is iterating through all the green columns (from K^t) that are the rotated
            //    (by RoPE). The values set in this loop into the `att` variable here (attention
            //    scores) are the elements in the pink row (at the bottom of the QK^t matrix) divide
            //    by sqrt(params.head_size) (in other words, this is what’s given to softmax afterward.
            for t in 0..=params.pos as usize {
                // Get the key vector for this head and at this timestep.
                let k = key_cache.column(t); // TODO: does key_cache have the right dim?
                let k_head = k.rows((h / kv_mul) * head_size, head_size);

                // Calculate the attention score as the dot product of q and k.
                let mut score = q.dot(&k_head);
                score /= (head_size as f32).sqrt();
                // Save the score to the attention buffer.
                att[t] = score;
            }

            // Softmax the scores to get attention weights from 0..=pos inclusively.
            softmax(&mut att.rows_mut(0, params.pos as usize + 1));

            // Weighted sum of the values, store back into xb.
            // /!\ xb is now changing semantic, storing the weighted sums for all the heads.
            //       Now xb contains the "Attention 4" row from https://youtu.be/Mn_9W1nCFLo?si=550ar5aUg1I1k60l&t=2940.
            let mut xb = xb.rows_mut(h * head_size, head_size);
            xb.fill(0.0);
            for t in 0..=params.pos as usize {
                let v = value_cache.column(t);
                let v_head = v.rows((h / kv_mul) * head_size, head_size);
                xb.axpy(att[t], &v_head, 1.0);
            }
        }
    }
}

/*
#[cfg(test)]
mod test {
    use crate::ops::{BatchedMultiqueryAttentionParams, SoftMax};
    use nalgebra::{DMatrix, DVector};
    use slang_hal::gpu::GpuInstance;
    use slang_hal::kernel::CommandEncoderExt;
    use stensor::shapes::ViewShapeBuffers;
    use stensor::tensor::{GpuMatrix, GpuScalar, GpuVector};
    use slang_hal::Shader;
    use stensor::Gemv;
    use wgpu::BufferUsages;

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_backend(gpu.backend()).unwrap();
        let mut encoder = gpu.backend().create_command_encoder(&Default::default());

        // let mut params = BatchedMultiqueryAttentionParams { seq_len: 131072, kv_dim: 256, kv_mul: 6, n_heads: 12, head_size: 128, pos: 9 };
        let params = BatchedMultiqueryAttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 6,
        };

        let q = DVector::new_random((params.n_heads * params.head_size) as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::zeros(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::zeros((params.n_heads * params.head_size) as usize);

        let gpu_params = GpuTensor::scalar(gpu.backend(), params, BufferUsages::UNIFORM);
        let gpu_q = GpuTensor::vector(gpu.backend(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = GpuTensor::matrix(gpu.backend(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = GpuTensor::matrix(gpu.backend(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = GpuTensor::matrix(
            gpu.backend(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = GpuTensor::vector(
            gpu.backend(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = GpuTensor::vector_uninit(
            gpu.backend(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = GpuTensor::matrix_uninit(
            gpu.backend(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut pass = encoder.compute_pass("test", None);
        batched_multihead_attention.launch(
            gpu.backend(),
            &mut pass,
            params.n_heads,
            &gpu_params,
            &gpu_q,
            &gpu_key_cache,
            &gpu_value_cache,
            &gpu_attn,
            &gpu_xb,
        );
        drop(pass);

        gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
        gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

        gpu.queue().submit(Some(encoder.finish()));

        super::BatchedMultiqueryAttention::run_cpu(
            &params,
            &q,
            &key_cache,
            &value_cache,
            &mut attn,
            &mut xb,
        );

        approx::assert_relative_eq!(
            DVector::from(gpu_staging_xb.read(gpu.backend()).await.unwrap()),
            xb,
            epsilon = 1.0e-5
        );

        approx::assert_relative_eq!(
            DMatrix::from_vec(
                attn.nrows(),
                attn.ncols(),
                gpu_staging_attn.read(gpu.backend()).await.unwrap()
            ),
            attn,
            epsilon = 1.0e-5
        );
    }

    #[futures_test::test]
    #[serial_test::serial]
    async fn gpu_attention_multi() {
        let gpu = GpuInstance::new().await.unwrap();
        let batched_multihead_attention =
            super::BatchedMultiqueryAttention::from_backend(gpu.backend()).unwrap();
        let shapes = ViewShapeBuffers::new();
        let matmul = Gemv::from_backend(gpu.backend()).unwrap();
        let softmax = SoftMax::from_backend(gpu.backend()).unwrap();

        // let mut params = BatchedMultiqueryAttentionParams { seq_len: 131072, kv_dim: 256, kv_mul: 6, n_heads: 12, head_size: 128, pos: 0 };
        let mut params = BatchedMultiqueryAttentionParams {
            seq_len: 1024,
            kv_dim: 768,
            kv_mul: 1,
            n_heads: 12,
            head_size: 64,
            pos: 0,
        };

        let q = DVector::new_random((params.n_heads * params.head_size) as usize);
        let key_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let value_cache = DMatrix::new_random(params.kv_dim as usize, params.seq_len as usize);
        let mut attn = DMatrix::zeros(params.seq_len as usize, params.n_heads as usize);
        let mut xb = DVector::zeros((params.n_heads * params.head_size) as usize);

        let gpu_q = GpuTensor::vector(gpu.backend(), q.as_slice(), BufferUsages::STORAGE);
        let gpu_key_cache = GpuTensor::matrix(gpu.backend(), &key_cache, BufferUsages::STORAGE);
        let gpu_value_cache = GpuTensor::matrix(gpu.backend(), &value_cache, BufferUsages::STORAGE);
        let gpu_attn = GpuTensor::matrix(
            gpu.backend(),
            &attn,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );
        let gpu_xb = GpuTensor::vector(
            gpu.backend(),
            xb.as_slice(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        );

        let gpu_staging_xb = GpuTensor::vector_uninit(
            gpu.backend(),
            xb.len() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );
        let gpu_staging_attn = GpuTensor::matrix_uninit(
            gpu.backend(),
            attn.nrows() as u32,
            attn.ncols() as u32,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        for pos in 0..9 {
            let mut encoder = gpu.backend().create_command_encoder(&Default::default());
            params.pos = pos;

            let gpu_params = GpuTensor::scalar(gpu.backend(), params, BufferUsages::UNIFORM);

            let mut pass = encoder.compute_pass("test", None);
            batched_multihead_attention.launch(
                gpu.backend(),
                &shapes,
                gpu.queue(),
                &mut pass,
                &matmul,
                &softmax,
                &params,
                &gpu_params,
                &gpu_q,
                &gpu_key_cache,
                &gpu_value_cache,
                &gpu_attn,
                &gpu_xb,
            );
            drop(pass);

            gpu_staging_xb.copy_from(&mut encoder, &gpu_xb);
            gpu_staging_attn.copy_from(&mut encoder, &gpu_attn);

            gpu.queue().submit(Some(encoder.finish()));

            super::BatchedMultiqueryAttention::run_cpu(
                &params,
                &q,
                &key_cache,
                &value_cache,
                &mut attn,
                &mut xb,
            );

            // NOTE: we can’t compare attn since they don’t have the same layout.
            // approx::assert_relative_eq!(
            //     DMatrix::from_vec(
            //         attn.nrows(),
            //         attn.ncols(),
            //         gpu_staging_attn.read(gpu.backend()).await.unwrap()
            //     ),
            //     attn,
            //     epsilon = 1.0e-5
            // );

            approx::assert_relative_eq!(
                DVector::from(gpu_staging_xb.read(gpu.backend()).await.unwrap()),
                xb,
                epsilon = 1.0e-5
            );
        }
    }
}
*/
