use crate::context::LlmContext;
use crate::gguf::Gguf;
use crate::models::llama2::cpu::Llama2Config;
use crate::models::llama2::LlamaModelType;
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, RmsNormConfig, RoPEConfig,
};
use crate::quantized_matrix::GpuQuantMatrix;
use nalgebra::{DMatrix, DVector};
use slang_hal::backend::Backend;
use slang_hal::re_exports::minislang::SlangCompiler;
use slang_hal::Shader;
use stensor::tensor::{GpuMatrix, GpuScalar, GpuTensor, GpuVector};
use wgpu::BufferUsages;

pub struct Llama2State<B: Backend> {
    /// Activation at current time stamp.
    pub x: GpuVector<f32, B>,
    /// Activation at current time stamp, inside a residual branch.
    pub xb: GpuVector<f32, B>,
    // DEBUG: useful for debugging the transformer.
    // pub xb_read: GpuVector<f32, B>,
    /// Additional buffer for convenience.
    xb2: GpuVector<f32, B>,
    /// Buffer for hidden dimension in the Feed-Forward net.
    hb: GpuVector<f32, B>,
    /// Another buffer for hidden dimension in the Feed-Forward net.
    hb2: GpuVector<f32, B>,
    /// Query.
    pub q: GpuVector<f32, B>,
    // DEBUG: useful for debugging the transformer.
    // pub q_read: GpuVector<f32, B>,
    /// Scores/attention values.
    att: GpuMatrix<f32, B>,
    /// Output logits.
    logits: GpuVector<f32, B>,
    logits_readback: GpuVector<f32, B>,
    // KV cache. Each Vec contains `layer` elements.
    key_cache: Vec<GpuMatrix<f32, B>>,
    value_cache: Vec<GpuMatrix<f32, B>>,
    rope_config: GpuScalar<RoPEConfig, B>,
    rms_norm_config: GpuScalar<RmsNormConfig, B>,
    attn_params: GpuScalar<BatchedMultiqueryAttentionParams, B>,
}

impl<B: Backend> Llama2State<B> {
    pub fn new(backend: &B, config: &Llama2Config) -> Result<Self, B::Error> {
        let kv_dim = (config.dim * config.n_kv_heads) / config.n_q_heads;
        const STORAGE: BufferUsages = BufferUsages::STORAGE;
        const UNIFORM: BufferUsages = BufferUsages::UNIFORM;

        let (rope_config, rms_norm_config, attn_params) = config.derived_configs(0);

        unsafe {
            Ok(Self {
                x: GpuTensor::vector_uninit(
                    backend,
                    config.dim as u32,
                    STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                )?,
                xb: GpuTensor::vector_uninit(
                    backend,
                    config.dim as u32,
                    STORAGE | BufferUsages::COPY_SRC,
                )?,
                // DEBUG: useful for debugging the transformer.
                // xb_read: GpuTensor::vector_uninit(
                //     backend,
                //     config.dim as u32,
                //     BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                // )?,
                xb2: GpuTensor::vector_uninit(backend, config.dim as u32, STORAGE)?,
                hb: GpuTensor::vector_uninit(backend, config.hidden_dim as u32, STORAGE)?,
                hb2: GpuTensor::vector_uninit(backend, config.hidden_dim as u32, STORAGE)?,
                q: GpuTensor::vector_uninit(
                    backend,
                    config.dim as u32,
                    STORAGE | BufferUsages::COPY_SRC,
                )?,
                // DEBUG: useful for debugging the transformer.
                // q_read: GpuTensor::vector_uninit(
                //     backend,
                //     config.dim as u32,
                //     BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                // )?,
                // TODO: for these two, the `kv_dim` doesn’t match the dimension in the field’s comment.
                key_cache: (0..config.n_layers)
                    .map(|_| {
                        GpuTensor::matrix_uninit(
                            backend,
                            kv_dim as u32,
                            config.seq_len as u32,
                            STORAGE,
                        )
                    })
                    .collect::<Result<Vec<_>, B::Error>>()?,
                value_cache: (0..config.n_layers)
                    .map(|_| {
                        GpuTensor::matrix_uninit(
                            backend,
                            kv_dim as u32,
                            config.seq_len as u32,
                            STORAGE,
                        )
                    })
                    .collect::<Result<Vec<_>, B::Error>>()?,
                att: GpuTensor::matrix_uninit(
                    backend,
                    config.seq_len as u32,
                    config.n_q_heads as u32,
                    STORAGE,
                )?,
                logits: GpuTensor::vector_uninit(
                    backend,
                    config.vocab_size as u32,
                    STORAGE | BufferUsages::COPY_SRC,
                )?,
                logits_readback: GpuTensor::vector_uninit(
                    backend,
                    config.vocab_size as u32,
                    BufferUsages::MAP_READ | BufferUsages::COPY_DST,
                )?,
                rope_config: GpuTensor::scalar(
                    backend,
                    rope_config,
                    UNIFORM | BufferUsages::COPY_DST,
                )?,
                rms_norm_config: GpuTensor::scalar(
                    backend,
                    rms_norm_config,
                    UNIFORM | BufferUsages::COPY_DST,
                )?,
                attn_params: GpuTensor::scalar(
                    backend,
                    attn_params,
                    UNIFORM | BufferUsages::COPY_DST,
                )?,
            })
        }
    }

    pub fn rope_config(&self) -> &GpuScalar<RoPEConfig, B> {
        &self.rope_config
    }

    pub fn rope_config_mut(&mut self) -> &mut GpuScalar<RoPEConfig, B> {
        &mut self.rope_config
    }

    pub fn rms_norm_config(&self) -> &GpuScalar<RmsNormConfig, B> {
        &self.rms_norm_config
    }

    pub fn rms_norm_config_mut(&mut self) -> &mut GpuScalar<RmsNormConfig, B> {
        &mut self.rms_norm_config
    }

    pub fn attn_params(&self) -> &GpuScalar<BatchedMultiqueryAttentionParams, B> {
        &self.attn_params
    }

    pub fn attn_params_mut(&mut self) -> &mut GpuScalar<BatchedMultiqueryAttentionParams, B> {
        &mut self.attn_params
    }

    pub fn logits(&self) -> &GpuVector<f32, B> {
        &self.logits
    }

    pub fn logits_readback(&self) -> &GpuVector<f32, B> {
        &self.logits_readback
    }

    pub fn logits_and_readback_mut(&mut self) -> (&GpuVector<f32, B>, &mut GpuVector<f32, B>) {
        (&self.logits, &mut self.logits_readback)
    }
}

pub struct Llama2LayerWeights<B: Backend> {
    pub attn_norm: GpuVector<f32, B>,
    pub attn_k: GpuQuantMatrix<B>,
    pub attn_q: GpuQuantMatrix<B>,
    pub attn_v: GpuQuantMatrix<B>,
    pub attn_k_bias: Option<GpuVector<f32, B>>,
    pub attn_q_bias: Option<GpuVector<f32, B>>,
    pub attn_v_bias: Option<GpuVector<f32, B>>,
    pub ffn_down: GpuQuantMatrix<B>,
    pub ffn_gate: GpuQuantMatrix<B>,
    pub ffn_norm: GpuVector<f32, B>,
    pub ffn_up: GpuQuantMatrix<B>,
    pub attn_output: GpuQuantMatrix<B>,
}

pub struct Llama2Weights<B: Backend> {
    pub layers: Vec<Llama2LayerWeights<B>>,
    pub token_embd: GpuMatrix<f32, B>,
    pub output: GpuQuantMatrix<B>,
    pub output_norm: GpuVector<f32, B>,
}

impl<B: Backend> Llama2Weights<B> {
    pub fn from_gguf(backend: &B, config: &Llama2Config, gguf: &Gguf) -> Result<Self, B::Error> {
        let usage = BufferUsages::STORAGE;
        let mut layers = vec![];

        for i_layer in 0..config.n_layers {
            log::info!("Loop {}/{}", i_layer, config.n_layers);
            let attn_q = format!("blk.{}.attn_q.weight", i_layer);
            let attn_k = format!("blk.{}.attn_k.weight", i_layer);
            let attn_v = format!("blk.{}.attn_v.weight", i_layer);
            let attn_q_bias = format!("blk.{}.attn_q.bias", i_layer);
            let attn_k_bias = format!("blk.{}.attn_k.bias", i_layer);
            let attn_v_bias = format!("blk.{}.attn_v.bias", i_layer);
            let attn_output = format!("blk.{}.attn_output.weight", i_layer);
            let ffn_down = format!("blk.{}.ffn_down.weight", i_layer);
            let ffn_gate = format!("blk.{}.ffn_gate.weight", i_layer);
            let ffn_up = format!("blk.{}.ffn_up.weight", i_layer);
            let ffn_norm = format!("blk.{}.ffn_norm.weight", i_layer);
            let attn_norm = format!("blk.{}.attn_norm.weight", i_layer);

            let attn_q = gguf.tensors[&attn_q].to_gpu_matrix(backend)?.unwrap();
            let attn_k = gguf.tensors[&attn_k].to_gpu_matrix(backend)?.unwrap();
            let attn_v = gguf.tensors[&attn_v].to_gpu_matrix(backend)?.unwrap();
            let attn_q_bias = gguf
                .tensors
                .get(&attn_q_bias)
                .map(|t| GpuTensor::vector(backend, t.data().as_f32().unwrap(), usage))
                .transpose()?;
            let attn_k_bias = gguf
                .tensors
                .get(&attn_k_bias)
                .map(|t| GpuTensor::vector(backend, t.data().as_f32().unwrap(), usage))
                .transpose()?;
            let attn_v_bias = gguf
                .tensors
                .get(&attn_v_bias)
                .map(|t| GpuTensor::vector(backend, t.data().as_f32().unwrap(), usage))
                .transpose()?;
            let attn_output = gguf.tensors[&attn_output].to_gpu_matrix(backend)?.unwrap();
            let ffn_down = gguf.tensors[&ffn_down].to_gpu_matrix(backend)?.unwrap();
            let ffn_gate = gguf.tensors[&ffn_gate].to_gpu_matrix(backend)?.unwrap();

            let test_tensor = &gguf.tensors[&ffn_up];
            println!(
                "Read dim: {:?}, computed dim: {}, {}",
                test_tensor.dimensions, config.hidden_dim, config.dim
            );
            let ffn_up = gguf.tensors[&ffn_up].to_gpu_matrix(backend)?.unwrap();

            let ffn_norm = gguf.tensors[&ffn_norm].data().as_f32().unwrap();
            let attn_norm = gguf.tensors[&attn_norm].data().as_f32().unwrap();

            layers.push(Llama2LayerWeights {
                attn_k,
                attn_norm: GpuTensor::vector(backend, attn_norm, usage)?,
                attn_q,
                attn_v,
                attn_k_bias,
                attn_q_bias,
                attn_v_bias,
                ffn_down,
                ffn_gate,
                ffn_norm: GpuTensor::vector(backend, ffn_norm, usage)?,
                ffn_up,
                attn_output,
            });
        }

        log::info!("Loop done");
        let token_embd_name = "token_embd.weight";
        let output = "output.weight";
        let output_norm = "output_norm.weight";

        // TODO: keep the token embeddings in quantized form
        let token_embd = &gguf.tensors[token_embd_name].data().dequantize().unwrap();
        let token_embd = DMatrix::from_column_slice(config.dim, config.vocab_size, token_embd);
        let token_embd = GpuTensor::matrix(backend, &token_embd, usage | BufferUsages::COPY_SRC)?;

        let output = if let Some(v) = gguf.tensors.get(output) {
            v.to_gpu_matrix(backend)?.unwrap()
        } else {
            gguf.tensors[token_embd_name]
                .to_gpu_matrix(backend)?
                .unwrap()
        };
        let output_norm = gguf.tensors[output_norm].data().as_f32().unwrap();
        let output_norm = DVector::from_row_slice(output_norm);
        let output_norm = GpuTensor::vector(backend, &output_norm, usage);

        Ok(Self {
            layers,
            token_embd,
            output,
            output_norm: output_norm?,
        })
    }
}

pub struct Llama2<B: Backend> {
    model_type: LlamaModelType,
    attn: BatchedMultiqueryAttention<B>,
}

impl<B: Backend> Llama2<B> {
    pub fn new(
        backend: &B,
        compiler: &SlangCompiler,
        model_type: LlamaModelType,
    ) -> Result<Self, B::Error> {
        Ok(Self {
            model_type,
            attn: BatchedMultiqueryAttention::from_backend(backend, compiler)?,
        })
    }

    pub fn launch(
        &self,
        ctxt: &mut LlmContext<B>,
        state: &mut Llama2State<B>,
        weights: &Llama2Weights<B>,
        config: &Llama2Config,
        attn_params: &BatchedMultiqueryAttentionParams,
        pos: u32,
    ) -> Result<(), B::Error> {
        for l in 0..config.n_layers {
            let wl = &weights.layers[l];
            ctxt.rms_norm_assign(&state.rms_norm_config, &state.xb, &state.x, &wl.attn_norm)?;

            let k_cache = state.key_cache[l].column(pos);
            let v_cache = state.value_cache[l].column(pos);

            ctxt.matmul_quant_assign(&state.q, &wl.attn_q, &state.xb)?;
            ctxt.matmul_quant_assign(k_cache, &wl.attn_k, &state.xb)?;
            ctxt.matmul_quant_assign(v_cache, &wl.attn_v, &state.xb)?;

            if let Some(q_bias) = &wl.attn_q_bias {
                ctxt.add_assign(&state.q, q_bias)?;
            }
            if let Some(k_bias) = &wl.attn_k_bias {
                ctxt.add_assign(k_cache, k_bias)?;
            }
            if let Some(v_bias) = &wl.attn_v_bias {
                ctxt.add_assign(v_cache, v_bias)?;
            }

            let rope_variant = self.model_type.rope_variant();
            ctxt.rope(rope_variant, &state.rope_config, &state.q, k_cache)?;

            // Start attention.
            self.dispatch_attn(ctxt, state, l, attn_params)?;

            ctxt.matmul_quant_assign(&state.xb2, &wl.attn_output, &state.xb)?;
            // End attention.

            ctxt.add_assign(&state.x, &state.xb2)?;
            ctxt.rms_norm_assign(&state.rms_norm_config, &state.xb, &state.x, &wl.ffn_norm)?;

            // Start ffn_silu
            ctxt.matmul_quant_assign(&state.hb, &wl.ffn_gate, &state.xb)?;
            ctxt.matmul_quant_assign(&state.hb2, &wl.ffn_up, &state.xb)?;
            ctxt.silu(&state.hb, &state.hb2)?;
            ctxt.matmul_quant_assign(&state.xb2, &wl.ffn_down, &state.hb)?;
            // End ffn_silu

            ctxt.add_assign(&state.x, &state.xb2)?;
        }

        ctxt.rms_norm_assign(
            &state.rms_norm_config,
            &state.xb,
            &state.x,
            &weights.output_norm,
        )?;

        ctxt.matmul_quant_assign(&state.logits, &weights.output, &state.xb)?;

        // // PERF: Softwax the logits so we don’t have to do it on the cpu side in the sampler?
        // self.soft_max.launch(backend, shapes, pass, &state.logits);

        Ok(())
    }

    fn dispatch_attn(
        &self,
        ctxt: &mut LlmContext<B>,
        state: &Llama2State<B>,
        layer: usize,
        attn_params: &BatchedMultiqueryAttentionParams,
    ) -> Result<(), B::Error> {
        self.attn.launch(
            ctxt,
            attn_params,
            &state.attn_params,
            &state.q,
            &state.key_cache[layer],
            &state.value_cache[layer],
            &state.att,
            &state.xb,
        )
    }
}
