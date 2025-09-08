use crate::gguf::Gguf;
use crate::models::gpt2::cpu::{Gpt2Model, Gpt2Params};
use crate::ops::{
    BatchedMultiqueryAttention, BatchedMultiqueryAttentionParams, LayerNorm, UnaryInplace, UnaryOp,
};
use naga_oil::compose::ComposerError;
use nalgebra::{DMatrix, DVector};
use stensor::shapes::ViewShapeBuffers;
use stensor::tensor::{GpuMatrix, GpuScalar, GpuVector};
use slang_hal::Shader;
use stensor::linalg::{Gemv, OpAssign, OpAssignVariant};
use wgpu::{BufferUsages, ComputePass, Device};

pub struct Gpt2State {
    memory_q: GpuVector<f32>,
    memory_att: GpuMatrix<f32>,
    layer_input: GpuVector<f32>,
    curr_768: GpuVector<f32>,
    curr_768_b: GpuVector<f32>,
    curr_2304: GpuVector<f32>,
    curr_3072: GpuVector<f32>,
    curr_vocab: GpuVector<f32>,
    logits_readback: GpuVector<f32>,
    attn_params: GpuScalar<BatchedMultiqueryAttentionParams>,
}

impl Gpt2State {
    pub fn new(backend: &Device, config: &Gpt2Params) -> Self {
        const STORAGE: BufferUsages = BufferUsages::STORAGE;
        const UNIFORM: BufferUsages = BufferUsages::UNIFORM;

        Self {
            memory_q: GpuTensor::vector_uninit(backend, config.n_embd as u32, STORAGE),
            memory_att: GpuTensor::matrix_uninit(
                backend,
                config.n_seq as u32,
                config.n_head as u32,
                STORAGE,
            ),
            layer_input: GpuTensor::vector_uninit(backend, config.n_embd as u32, STORAGE),
            curr_768: GpuTensor::vector_uninit(backend, config.n_embd as u32, STORAGE),
            curr_768_b: GpuTensor::vector_uninit(backend, config.n_embd as u32, STORAGE),
            curr_2304: GpuTensor::vector_uninit(backend, config.attn_b as u32, STORAGE),
            curr_3072: GpuTensor::vector_uninit(backend, config.ff_len as u32, STORAGE),
            curr_vocab: GpuTensor::vector_uninit(
                backend,
                config.n_vocab as u32,
                STORAGE | BufferUsages::COPY_SRC,
            ),
            attn_params: GpuTensor::scalar_uninit(backend, UNIFORM | BufferUsages::COPY_DST),
            logits_readback: GpuTensor::vector_uninit(
                backend,
                config.n_vocab as u32,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
        }
    }

    pub fn logits_readback(&self) -> &GpuVector<f32> {
        &self.logits_readback
    }

    pub fn logits(&self) -> &GpuVector<f32> {
        &self.curr_vocab
    }

    pub fn attn_params(&self) -> &GpuScalar<BatchedMultiqueryAttentionParams> {
        &self.attn_params
    }
}

pub struct Gpt2LayerWeights {
    // Normalization.
    ln_1_g: GpuVector<f32>,
    ln_1_b: GpuVector<f32>,
    ln_2_g: GpuVector<f32>,
    ln_2_b: GpuVector<f32>,

    // attention
    c_attn_attn_w: GpuMatrix<f32>,
    c_attn_attn_b: GpuVector<f32>,
    c_attn_proj_w: GpuMatrix<f32>,
    c_attn_proj_b: GpuVector<f32>,

    // KV cache
    key_cache: GpuMatrix<f32>,
    value_cache: GpuMatrix<f32>,

    // mlp
    c_mlp_fc_w: GpuMatrix<f32>,
    c_mlp_fc_b: GpuVector<f32>,
    c_mlp_proj_w: GpuMatrix<f32>,
    c_mlp_proj_b: GpuVector<f32>,
}

pub struct Gpt2Weights {
    // Normalization
    ln_f_g: GpuVector<f32>,
    ln_f_b: GpuVector<f32>,

    wte: GpuMatrix<f32>,     // token embedding
    wpe: GpuMatrix<f32>,     // position embedding
    lm_head: GpuMatrix<f32>, // language model head

    layers: Vec<Gpt2LayerWeights>,
}

impl Gpt2Weights {
    pub fn from_gguf(backend: &Device, params: &Gpt2Params, gguf: &Gguf) -> Self {
        const STORAGE: BufferUsages = BufferUsages::STORAGE;

        let mut layers = vec![];

        for i_layer in 0..params.n_layer {
            let ln_1_g = format!("blk.{}.attn_norm.weight", i_layer);
            let ln_1_b = format!("blk.{}.attn_norm.bias", i_layer);
            let ln_2_g = format!("blk.{}.ffn_norm.weight", i_layer);
            let ln_2_b = format!("blk.{}.ffn_norm.bias", i_layer);
            let c_attn_attn_w = format!("blk.{}.attn_qkv.weight", i_layer);
            let c_attn_attn_b = format!("blk.{}.attn_qkv.bias", i_layer);
            let c_attn_proj_w = format!("blk.{}.attn_output.weight", i_layer);
            let c_attn_proj_b = format!("blk.{}.attn_output.bias", i_layer);

            let c_mlp_fc_w = format!("blk.{}.ffn_up.weight", i_layer);
            let c_mlp_fc_b = format!("blk.{}.ffn_up.bias", i_layer);
            let c_mlp_proj_w = format!("blk.{}.ffn_down.weight", i_layer);
            let c_mlp_proj_b = format!("blk.{}.ffn_down.bias", i_layer);

            let ln_1_g = gguf.tensors[&ln_1_g].data().as_f32().unwrap();
            let ln_1_b = gguf.tensors[&ln_1_b].data().as_f32().unwrap();
            let ln_2_g = gguf.tensors[&ln_2_g].data().as_f32().unwrap();
            let ln_2_b = gguf.tensors[&ln_2_b].data().as_f32().unwrap();
            let c_attn_attn_w = &gguf.tensors[&c_attn_attn_w].data().dequantize().unwrap();
            let c_attn_attn_b = gguf.tensors[&c_attn_attn_b].data().as_f32().unwrap();
            let c_attn_proj_w = &gguf.tensors[&c_attn_proj_w].data().dequantize().unwrap();
            let c_attn_proj_b = gguf.tensors[&c_attn_proj_b].data().as_f32().unwrap();
            let c_mlp_fc_w = &gguf.tensors[&c_mlp_fc_w].data().dequantize().unwrap();
            let c_mlp_fc_b = gguf.tensors[&c_mlp_fc_b].data().as_f32().unwrap();
            let c_mlp_proj_w = &gguf.tensors[&c_mlp_proj_w].data().dequantize().unwrap();
            let c_mlp_proj_b = gguf.tensors[&c_mlp_proj_b].data().as_f32().unwrap();

            let ln_1_g = DVector::from_row_slice(ln_1_g);
            let ln_1_b = DVector::from_row_slice(ln_1_b);
            let ln_2_g = DVector::from_row_slice(ln_2_g);
            let ln_2_b = DVector::from_row_slice(ln_2_b);

            let c_attn_attn_w =
                DMatrix::from_row_slice(params.attn_b, params.n_embd, c_attn_attn_w);
            let c_attn_attn_b = DVector::from_row_slice(c_attn_attn_b);
            let c_attn_proj_w =
                DMatrix::from_row_slice(params.n_embd, params.n_embd, c_attn_proj_w);
            let c_attn_proj_b = DVector::from_row_slice(c_attn_proj_b);
            let c_mlp_fc_w = DMatrix::from_row_slice(params.ff_len, params.n_embd, c_mlp_fc_w);
            let c_mlp_fc_b = DVector::from_row_slice(c_mlp_fc_b);
            let c_mlp_proj_w = DMatrix::from_row_slice(params.n_embd, params.ff_len, c_mlp_proj_w);
            let c_mlp_proj_b = DVector::from_row_slice(c_mlp_proj_b);

            let key_cache = DMatrix::zeros(params.n_embd, params.n_seq);
            let value_cache = DMatrix::zeros(params.n_embd, params.n_seq);

            layers.push(Gpt2LayerWeights {
                ln_1_g: GpuTensor::vector(backend, &ln_1_g, STORAGE),
                ln_1_b: GpuTensor::vector(backend, &ln_1_b, STORAGE),
                ln_2_g: GpuTensor::vector(backend, &ln_2_g, STORAGE),
                ln_2_b: GpuTensor::vector(backend, &ln_2_b, STORAGE),
                c_attn_attn_w: GpuTensor::matrix(backend, &c_attn_attn_w, STORAGE),
                c_attn_attn_b: GpuTensor::vector(backend, &c_attn_attn_b, STORAGE),
                c_attn_proj_w: GpuTensor::matrix(backend, &c_attn_proj_w, STORAGE),
                c_attn_proj_b: GpuTensor::vector(backend, &c_attn_proj_b, STORAGE),
                key_cache: GpuTensor::matrix(backend, &key_cache, STORAGE),
                value_cache: GpuTensor::matrix(backend, &value_cache, STORAGE),
                c_mlp_fc_w: GpuTensor::matrix(backend, &c_mlp_fc_w, STORAGE),
                c_mlp_fc_b: GpuTensor::vector(backend, &c_mlp_fc_b, STORAGE),
                c_mlp_proj_w: GpuTensor::matrix(backend, &c_mlp_proj_w, STORAGE),
                c_mlp_proj_b: GpuTensor::vector(backend, &c_mlp_proj_b, STORAGE),
            });
        }

        let ln_f_g = gguf.tensors["output_norm.weight"].data().as_f32().unwrap();
        let ln_f_b = gguf.tensors["output_norm.bias"].data().as_f32().unwrap();
        let wte = gguf.tensors["token_embd.weight"]
            .data()
            .dequantize()
            .unwrap();
        let wpe = &gguf.tensors["position_embd.weight"]
            .data()
            .dequantize()
            .unwrap();

        let ln_f_g = DVector::from_row_slice(ln_f_g);
        let ln_f_b = DVector::from_row_slice(ln_f_b);
        let wte = DMatrix::from_column_slice(params.n_embd, params.n_vocab, &wte);
        let wpe = DMatrix::from_column_slice(params.n_embd, params.n_seq, wpe);
        // NOTE: GPT2 shares the lm_head tensor with wte.
        let lm_head = wte.transpose();

        Self {
            ln_f_g: GpuTensor::vector(backend, &ln_f_g, STORAGE),
            ln_f_b: GpuTensor::vector(backend, &ln_f_b, STORAGE),
            wte: GpuTensor::matrix(backend, &wte, STORAGE),
            wpe: GpuTensor::matrix(backend, &wpe, STORAGE),
            lm_head: GpuTensor::matrix(backend, &lm_head, STORAGE),
            layers,
        }
    }

    pub fn from_ram(backend: &Device, w: &Gpt2Model) -> Self {
        const STORAGE: BufferUsages = BufferUsages::STORAGE;

        Self {
            ln_f_g: GpuTensor::vector(backend, &w.ln_f_g, STORAGE),
            ln_f_b: GpuTensor::vector(backend, &w.ln_f_b, STORAGE),
            wte: GpuTensor::matrix(backend, &w.wte, STORAGE),
            wpe: GpuTensor::matrix(backend, &w.wpe, STORAGE),
            lm_head: GpuTensor::matrix(backend, &w.lm_head, STORAGE),
            layers: w
                .layers
                .iter()
                .map(|l| Gpt2LayerWeights {
                    ln_1_g: GpuTensor::vector(backend, &l.ln_1_g, STORAGE),
                    ln_1_b: GpuTensor::vector(backend, &l.ln_1_b, STORAGE),
                    ln_2_g: GpuTensor::vector(backend, &l.ln_2_g, STORAGE),
                    ln_2_b: GpuTensor::vector(backend, &l.ln_2_b, STORAGE),
                    c_attn_attn_w: GpuTensor::matrix(backend, &l.c_attn_attn_w, STORAGE),
                    c_attn_attn_b: GpuTensor::vector(backend, &l.c_attn_attn_b, STORAGE),
                    c_attn_proj_w: GpuTensor::matrix(backend, &l.c_attn_proj_w, STORAGE),
                    c_attn_proj_b: GpuTensor::vector(backend, &l.c_attn_proj_b, STORAGE),
                    key_cache: GpuTensor::matrix(backend, &l.key_cache, STORAGE),
                    value_cache: GpuTensor::matrix(backend, &l.value_cache, STORAGE),
                    c_mlp_fc_w: GpuTensor::matrix(backend, &l.c_mlp_fc_w, STORAGE),
                    c_mlp_fc_b: GpuTensor::vector(backend, &l.c_mlp_fc_b, STORAGE),
                    c_mlp_proj_w: GpuTensor::matrix(backend, &l.c_mlp_proj_w, STORAGE),
                    c_mlp_proj_b: GpuTensor::vector(backend, &l.c_mlp_proj_b, STORAGE),
                })
                .collect(),
        }
    }
}

pub struct Gpt2 {
    layernorm: LayerNorm,
    gelu: UnaryInplace,
    matmul: Gemv,
    attn: BatchedMultiqueryAttention,
    op_assign: OpAssign,
}

impl Gpt2 {
    pub fn new(backend: &Device) -> Result<Self, ComposerError> {
        Ok(Self {
            layernorm: LayerNorm::from_backend(backend)?,
            gelu: UnaryInplace::new(backend, UnaryOp::Gelu)?,
            matmul: Gemv::from_backend(backend)?,
            attn: BatchedMultiqueryAttention::from_backend(backend)?,
            op_assign: OpAssign::from_backend(backend)?,
        })
    }

    pub fn launch(
        &self,
        backend: &Device,
        shapes: &ViewShapeBuffers,
        pass: &mut ComputePass,
        state: &Gpt2State,
        weights: &Gpt2Weights,
        config: &Gpt2Params,
        embd: u32,
        pos: u32,
    ) {
        // Positional encoding.
        self.op_assign.launch(
            backend,
            shapes,
            pass,
            OpAssignVariant::Copy,
            &state.layer_input,
            weights.wte.column(embd),
        );
        self.op_assign.launch(
            backend,
            shapes,
            pass,
            OpAssignVariant::Add,
            &state.layer_input,
            weights.wpe.column(pos),
        );

        for layer in &weights.layers {
            // Layer norm.
            {
                self.layernorm
                    .launch(backend, shapes, pass, &state.curr_768, &state.layer_input);

                // cur = ln_1_g*cur + ln_1_b
                self.op_assign
                    .launch(backend, shapes, pass, OpAssignVariant::Mul, &state.curr_768, &layer.ln_1_g);
                self.op_assign
                    .launch(backend, shapes, pass, OpAssignVariant::Add, &state.curr_768, &layer.ln_1_b);
            }

            // attn
            {
                self.matmul.launch(
                    backend,
                    shapes,
                    pass,
                    &state.curr_2304,
                    &layer.c_attn_attn_w,
                    &state.curr_768,
                );
                self.op_assign.launch(
                    backend,
                    shapes,
                    pass,
                    OpAssignVariant::Add,
                    &state.curr_2304,
                    &layer.c_attn_attn_b,
                );
            }

            // self-attention
            {
                let k_cache = layer.key_cache.column(pos);
                let v_cache = layer.value_cache.column(pos);

                self.op_assign.launch(
                    backend,
                    shapes,
                    pass,
                    OpAssignVariant::Copy,
                    &state.memory_q,
                    state.curr_2304.rows(0, config.n_embd as u32),
                );
                self.op_assign.launch(
                    backend,
                    shapes,
                    pass,
                    OpAssignVariant::Copy,
                    k_cache,
                    state
                        .curr_2304
                        .rows(config.n_embd as u32, config.n_embd as u32),
                );
                self.op_assign.launch(
                    backend,
                    shapes,
                    pass,
                    OpAssignVariant::Copy,
                    v_cache,
                    state
                        .curr_2304
                        .rows(2 * config.n_embd as u32, config.n_embd as u32),
                );

                // attention.
                self.attn.dispatch_legacy(
                    backend,
                    pass,
                    config.n_head as u32,
                    &state.attn_params,
                    &state.memory_q,
                    &layer.key_cache,
                    &layer.value_cache,
                    &state.memory_att,
                    &state.curr_768,
                );
            }

            // projection
            // cur = proj_w*cur + proj_b
            {
                self.matmul.launch(
                    backend,
                    shapes,
                    pass,
                    &state.curr_768_b,
                    &layer.c_attn_proj_w,
                    &state.curr_768,
                );
                self.op_assign.launch(
                    backend,
                    shapes,
                    pass,
                    OpAssignVariant::Add,
                    &state.curr_768_b,
                    &layer.c_attn_proj_b,
                );
            }

            // add the input
            self.op_assign
                .launch(backend, shapes, pass, OpAssignVariant::Add, &state.curr_768_b, &state.layer_input);

            // prep input for next layer
            self.op_assign
                .launch(backend, shapes, pass, OpAssignVariant::Copy, &state.layer_input, &state.curr_768_b);

            // feed-forward network
            {
                // norm
                {
                    self.layernorm.launch(
                        backend,
                        shapes,
                        pass,
                        &state.curr_768,
                        &state.curr_768_b,
                    );

                    // cur = ln_2_g*cur + ln_2_b
                    self.op_assign
                        .launch(backend, shapes, pass, OpAssignVariant::Mul, &state.curr_768, &layer.ln_2_g);
                    self.op_assign
                        .launch(backend, shapes, pass, OpAssignVariant::Add, &state.curr_768, &layer.ln_2_b);
                }

                // fully connected
                self.matmul.launch(
                    backend,
                    shapes,
                    pass,
                    &state.curr_3072,
                    &layer.c_mlp_fc_w,
                    &state.curr_768,
                );
                self.op_assign
                    .launch(backend, shapes, pass, OpAssignVariant::Add, &state.curr_3072, &layer.c_mlp_fc_b);

                // GELU activation
                self.gelu
                    .launch(backend, shapes, pass, &state.curr_3072, None);

                // projection
                self.matmul.launch(
                    backend,
                    shapes,
                    pass,
                    &state.curr_768,
                    &layer.c_mlp_proj_w,
                    &state.curr_3072,
                );
                self.op_assign.launch(
                    backend,
                    shapes,
                    pass,
                    OpAssignVariant::Add,
                    &state.curr_768,
                    &layer.c_mlp_proj_b,
                );
            }

            // finalize input for next layer
            self.op_assign
                .launch(backend, shapes, pass, OpAssignVariant::Add, &state.layer_input, &state.curr_768);
        }

        // norm
        {
            self.layernorm
                .launch(backend, shapes, pass, &state.curr_768, &state.layer_input);

            // inpL = ln_f_g*inpL + ln_f_b
            self.op_assign
                .launch(backend, shapes, pass, OpAssignVariant::Mul, &state.curr_768, &weights.ln_f_g);
            self.op_assign
                .launch(backend, shapes, pass, OpAssignVariant::Add, &state.curr_768, &weights.ln_f_b);
        }

        // inpL = WTE * inpL

        self.matmul.launch(
            backend,
            shapes,
            pass,
            &state.curr_vocab,
            &weights.lm_head,
            &state.curr_768,
        );
    }
}
