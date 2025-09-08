use crate::context::LlmContext;
use crate::models::segment_anything::encode_prompt::SamPromptEncoderResult;
use crate::models::segment_anything::layernorm2d::sam_layernorm_2d;
use crate::models::segment_anything::{SamLayerDecTransformerAttn, SamModel, SamState};
use crate::tensor_cache::CachedTensor;
use slang_hal::backend::Backend;
use stensor::tensor::{GpuTensor, GpuTensorView};

pub fn sam_decode_mask<B: Backend>(
    ctxt: &mut LlmContext<B>,
    model: &SamModel<B>,
    prompt: &SamPromptEncoderResult<B>,
    state: &mut SamState<B>,
) -> Result<(), B::Error> {
    let hparams = &model.hparams;
    let dec = &model.dec;
    let n_img_embd = hparams.n_img_embd();

    let tokens;
    {
        // Concatenate output tokens
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L120
        let sparse = &prompt.embd_prompt_sparse;
        tokens = ctxt.tensor_uninit([
            dec.iou_token_w.size_ggml(1) + dec.mask_tokens_w.size_ggml(1) + sparse.size_ggml(1),
            dec.iou_token_w.size_ggml(0),
            sparse.size_ggml(2),
        ])?;
        let offsets = [
            0,
            dec.iou_token_w.size_ggml(1) * tokens.stride_ggml(1),
            dec.iou_token_w.size_ggml(1) * tokens.stride_ggml(1)
                + dec.mask_tokens_w.size_ggml(1) * tokens.stride_ggml(1),
        ];
        ctxt.copy(
            &dec.iou_token_w,
            tokens.view_ggml(
                offsets[0],
                [tokens.size_ggml(0), dec.iou_token_w.size_ggml(1)],
                [Some(1), Some(tokens.stride_ggml(1))],
            ),
        )?;
        ctxt.copy(
            &dec.mask_tokens_w,
            tokens.view_ggml(
                offsets[1],
                [tokens.size_ggml(0), dec.mask_tokens_w.size_ggml(1)],
                [Some(1), Some(tokens.stride_ggml(1))],
            ),
        )?;
        ctxt.copy(
            sparse,
            tokens.view_ggml(
                offsets[2],
                [tokens.size_ggml(0), sparse.size_ggml(1)],
                [Some(1), Some(tokens.stride_ggml(1))],
            ),
        )?;
        // TODO: Sparse prompt embeddings can have more than one point.
    }

    let mut src;
    let mut pos_src;
    let mut src_ne;

    {
        // Expand per-image data in the batch direction to be per-mask
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L125
        src = ctxt.tensor_uninit([
            state.embd_img.size_ggml(1),
            state.embd_img.size_ggml(0),
            state.embd_img.size_ggml(2),
            tokens.size_ggml(2),
        ])?;

        src = {
            let rep = ctxt.repeat(&*state.embd_img, &src)?;
            ctxt.add(&rep, &prompt.embd_prompt_dense)?
        };

        src_ne = src.as_view().shape().size;
        src_ne.swap(0, 1); // Convert to ggml convention.

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        src = ctxt.contiguous(
            src.view_ggml(
                0,
                [
                    src.size_ggml(0) * src.size_ggml(1),
                    src.size_ggml(2),
                    src.size_ggml(3),
                ],
                [Some(1), Some(src.stride_ggml(2)), Some(src.stride_ggml(3))],
            )
            .permute_ggml([1, 0, 2, 3]),
        )?;

        pos_src = ctxt.tensor_uninit([
            state.pe_img.size_ggml(1),
            state.pe_img.size_ggml(0),
            state.pe_img.size_ggml(2),
            tokens.size_ggml(2),
        ])?;
        pos_src = ctxt.repeat(&*state.pe_img, &pos_src)?;

        // flatten & permute
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L83
        pos_src = ctxt.contiguous(
            pos_src
                .view_ggml(
                    0,
                    [
                        pos_src.size_ggml(0) * pos_src.size_ggml(1),
                        pos_src.size_ggml(2),
                        pos_src.size_ggml(3),
                    ],
                    [
                        Some(1),
                        Some(pos_src.stride_ggml(2)),
                        Some(pos_src.stride_ggml(3)),
                    ],
                )
                .permute_ggml([1, 0, 2, 3]),
        )?;
    }

    let mut queries = ctxt.tensor_uninit([0; 4])?; // Will be initialized in the first transformer layer.
    let mut keys = src;

    {
        // Run the transformer
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L62
        for i in 0..model.dec.transformer_layers.len() {
            let tfm_layer = &model.dec.transformer_layers[i];

            // Self attention block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L154
            let skip_first_layer_pe = i == 0;
            if skip_first_layer_pe {
                queries = sam_decode_mask_transformer_attn(
                    ctxt,
                    &tfm_layer.self_attn,
                    &tokens, // queries,
                    &tokens, // queries,
                    &tokens, // queries,
                    model,
                )?;
            } else {
                let q_0 = ctxt.add(&queries, &tokens)?;
                let self_attn = sam_decode_mask_transformer_attn(
                    ctxt,
                    &tfm_layer.self_attn,
                    &q_0,
                    &q_0,
                    &queries,
                    model,
                )?;
                queries = ctxt.add(&queries, &self_attn)?;
            }

            queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
            queries = {
                let w_queries = ctxt.mul(&queries, &tfm_layer.norm1_w)?;
                ctxt.add(&w_queries, &tfm_layer.norm1_b)?
            };

            // Cross attention block, tokens attending to image embedding
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L163
            let q_1 = ctxt.add(&queries, &tokens)?;
            let k_1 = ctxt.add(&keys, &pos_src)?;

            let cross_attn_token_to_img = sam_decode_mask_transformer_attn(
                ctxt,
                &tfm_layer.cross_attn_token_to_img,
                &q_1,
                &k_1,
                &keys,
                model,
            )?;

            ctxt.add_assign(&queries, &cross_attn_token_to_img)?;
            queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
            queries = {
                let w_queries = ctxt.mul(&queries, &tfm_layer.norm2_w)?;
                ctxt.add(&w_queries, &tfm_layer.norm2_b)?
            };
            // MLP block
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L170
            let mut mlp_out = ctxt.matmul_ggml(&tfm_layer.mlp_lin1_w, &queries)?;

            ctxt.add_assign(&mlp_out, &tfm_layer.mlp_lin1_b)?;

            // RELU activation
            ctxt.relu_inplace(&mlp_out)?;
            mlp_out = ctxt.matmul_ggml(&tfm_layer.mlp_lin2_w, &mlp_out)?;
            ctxt.add_assign(&mlp_out, &tfm_layer.mlp_lin2_b)?;

            ctxt.add_assign(&queries, &mlp_out)?;
            queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
            queries = {
                let w_queries = ctxt.mul(&queries, &tfm_layer.norm3_w)?;
                ctxt.add(&w_queries, &tfm_layer.norm3_b)?
            };

            // Cross attention block, image embedding attending to tokens
            // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L175
            let q_2 = ctxt.add(&queries, &tokens)?;
            let k_2 = ctxt.add(&keys, &pos_src)?;
            let cross_attn_img_to_token = sam_decode_mask_transformer_attn(
                ctxt,
                &tfm_layer.cross_attn_img_to_token,
                &k_2,
                &q_2,
                &queries,
                model,
            )?;
            ctxt.add_assign(&keys, &cross_attn_img_to_token)?;
            keys = ctxt.layernorm(&keys, hparams.eps_decoder_transformer)?;
            keys = {
                let w_keys = ctxt.mul(&keys, &tfm_layer.norm4_w)?;
                ctxt.add(&w_keys, &tfm_layer.norm4_b)?
            };
        }

        // Apply the final attention layer from the points to the image
        // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/transformer.py#L99
        let q = ctxt.add(&queries, &tokens)?;
        let k = ctxt.add(&keys, &pos_src)?;
        let final_attn_token_to_img = sam_decode_mask_transformer_attn(
            ctxt,
            &dec.transformer_final_attn_token_to_img,
            &q,
            &k,
            &keys,
            model,
        )?;
        ctxt.add_assign(&queries, &final_attn_token_to_img)?;
        queries = ctxt.layernorm(&queries, hparams.eps_decoder_transformer)?;
        queries = {
            let w_queries = ctxt.mul(&queries, &dec.transformer_norm_final_w)?;
            ctxt.add(&w_queries, &dec.transformer_norm_final_b)?
        };
    }

    let iou_pred = queries.view_ggml(
        0,
        [queries.size_ggml(0), queries.size_ggml(2)],
        [Some(1), Some(queries.stride_ggml(2))],
    );
    let num_mask_tokens = 4; // num_multimask_outputs + 1
    let mask_tokens_out = queries.view_ggml(
        queries.stride_ggml(1),
        [queries.size_ggml(0), num_mask_tokens, queries.size_ggml(2)],
        [
            Some(1),
            Some(queries.stride_ggml(1)),
            Some(num_mask_tokens * queries.stride_ggml(1)),
        ],
    );

    // Upscale mask embeddings and predict masks using the mask tokens
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L136
    keys = ctxt.contiguous(keys.as_view().transposed())?;

    let keys_view = keys.view_ggml(
        0,
        src_ne,
        [
            Some(1),
            Some(src_ne[0] * keys.stride_ggml(0)),
            Some(keys.stride_ggml(1)),
            Some(keys.stride_ggml(2)),
        ],
    );

    let upscaled_embedding;
    {
        // ConvTranspose2d
        // TODO: not 100% sure this runs properly. Some values are not lining up very well
        //       compared to ggml but do look very close.
        keys = ctxt.conv_transpose_2d_p0(&dec.output_upscaling_0_w, keys_view, 2)?;

        {
            let rep = ctxt.repeat(
                dec.output_upscaling_0_b.reshape_ggml([
                    1,
                    1,
                    dec.output_upscaling_0_b.size_ggml(0),
                ]),
                &keys,
            )?;
            ctxt.add_assign(&keys, &rep)?;
        }

        keys = sam_layernorm_2d(
            ctxt,
            &keys,
            n_img_embd,
            &dec.output_upscaling_1_w,
            &dec.output_upscaling_1_b,
            hparams.eps,
        )?;

        // GELU activation
        ctxt.gelu_inplace(&keys)?;

        // ConvTranspose2d
        keys = ctxt.conv_transpose_2d_p0(&dec.output_upscaling_3_w, &keys, 2)?;
        keys = {
            let rep = ctxt.repeat(
                dec.output_upscaling_3_b.reshape_ggml([
                    1,
                    1,
                    dec.output_upscaling_3_b.size_ggml(0),
                    1,
                ]),
                &keys,
            )?;
            ctxt.add(&rep, &keys)?
        };

        // GELU activation
        ctxt.gelu_inplace(&keys)?;
        let upscaled_embedding_ = keys.reshape_ggml([
            keys.size_ggml(0) * keys.size_ggml(1),
            keys.size_ggml(2),
            keys.size_ggml(3),
        ]);
        // TODO: the transpose shouldn’t be needed
        upscaled_embedding = ctxt.contiguous(upscaled_embedding_.transposed())?;
    }

    let hyper_in = ctxt.tensor_uninit([
        num_mask_tokens,
        n_img_embd / 2,
        mask_tokens_out.size_ggml(2),
    ])?;

    for i in 0..num_mask_tokens {
        let mlp = &dec.output_hypernet_mlps[i as usize];
        let in_ = mask_tokens_out.view_ggml(
            i * mask_tokens_out.stride_ggml(1),
            [mask_tokens_out.size_ggml(0), mask_tokens_out.size_ggml(2)],
            [Some(1), Some(mask_tokens_out.stride_ggml(1))],
        );

        let out = sam_decode_mask_mlp_relu_3(
            ctxt, in_, &mlp.w_0, &mlp.b_0, &mlp.w_1, &mlp.b_1, &mlp.w_2, &mlp.b_2,
        )?;

        ctxt.copy_with_offsets(
            &out,
            0,
            hyper_in.view_ggml(
                0,
                [hyper_in.size_ggml(0), hyper_in.size_ggml(2)],
                [Some(1), Some(hyper_in.stride_ggml(1))],
            ),
            i * hyper_in.stride_ggml(1),
        )?;
    }

    let masks = ctxt.matmul_ggml(&upscaled_embedding, &hyper_in)?;
    // let mut masks = ctxt.matmul_ggml(&hyper_in, &upscaled_embedding)?;
    // masks = ctxt.contiguous(masks.as_view().transposed())?; // TODO: shouldn’t be needed.
    let masks = masks.reshape_ggml([
        keys.size_ggml(0),
        keys.size_ggml(1),
        masks.size_ggml(1),
        keys.size_ggml(3),
    ]);

    // Generate mask quality predictions
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L146
    let iou_pred = sam_decode_mask_mlp_relu_3(
        ctxt,
        iou_pred,
        &dec.iou_prediction_head_0_w,
        &dec.iou_prediction_head_0_b,
        &dec.iou_prediction_head_1_w,
        &dec.iou_prediction_head_1_b,
        &dec.iou_prediction_head_2_w,
        &dec.iou_prediction_head_2_b,
    )?;

    // Select the correct mask or masks for output
    // ref: https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/segment_anything/modeling/mask_decoder.py#L101
    ctxt.copy_with_offsets(
        iou_pred.view_ggml(0, [iou_pred.size_ggml(0) - 1, 1], [Some(1), Some(1)]),
        iou_pred.stride_ggml(0),
        &state.iou_predictions,
        0,
    )?;

    let masks = masks.view_ggml(
        masks.stride_ggml(2),
        [
            masks.size_ggml(0),
            masks.size_ggml(1),
            masks.size_ggml(2) - 1,
            masks.size_ggml(3),
        ],
        [
            Some(1),
            Some(masks.stride_ggml(1)),
            Some(masks.stride_ggml(2)),
            Some(masks.stride_ggml(3)),
        ],
    );
    state.low_res_masks = ctxt.contiguous(masks)?.into_inner();

    Ok(())
}

fn sam_decode_mask_transformer_attn<B: Backend>(
    ctxt: &mut LlmContext<B>,
    attn: &SamLayerDecTransformerAttn<B>,
    queries: &GpuTensor<f32, B>,
    keys: &GpuTensor<f32, B>,
    values: &GpuTensor<f32, B>,
    model: &SamModel<B>,
) -> Result<CachedTensor<f32, B>, B::Error> {
    let hparams = &model.hparams;
    let n_heads = hparams.n_dec_heads;
    let q_cur = ctxt.matmul_ggml(&attn.q_w, queries)?;
    ctxt.add_assign(&q_cur, &attn.q_b)?;

    let k_cur = ctxt.matmul_ggml(&attn.k_w, keys)?;
    ctxt.add_assign(&k_cur, &attn.k_b)?;

    let v_cur = ctxt.matmul_ggml(&attn.v_w, values)?;
    ctxt.add_assign(&v_cur, &attn.v_b)?;

    let q = q_cur.reshape_ggml([
        q_cur.size_ggml(0) / n_heads,
        n_heads,
        q_cur.size_ggml(1),
        q_cur.size_ggml(2),
    ]);
    let q = ctxt.contiguous(q.permute_ggml([0, 2, 1, 3]))?;

    let k = k_cur.reshape_ggml([
        k_cur.size_ggml(0) / n_heads,
        n_heads,
        k_cur.size_ggml(1),
        k_cur.size_ggml(2),
    ]);
    let k = ctxt.contiguous(k.permute_ggml([0, 2, 1, 3]))?;

    let v = v_cur.reshape_ggml([
        v_cur.size_ggml(0) / n_heads,
        n_heads,
        v_cur.size_ggml(1),
        v_cur.size_ggml(2),
    ]);
    let v = ctxt.contiguous(v.permute_ggml([0, 2, 1, 3]))?;

    // Q * K
    let kq = ctxt.matmul_ggml(&k, &q)?;
    ctxt.scale_assign(&kq, 1.0 / (q.size_ggml(0) as f32).sqrt())?;
    ctxt.softmax_rows(&kq)?;
    let kqv = {
        let v_tr = ctxt.contiguous(v.as_view().transposed())?;
        ctxt.matmul_ggml(&kq, &v_tr)?
    };
    let mut kqv_merged = ctxt.contiguous(kqv.as_view().transposed())?;
    kqv_merged = ctxt.contiguous(kqv_merged.as_view().permute_ggml([0, 2, 1, 3]))?;
    let kqv_merged = kqv_merged.reshape_ggml([
        kqv_merged.size_ggml(0) * kqv_merged.size_ggml(1),
        kqv_merged.size_ggml(2),
        kqv_merged.size_ggml(3),
    ]);
    let kqv_merged = ctxt.matmul_ggml(&attn.out_w, kqv_merged)?;
    ctxt.add_assign(&kqv_merged, &attn.out_b)?;

    Ok(kqv_merged)
}

fn sam_decode_mask_mlp_relu_3<'b, B: Backend>(
    ctxt: &mut LlmContext<B>,
    in_: impl Into<GpuTensorView<'b, f32, B>>,
    w_0: &GpuTensor<f32, B>,
    b_0: &GpuTensor<f32, B>,
    w_1: &GpuTensor<f32, B>,
    b_1: &GpuTensor<f32, B>,
    w_2: &GpuTensor<f32, B>,
    b_2: &GpuTensor<f32, B>,
) -> Result<CachedTensor<f32, B>, B::Error> {
    let in_ = in_.into();
    let cur = ctxt.matmul_ggml(w_0, in_)?;
    ctxt.add_assign(&cur, b_0)?;
    ctxt.relu_inplace(&cur)?;

    let cur = ctxt.matmul_ggml(w_1, &cur)?;
    ctxt.add_assign(&cur, b_1)?;
    ctxt.relu_inplace(&cur)?;

    let cur = ctxt.matmul_ggml(w_2, &cur)?;
    ctxt.add_assign(&cur, b_2)?;

    Ok(cur)
}
