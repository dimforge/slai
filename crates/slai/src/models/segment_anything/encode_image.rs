use crate::context::LlmContext;
use crate::models::segment_anything::layernorm2d::sam_layernorm_2d;
use crate::models::segment_anything::{SamImage, SamModel, SamState};
use slang_hal::backend::Backend;

pub fn sam_encode_image<B: Backend>(
    ctxt: &mut LlmContext<B>,
    model: &SamModel<B>,
    state: &mut SamState<B>,
    img: &SamImage,
) -> Result<(), B::Error> {
    let hparams = &model.hparams;
    let enc = &model.enc_img;

    let n_enc_state = hparams.n_enc_state;
    let n_enc_layer = hparams.n_enc_layer;
    let n_enc_head = hparams.n_enc_head;
    let n_enc_head_dim = hparams.n_enc_head_dim();
    let n_enc_out_chans = hparams.n_enc_out_chans;
    let n_img_size = hparams.n_img_size();
    let n_window_size = hparams.n_window_size();

    let inp = {
        let mut data = vec![0.0; img.pixels.len() * 3];
        let n = img.pixels.len();
        let nx = img.pixels.nrows();
        let ny = img.pixels.ncols();
        assert_eq!(nx as u32, n_img_size);
        assert_eq!(ny as u32, n_img_size);

        for k in 0..3 {
            for y in 0..ny {
                for x in 0..nx {
                    data[k * n + y * nx + x] = img.pixels[y * nx + x][k];
                }
            }
        }

        ctxt.tensor([n_img_size, n_img_size, 3, 1], &data)?
    };

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L392
    let mut cur = ctxt.conv_2d_sk_p0(&enc.proj_w, &inp)?;

    let to_add = ctxt.repeat(&enc.proj_b, &cur)?;

    ctxt.add_assign(&cur, &to_add)?;

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L394
    cur = ctxt.contiguous(cur.as_view().permute_ggml([1, 2, 0, 3]))?;

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L108-L109
    ctxt.add_assign(&cur, &enc.pe)?;

    let mut inp_l = cur;
    for il in 0..n_enc_layer {
        ctxt.begin_submission();

        let layer = &enc.layers[il as usize];

        // norm
        // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L168
        {
            // TODO: mostly matches but the sum is fairly different
            //       from sam.cpp (-0.06961 != -0.0002963)
            cur = ctxt.layernorm(&inp_l, hparams.eps)?;

            // cur = ln_0_w*cur + ln_0_b
            ctxt.mul_assign(&cur, &layer.norm1_w)?;
            ctxt.add_assign(&cur, &layer.norm1_b)?;
        }

        let w0 = cur.size_ggml(1);
        let h0 = cur.size_ggml(2);

        if !hparams.is_global_attn(il) {
            // local attention layer - apply window partition
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L169-L172
            cur = ctxt.win_part(&cur, n_window_size)?;
        }

        let w = cur.size_ggml(1);
        let h = cur.size_ggml(2);

        // self-attention
        {
            cur = ctxt.matmul_ggml(&layer.qkv_w, &cur)?;
            ctxt.add_assign(&cur, &layer.qkv_b)?;

            // split qkv into separate tensors
            // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L225-L229
            let b = cur.size(3);

            cur = ctxt.contiguous(
                cur.reshape_ggml([n_enc_state, 3, w * h, b])
                    .permute_ggml([0, 3, 1, 2]),
            )?;

            let q_view = cur
                .view_ggml(0, [n_enc_state, w * h, b], [None; 3])
                .reshape_ggml([n_enc_head_dim, n_enc_head, w * h, b])
                .permute_ggml([0, 2, 1, 3]);
            let q_cont = ctxt.contiguous(q_view)?;
            let q = q_cont.reshape_ggml([n_enc_head_dim, w * h, b * n_enc_head]);

            let k_view = cur
                .view_ggml(cur.stride(3), [n_enc_state, w * h, b], [None; 3])
                .reshape_ggml([n_enc_head_dim, n_enc_head, w * h, b])
                .permute_ggml([0, 2, 1, 3]);
            let k_cont = ctxt.contiguous(k_view)?;
            let k = k_cont.reshape_ggml([n_enc_head_dim, w * h, b * n_enc_head]);

            let v_view = cur
                .view_ggml(2 * cur.stride(3), [n_enc_state, w * h, b, 1], [None; 4])
                .reshape_ggml([n_enc_head_dim, n_enc_head, w * h, b])
                .permute_ggml([1, 2, 0, 3]);
            let v_cont = ctxt.contiguous(v_view)?;
            let v = v_cont.reshape_ggml([w * h, n_enc_head_dim, b * n_enc_head]);

            let kq = ctxt.matmul_ggml(k, q)?;
            ctxt.scale_assign(&kq, 1.0 / (n_enc_head_dim as f32).sqrt())?;

            let rw = ctxt.get_rel_pos(&layer.rel_pos_w, w, w)?;
            let rh = ctxt.get_rel_pos(&layer.rel_pos_h, h, h)?;
            let q_r = q.reshape_ggml([n_enc_head_dim, w, h, b * n_enc_head]);

            let rhs = ctxt.contiguous(q_r.permute_ggml([0, 2, 1, 3]))?;
            let rel_w = ctxt.matmul_ggml(&rw, &rhs)?;
            let rel_w = ctxt.contiguous(rel_w.permute_ggml([0, 2, 1, 3]))?;
            let rel_h = ctxt.matmul_ggml(&rh, q_r)?;

            ctxt.add_rel_pos_assign(&kq, &rel_w, &rel_h)?;

            let attn = kq;
            ctxt.softmax_rows(&attn)?;

            let kqv = ctxt.matmul_ggml(v, &attn)?;

            cur = ctxt.contiguous(
                kqv.reshape_ggml([n_enc_head_dim, w * h, n_enc_head, b])
                    .permute_ggml([0, 2, 1, 3]),
            )?;
            cur = ctxt.matmul_ggml(&layer.proj_w, cur.reshape_ggml([n_enc_state, w, h, b]))?;
            ctxt.add_assign(&cur, &layer.proj_b)?;
        }

        if !hparams.is_global_attn(il) {
            // local attention layer - reverse window partition
            cur = ctxt.win_unpart(&cur, w0, h0, n_window_size)?;
        }

        ctxt.add_assign(&cur, &inp_l)?;
        let inp_ff = cur;

        // feed-forward network
        // norm
        // TODO: mostly matches but the sum is fairly different
        //       from sam.cpp (-0.078239 != 0.0004176)
        let mut cur = ctxt.layernorm(&inp_ff, hparams.eps)?;

        // cur = mlp_ln_w*cur + mlp_ln_b
        ctxt.mul_assign(&cur, &layer.norm2_w)?;
        ctxt.add_assign(&cur, &layer.norm2_b)?;

        // fully connected
        cur = ctxt.matmul_ggml(&layer.mlp_lin1_w, &cur)?;
        ctxt.add_assign(&cur, &layer.mlp_lin1_b)?;

        // GELU activation
        cur = ctxt.gelu(&cur)?;

        // projection
        cur = ctxt.matmul_ggml(&layer.mlp_lin2_w, &cur)?;
        ctxt.add_assign(&cur, &layer.mlp_lin2_b)?;

        inp_l = ctxt.add(&cur, &inp_ff)?;
    }
    ctxt.begin_submission();

    cur = ctxt.contiguous(inp_l.permute_ggml([2, 0, 1, 3]))?;
    cur = ctxt.conv_2d_sk_p0(&enc.neck_conv_0, &cur)?;

    cur = sam_layernorm_2d(
        ctxt,
        &cur,
        n_enc_out_chans,
        &enc.neck_norm_0_w,
        &enc.neck_norm_0_b,
        hparams.eps,
    )?;

    cur = ctxt.conv_2d_s1_ph(&enc.neck_conv_1, &cur)?;
    cur = sam_layernorm_2d(
        ctxt,
        &cur,
        n_enc_out_chans,
        &enc.neck_norm_1_w,
        &enc.neck_norm_1_b,
        hparams.eps,
    )?;

    state.embd_img = cur.into_inner();

    Ok(())
}
