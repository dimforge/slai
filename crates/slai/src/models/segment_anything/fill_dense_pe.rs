use crate::context::LlmContext;
use crate::models::segment_anything::SamModel;
use crate::tensor_cache::CachedTensor;
use slang_hal::backend::Backend;

pub fn sam_fill_dense_pe<B: Backend>(
    ctxt: &mut LlmContext<B>,
    model: &SamModel<B>,
) -> Result<CachedTensor<f32, B>, B::Error> {
    let hparams = &model.hparams;
    let enc = &model.enc_prompt;

    let n_img_embd = hparams.n_img_embd() as usize;
    let n_img_embd_inv = 1.0 / n_img_embd as f32;

    let mut xy_embed_stacked = vec![0.0; 2 * n_img_embd * n_img_embd];
    for i in 0..n_img_embd {
        let row = 2 * i * n_img_embd;
        let y_val = 2.0 * (i as f32 + 0.5) * n_img_embd_inv - 1.0;

        for j in 0..n_img_embd {
            let x_val = 2.0 * (j as f32 + 0.5) * n_img_embd_inv - 1.0;
            xy_embed_stacked[row + 2 * j] = x_val;
            xy_embed_stacked[row + 2 * j + 1] = y_val;
        }
    }

    let xy_embed_stacked =
        ctxt.tensor([n_img_embd as u32, 2, n_img_embd as u32], &xy_embed_stacked)?;

    let pe_transposed = ctxt.contiguous(enc.pe.as_view().transposed())?;
    let mut cur = ctxt.matmul_ggml(&pe_transposed, &xy_embed_stacked)?;
    cur = ctxt.scale(&cur, std::f32::consts::TAU)?;

    // concat
    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py#L192
    {
        let t_sin = ctxt.sin(&cur)?;
        let t_cos = ctxt.cos(&cur)?;

        cur = ctxt.tensor_uninit([
            cur.size_ggml(1),
            t_sin.size_ggml(0) + t_cos.size_ggml(0),
            cur.size_ggml(2),
        ])?;
        ctxt.copy(
            &t_sin,
            cur.view_ggml(
                0,
                [t_sin.size_ggml(0), t_sin.size_ggml(1), t_sin.size_ggml(2)],
                [Some(1), Some(cur.stride_ggml(1)), Some(cur.stride_ggml(2))],
            ),
        )?;
        ctxt.copy(
            &t_cos,
            cur.view_ggml(
                t_sin.stride_ggml(1),
                [t_sin.size_ggml(0), t_sin.size_ggml(1), t_sin.size_ggml(2)],
                [Some(1), Some(cur.stride_ggml(1)), Some(cur.stride_ggml(2))],
            ),
        )?;
    }

    ctxt.contiguous(cur.permute_ggml([2, 0, 1, 3]))
}
