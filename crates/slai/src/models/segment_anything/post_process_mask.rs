use crate::models::segment_anything::{SamHParams, SamImageU8, SamState};
use slang_hal::backend::Backend;

pub async fn sam_postprocess_masks<B: Backend>(
    backend: &B,
    hparams: &SamHParams,
    nx: usize,
    ny: usize,
    state: &SamState<B>,
    mask_on_val: u8,
    mask_off_val: u8,
) -> anyhow::Result<Vec<SamImageU8>> {
    if state.low_res_masks.size_ggml(2) == 0 {
        return Ok(vec![]);
    }

    assert_eq!(
        state.low_res_masks.size_ggml(2),
        state.iou_predictions.size_ggml(0)
    );

    let n_img_size = hparams.n_img_size();
    let mask_threshold = hparams.mask_threshold;
    let iou_threshold = hparams.iou_threshold;
    let stability_score_threshold = hparams.stability_score_threshold;
    let intersection_threshold = mask_threshold + hparams.stability_score_offset;
    let union_threshold = mask_threshold - hparams.stability_score_offset;

    let ne0 = state.low_res_masks.size_ggml(0) as usize;
    let ne1 = state.low_res_masks.size_ggml(1) as usize;
    let ne2 = state.low_res_masks.size_ggml(2) as usize;

    // Remove padding and upscale masks to the original image size.
    // ref: https://github.com/facebookresearch/segment-anything/blob/efeab7296ab579d4a261e554eca80faf6b33924a/segment_anything/modeling/sam.py#L140

    let preprocess_scale = nx.max(ny) as f32 / n_img_size as f32;
    let cropped_nx = (nx as f32 / preprocess_scale + 0.5) as usize;
    let cropped_ny = (ny as f32 / preprocess_scale + 0.5) as usize;

    let scale_x_1 = ne0 as f32 / n_img_size as f32;
    let scale_y_1 = ne1 as f32 / n_img_size as f32;

    let scale_x_2 = cropped_nx as f32 / nx as f32;
    let scale_y_2 = cropped_ny as f32 / ny as f32;

    let iou_data = backend
        .slow_read_vec(state.iou_predictions.buffer())
        .await?;
    let low_res_masks = backend.slow_read_vec(state.low_res_masks.buffer()).await?;

    let mut res_map = vec![];

    for i in 0..ne2 {
        if iou_threshold > 0.0 && iou_data[i] < iou_threshold {
            println!(
                "Skipping mask {} with iou {} below threshold {}",
                i, iou_data[i], iou_threshold
            );
            continue; // Filtering masks with iou below the threshold
        }

        let mut mask_data = vec![0.0; (n_img_size * n_img_size) as usize];
        {
            let data = &low_res_masks[i * ne0 * ne1..];

            for iy in 0..n_img_size {
                for ix in 0..n_img_size {
                    let sx = (scale_x_1 * (ix as f32 + 0.5) - 0.5).max(0.0);
                    let sy = (scale_y_1 * (iy as f32 + 0.5) - 0.5).max(0.0);

                    let x0 = sx as usize;
                    let y0 = sy as usize;

                    let x1 = (x0 + 1).min(ne0 - 1);
                    let y1 = (y0 + 1).min(ne1 - 1);

                    let dx = sx - x0 as f32;
                    let dy = sy - y0 as f32;

                    let j00 = y0 * ne0 + x0;
                    let j01 = y0 * ne0 + x1;
                    let j10 = y1 * ne0 + x0;
                    let j11 = y1 * ne0 + x1;

                    let v00 = data[j00];
                    let v01 = data[j01];
                    let v10 = data[j10];
                    let v11 = data[j11];

                    let v0 = (1.0 - dx) * v00 + dx * v01;
                    let v1 = (1.0 - dx) * v10 + dx * v11;

                    let v = (1.0 - dy) * v0 + dy * v1;

                    mask_data[(iy * n_img_size + ix) as usize] = v;
                }
            }
        }

        let mut intersections = 0;
        let mut unions = 0;
        let mut res = SamImageU8 {
            nx,
            ny,
            data: vec![mask_off_val; nx * ny],
        };
        let mut min_iy = ny;
        let mut max_iy = 0;
        let mut min_ix = nx;
        let mut max_ix = 0;

        {
            let data = &mut mask_data;

            for iy in 0..ny {
                for ix in 0..nx {
                    let sx = (scale_x_2 * (ix as f32 + 0.5) - 0.5).max(0.0);
                    let sy = (scale_y_2 * (iy as f32 + 0.5) - 0.5).max(0.0);

                    let x0 = sx as usize;
                    let y0 = sy as usize;

                    let x1 = (x0 + 1).min(cropped_nx - 1);
                    let y1 = (y0 + 1).min(cropped_ny - 1);

                    let dx = sx - x0 as f32;
                    let dy = sy - y0 as f32;

                    let j00 = y0 * n_img_size as usize + x0;
                    let j01 = y0 * n_img_size as usize + x1;
                    let j10 = y1 * n_img_size as usize + x0;
                    let j11 = y1 * n_img_size as usize + x1;

                    let v00 = data[j00];
                    let v01 = data[j01];
                    let v10 = data[j10];
                    let v11 = data[j11];

                    let v0 = (1.0 - dx) * v00 + dx * v01;
                    let v1 = (1.0 - dx) * v10 + dx * v11;

                    let v = (1.0 - dy) * v0 + dy * v1;

                    if v > intersection_threshold {
                        intersections += 1;
                    }
                    if v > union_threshold {
                        unions += 1;
                    }
                    if v > mask_threshold {
                        min_iy = min_iy.min(iy);
                        max_iy = max_iy.max(iy);
                        min_ix = min_ix.min(ix);
                        max_ix = max_ix.max(ix);

                        res.data[iy * nx + ix] = mask_on_val;
                    }
                }
            }
        }

        let stability_score = intersections as f32 / unions as f32;
        if stability_score_threshold > 0.0 && stability_score < stability_score_threshold {
            println!(
                "Skipping mask {} with stability score {} below threshold {}\n",
                i, stability_score, stability_score_threshold
            );
            continue; // Filtering masks with stability score below the threshold
        }

        println!(
            "Mask {}: iou = {}, stability_score = {}, bbox ({}, {}), ({}, {})",
            i, iou_data[i], stability_score, min_ix, max_ix, min_iy, max_iy
        );

        res_map.push((iou_data[i] + stability_score, res));
    }

    res_map.sort_by(|a, b| a.0.total_cmp(&b.0));

    Ok(res_map.into_iter().map(|e| e.1).collect())
}
