module gemv_quant_q4_k;

import stensor.linalg.shape;

static const int WORKGROUP_SIZE = 32;

groupshared float4 sketch[WORKGROUP_SIZE];

func reduce_sum(index: uint, stride: uint) {
    if (index < stride) {
        sketch[index] += sketch[index + stride];
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv(
    uint3 workgroup_id: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<uint> m, // StructuredBuffer<BlockQ4K>
    StructuredBuffer<float4> v,
) {
    let j_ref = shape_v.iv(0u);
    let m_ref = shape_m.im(0u, 0u);
    var sum = float4(0.0);

    for (int j = 0u; j < shape_m.ncols; j++) {
        let quant0 = shape_m.im(workgroup_id.x * 4u + 0, j);
        let quant1 = shape_m.im(workgroup_id.x * 4u + 1, j);
        let quant2 = shape_m.im(workgroup_id.x * 4u + 2, j);
        let quant3 = shape_m.im(workgroup_id.x * 4u + 3, j);

        let parts0 = dequantize_q4_k_workgroup(m, quant0, local_id.x);
        let parts1 = dequantize_q4_k_workgroup(m, quant1, local_id.x);
        let parts2 = dequantize_q4_k_workgroup(m, quant2, local_id.x);
        let parts3 = dequantize_q4_k_workgroup(m, quant3, local_id.x);

        let j_base = j_ref + j * 64u;
        let jj = local_id.x & 0xfffffff8u; // == (local_id.x / 8u) * 8u;
        let vj_a = v[j_base + local_id.x + jj];
        let vj_b = v[j_base + local_id.x + jj + 8u];

        let mat_a = float4x4(parts0[0], parts1[0], parts2[0], parts3[0]);
        sum += mul(mat_a, vj_a);
        let mat_b = float4x4(parts0[1], parts1[1], parts2[1], parts3[1]);
        sum += mul(mat_b, vj_b);
    }

    sketch[local_id.x] = sum;

    GroupMemoryBarrierWithGroupSync();

//    reduce_sum(local_id.x, 32u);
    reduce_sum(local_id.x, 16u);
    reduce_sum(local_id.x, 8u);
    reduce_sum(local_id.x, 4u);
    reduce_sum(local_id.x, 2u);
    reduce_sum(local_id.x, 1u);

    if (local_id.x == 0u) {
        let i_out = shape_out.iv(workgroup_id.x);
        out[i_out] = sketch[0];
    }
}

public struct BlockQ4K {
    // FIXME: slang supports types like u8 and u16, just use them instead of our hacks based
    //        on wgsl limitations?
    uint d_dmin;    // (u16, u16) = super-block scale, super-block scale for quantized mins
    uint scales[3]; // [u8; 12] = scales and mins, quantized with 6 bits
    uint qs[32];    // [u8; 256 / 2] = 4-bit quants

    public static func unpack_scale_and_min(j: uint, qj_prev_: uint, qj_: uint, qj_next_: uint) -> uint2 {
        let shift = (j % 4) * 8;
        let qj_prev = (qj_prev_ >> shift) & 0x00ff;
        let qj = (qj_ >> shift) & 0x00ff;
        let qj_next = (qj_next_ >> shift) & 0x00ff;

        if (j < 4) {
            let d = qj & 63;
            let m = qj_next & 63;
            return uint2(d, m);
        } else {
            let d = (qj_next & 0xf) | ((qj_prev >> 6) << 4);
            let m = (qj_next >> 4) | ((qj >> 6) << 4);
            return uint2(d, m);
        }
    }
}

func dequantize_q4_k_workgroup(StructuredBuffer<uint> m, block_id: uint, k: uint) -> float4[2] {
    let d_dmin_id = block_id * sizeof(BlockQ4K) / 4;
    let scales_id = d_dmin_id + 1;
    let qs_id = d_dmin_id + 4;

    let d_dmin = unpackHalf2x16ToHalf(m[d_dmin_id]);
    let d = d_dmin.x;
    let min = d_dmin.y;

    // 32 threads workgroups
    let j = k / 8u;
    let is = j * 2u;
    let qj_prev1 = m[scales_id + max(is / 4, 1u) - 1];
    let qj1 = m[scales_id + is / 4];
    let qj_next1 = m[scales_id + is / 4 + 1];
    let sc_m1 = BlockQ4K.unpack_scale_and_min(is, qj_prev1, qj1, qj_next1);
    let d1 = d * float(sc_m1.x);
    let m1 = min * float(sc_m1.y);

    let qj_prev2 = m[scales_id + max((is + 1) / 4, 1u) - 1];
    let qj2 = m[scales_id + (is + 1) / 4];
    let qj_next2 = m[scales_id + (is + 1) / 4 + 1];
    let sc_m2 = BlockQ4K.unpack_scale_and_min(is + 1, qj_prev2, qj2, qj_next2);
    let d2 = d * float(sc_m2.x);
    let m2 = min * float(sc_m2.y);

    // NOTE: from the ref implementation:
    //    iq     == (k / 8u) * 8u
    // => iq + l == (k / 8u) * 8u + k % 8u == k
    let qs = m[qs_id + k];

    // NOTE: from the ref implementation:
    //    j * 16  == (k / 8u) * 16u
    //    j * 16 + [0..7u] == (k / 8u) * 8u * 2u + k % 8u
    //                     == (k / 8u) * 8u + k % 8u + (k / 8u) * 8u
    //                     == k + (k / 8u) * 8u
    //                     == k + j * 8u
    let res_a = float4(
       float(qs & 0xF),
       float((qs >> 8) & 0xF),
       float((qs >> 16) & 0xF),
       float((qs >> 24) & 0xF),
   ) * d1 - m1;
   let res_b = float4(
       float((qs >> 4) & 0xF),
       float((qs >> 12) & 0xF),
       float((qs >> 20) & 0xF),
       float((qs >> 28) & 0xF),
   ) * d2 - m2;

    return { res_a, res_b };
}