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
    StructuredBuffer<uint> m, // StructuredBuffer<BlockQ6Kx2>
    StructuredBuffer<float4> v,
) {
    let j_ref = shape_v.iv(0u);
    let m_ref = shape_m.im(0u, 0u);
    var sum = float4(0.0);

    for (var j = 0u; j < shape_m.ncols; j++) {
        let quant0 = shape_m.im(workgroup_id.x * 4u + 0, j);
        let quant1 = shape_m.im(workgroup_id.x * 4u + 1, j);
        let quant2 = shape_m.im(workgroup_id.x * 4u + 2, j);
        let quant3 = shape_m.im(workgroup_id.x * 4u + 3, j);

        let parts0 = dequantize_q6_kx2_workgroup(m, quant0, local_id.x);
        let parts1 = dequantize_q6_kx2_workgroup(m, quant1, local_id.x);
        let parts2 = dequantize_q6_kx2_workgroup(m, quant2, local_id.x);
        let parts3 = dequantize_q6_kx2_workgroup(m, quant3, local_id.x);

        let j_base = j_ref + j * 128u;
        let jj = (local_id.x / 16u) * 64u + ((local_id.x / 8u) % 2u) * 32u + (local_id.x % 8u);
        let vj_a = v[j_base + jj];
        let vj_b = v[j_base + jj + 8u];
        let vj_c = v[j_base + jj + 16u];
        let vj_d = v[j_base + jj + 24u];

        let mat_a = float4x4(parts0[0], parts1[0], parts2[0], parts3[0]);
        sum += mul(mat_a, vj_a);
        let mat_b = float4x4(parts0[1], parts1[1], parts2[1], parts3[1]);
        sum += mul(mat_b, vj_b);
        let mat_c = float4x4(parts0[2], parts1[2], parts2[2], parts3[2]);
        sum += mul(mat_c, vj_c);
        let mat_d = float4x4(parts0[3], parts1[3], parts2[3], parts3[3]);
        sum += mul(mat_d, vj_d);
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



struct BlockQ6Kx2 {
    uint data[105];
}

// TODO: consider using __ref for the input block once it's stable?
//       See https://github.com/shader-slang/slang/issues/5941#issuecomment-2564807874
func dequantize_q6_kx2_workgroup(StructuredBuffer<uint> m, block_id: uint, k: uint) -> float4[4] {
    const uint4 _0xF = uint4(0xFu);
    const uint4 _6 = uint4(6u);
    const uint4 _4 = uint4(4u);
    const uint4 _3 = uint4(3u);
    const uint4 _2 = uint4(2u);
    const int4 _32 = int4(32);

    let data_id = block_id * sizeof(BlockQ6Kx2) / 4;

    if (k / 16u == 0u) {
        // Block A
        // Its data goes from data[0] to half of data[52]
        // It is mostly well aligned with the original BlockQ6K except for the original value.
        let d_a = unpackHalf2x16ToHalf(m[data_id + 52]).x;

        const uint QL0 = 0u;
        const uint QH0 = 32u;
        const uint SC0 = 48u;

        let i = (k / 8u) % 2u;
        let data0 = float4(unpackInt4x8ToInt32(m[data_id + SC0 + i * 2u])) * d_a;
        let data1 = float4(unpackInt4x8ToInt32(m[data_id + SC0 + i * 2u + 1])) * d_a;

        let l = k % 8u;
        let is = l / 4u; // NOTE: is is either 0 or 1

        let is_shift = is * 8u;
        let is_shift2 = (is + 2u) * 8u;
        let qh = unpackUint4x8ToUint32(m[data_id + l + QH0 + i * 8u]);
        let ql0 = unpackUint4x8ToUint32(m[data_id + l + QL0 + i * 16u]);
        let ql32 = unpackUint4x8ToUint32(m[data_id + l + QL0 + i * 16u + 8u]);

        let q1 = int4((ql0 & _0xF) | ((qh & _3) << _4)) - _32;
        let q2 = int4((ql32 & _0xF) | (((qh >> _2) & _3) << _4)) - _32;
        let q3 = int4((ql0 >> _4) | (((qh >> _4) & _3) << _4)) - _32;
        let q4 = int4((ql32 >> _4) | (((qh >> _6) & _3) << _4)) - _32;

        return {
            data0[is] * float4(q1),
            data0[is + 2] * float4(q2),
            data1[is] * float4(q3),
            data1[is + 2] * float4(q4),
        };
    } else {
        // Block B
        // Its data goes from half of data[52] to data[104].
        // All values are starting with the u16 leftmost bits of the previous index.
        // So this is a copy-past of BlockB butwith the `-1u` and `>> 16`, `<< 16`
        // schenanigans to reconstruct the properly aligned uint values.
        let d_b = unpackHalf2x16ToHalf(m[data_id + 104]).y;

        const uint QL0 = 53u + 0u;
        const uint QH0 = 53u + 32u;
        const uint SC0 = 53u + 48u;

        let i = (k / 8u) % 2u;
        let l = k % 8u;
        let isc0 = SC0 + i * 2u;
        let isc1 = SC0 + i * 2u + 1;
        let data0 = float4(unpackInt4x8ToInt32((m[data_id + isc0 - 1u] >> 16) | (m[data_id + isc0] << 16))) * d_b;
        let data1 = float4(unpackInt4x8ToInt32((m[data_id + isc1 - 1u] >> 16) | (m[data_id + isc1] << 16))) * d_b;

        // NOTE: the `* 4u` in these consts is needed because we divide by 4 further down.
        let is = l / 4u; // NOTE: is is either 0 or 1

        let is_shift = is * 8u;
        let is_shift2 = (is + 2u) * 8u;

        let iqh = l + QH0 + i * 8u;
        let iql0 = l + QL0 + i * 16u;
        let iql32 = l + QL0 + i * 16u + 8u;

        let qh = unpackUint4x8ToUint32((m[data_id + iqh - 1u] >> 16) | (m[data_id + iqh] << 16));
        let ql0 = unpackUint4x8ToUint32((m[data_id + iql0 - 1u] >> 16) | (m[data_id + iql0] << 16));
        let ql32 = unpackUint4x8ToUint32((m[data_id + iql32 - 1u] >> 16) | (m[data_id + iql32] << 16));

        let q1 = int4((ql0 & _0xF) | ((qh & _3) << _4)) - _32;
        let q2 = int4((ql32 & _0xF) | (((qh >> _2) & _3) << _4)) - _32;
        let q3 = int4((ql0 >> _4) | (((qh >> _4) & _3) << _4)) - _32;
        let q4 = int4((ql32 >> _4) | (((qh >> _6) & _3) << _4)) - _32;

        return {
            data0[is] * float4(q1),
            data0[is + 2] * float4(q2),
            data1[is] * float4(q3),
            data1[is + 2] * float4(q4),
        };
    }
}