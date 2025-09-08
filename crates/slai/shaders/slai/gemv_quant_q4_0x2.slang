import stensor.linalg.shape;

//// NOTE: this assumes a rhs size containing this multiple of f32s:
////       64 * 4 = 256
//const WORKGROUP_SIZE: u32 = 32u;
//const COLS_STEP: u32 = 4u;

// NOTE: this assumes a rhs size containing this multiple of f32s:
//       128 * 4 = 512
static const int WORKGROUP_SIZE = 64;
static const uint COLS_STEP = 8u;

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
    StructuredBuffer<uint> m, // StructuredBuffer<BlockQ4_0x2>
    StructuredBuffer<float4> v,
) {
    let j_ref = shape_v.iv(0u);
    var sum = float4(0.0);

    for (var j = 0u; j < shape_m.ncols / COLS_STEP; j++) {
        // PERF: can we take a reference to the elements?
        let quant0 = shape_m.im(workgroup_id.x * 4u + 0, j * COLS_STEP + local_id.x / 8u) * sizeof(BlockQ4_0x2) / 4;
        let quant1 = shape_m.im(workgroup_id.x * 4u + 1, j * COLS_STEP + local_id.x / 8u) * sizeof(BlockQ4_0x2) / 4;
        let quant2 = shape_m.im(workgroup_id.x * 4u + 2, j * COLS_STEP + local_id.x / 8u) * sizeof(BlockQ4_0x2) / 4;
        let quant3 = shape_m.im(workgroup_id.x * 4u + 3, j * COLS_STEP + local_id.x / 8u) * sizeof(BlockQ4_0x2) / 4;

        let j_base = j_ref + j * (16u * COLS_STEP);
        let vj_a = v[j_base + local_id.x + (local_id.x / 4u) * 4u];
        let vj_b = v[j_base + local_id.x + (local_id.x / 4u) * 4u + 4u];

        if ((local_id.x / 4u) % 2u == 0) {
            // Dequantizing block 1
            let lid = local_id.x % 4u;

            let scale0 = unpackHalf2x16ToHalf(m[quant0 + 0]).x;
            let data0 = m[quant0 + lid] >> 16 | m[quant0 + lid + 1u] << 16;
            let parts0 = BlockQ4_0x2.dequantize_part(data0, scale0);
            let scale1 = unpackHalf2x16ToHalf(m[quant1 + 0]).x;
            let data1 = m[quant1 + lid] >> 16 | m[quant1 + lid + 1u] << 16;
            let parts1 = BlockQ4_0x2.dequantize_part(data1, scale1);
            let scale2 = unpackHalf2x16ToHalf(m[quant2 + 0]).x;
            let data2 = m[quant2 + lid] >> 16 | m[quant2 + lid + 1u] << 16;
            let parts2 = BlockQ4_0x2.dequantize_part(data2, scale2);
            let scale3 = unpackHalf2x16ToHalf(m[quant3 + 0]).x;
            let data3 = m[quant3 + lid] >> 16 | m[quant3 + lid + 1u] << 16;
            let parts3 = BlockQ4_0x2.dequantize_part(data3, scale3);

            let mat_a = float4x4(parts0[0], parts1[0], parts2[0], parts3[0]);
            sum += mul(mat_a, vj_a);
            let mat_b = float4x4(parts0[1], parts1[1], parts2[1], parts3[1]);
            sum += mul(mat_b, vj_b);
        } else {
            // Dequantizing block 2
            let lid = local_id.x % 4u;
            let scale0 = unpackHalf2x16ToHalf(m[quant0 + 4]).y;
            let data0 = m[quant0 + lid + 5u];
            let parts0 = BlockQ4_0x2.dequantize_part(data0, scale0);
            let scale1 = unpackHalf2x16ToHalf(m[quant1 + 4]).y;
            let data1 = m[quant1 + lid + 5u];
            let parts1 = BlockQ4_0x2.dequantize_part(data1, scale1);
            let scale2 = unpackHalf2x16ToHalf(m[quant2 + 4]).y;
            let data2 = m[quant2 + lid + 5u];
            let parts2 = BlockQ4_0x2.dequantize_part(data2, scale2);
            let scale3 = unpackHalf2x16ToHalf(m[quant3 + 4]).y;
            let data3 = m[quant3 + lid + 5u];
            let parts3 = BlockQ4_0x2.dequantize_part(data3, scale3);

            let mat_a = float4x4(parts0[0], parts1[0], parts2[0], parts3[0]);
            sum += mul(mat_a, vj_a);
            let mat_b = float4x4(parts0[1], parts1[1], parts2[1], parts3[1]);
            sum += mul(mat_b, vj_b);
        }
    }

    sketch[local_id.x] = sum;

    GroupMemoryBarrierWithGroupSync();

    if (WORKGROUP_SIZE >= 64u) {
        reduce_sum(local_id.x, 32u);
    }
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

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L11-L14
// Contains two `BlockQ4_0(f16, [i8; 16])`.
struct BlockQ4_0x2 {
    uint data[9];

    static func dequantize_part(data: uint, scale: float) -> float4[2] {
        let x0 = int((data >> (0u * 8u)) & 0x0F) - 8;
        let x1 = int((data >> (0u * 8u + 4u)) & 0x0F) - 8;
        let x2 = int((data >> (1u * 8u)) & 0x0F) - 8;
        let x3 = int((data >> (1u * 8u + 4u)) & 0x0F) - 8;
        let x4 = int((data >> (2u * 8u)) & 0x0F) - 8;
        let x5 = int((data >> (2u * 8u + 4u)) & 0x0F) - 8;
        let x6 = int((data >> (3u * 8u)) & 0x0F) - 8;
        let x7 = int((data >> (3u * 8u + 4u)) & 0x0F) - 8;
        return {
            float4(float(x0), float(x2), float(x4), float(x6)) * scale,
            float4(float(x1), float(x3), float(x5), float(x7)) * scale
        };
    }
}
