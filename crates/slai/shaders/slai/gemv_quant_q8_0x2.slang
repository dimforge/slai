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
    StructuredBuffer<uint> m, // StructuredBuffer<BlockQ8_0x2> m,
    StructuredBuffer<float4> v,
) {
    let j_ref = shape_v.iv(0u);
    var sum = float4(0.0);

    for (var j = 0u; j < shape_m.ncols / 2u; j++) {
        // PERF: can we take a ref to &m[..] instead of a copy?
        let quant0 = shape_m.im(workgroup_id.x * 4u + 0, j * 2u + local_id.x / 16u) * sizeof(BlockQ8_0x2) / 4;
        let quant1 = shape_m.im(workgroup_id.x * 4u + 1, j * 2u + local_id.x / 16u) * sizeof(BlockQ8_0x2) / 4;
        let quant2 = shape_m.im(workgroup_id.x * 4u + 2, j * 2u + local_id.x / 16u) * sizeof(BlockQ8_0x2) / 4;
        let quant3 = shape_m.im(workgroup_id.x * 4u + 3, j * 2u + local_id.x / 16u) * sizeof(BlockQ8_0x2) / 4;

        let j_base = j_ref + j * 32u;
        let vj = v[j_base + local_id.x];

        if ((local_id.x / 8u) % 2u == 0) {
            // Dequantizing block 1
            let lid = local_id.x % 8u;
            let scale0 = unpackHalf2x16ToHalf(m[quant0 + 0]).x;
            let data0 = unpackInt4x8ToInt32(m[quant0 + lid] >> 16 | m[quant0 + lid + 1u] << 16);
            let scale1 = unpackHalf2x16ToHalf(m[quant1 + 0]).x;
            let data1 = unpackInt4x8ToInt32(m[quant1 + lid] >> 16 | m[quant1 + lid + 1u] << 16);
            let scale2 = unpackHalf2x16ToHalf(m[quant2 + 0]).x;
            let data2 = unpackInt4x8ToInt32(m[quant2 + lid] >> 16 | m[quant2 + lid + 1u] << 16);
            let scale3 = unpackHalf2x16ToHalf(m[quant3 + 0]).x;
            let data3 = unpackInt4x8ToInt32(m[quant3 + lid] >> 16 | m[quant3 + lid + 1u] << 16);

            let mat = float4x4(
                float4(data0) * scale0,
                float4(data1) * scale1,
                float4(data2) * scale2,
                float4(data3) * scale3,
            );

            sum += mul(mat, vj);
        } else {
            // Dequantizing block 2
            let lid = local_id.x % 8u;
            let scale0 = unpackHalf2x16ToHalf(m[quant0 + 8]).y;
            let data0 = unpackInt4x8ToInt32(m[quant0 + lid + 9u]);
            let scale1 = unpackHalf2x16ToHalf(m[quant1 + 8]).y;
            let data1 = unpackInt4x8ToInt32(m[quant1 + lid + 9u]);
            let scale2 = unpackHalf2x16ToHalf(m[quant2 + 8]).y;
            let data2 = unpackInt4x8ToInt32(m[quant2 + lid + 9u]);
            let scale3 = unpackHalf2x16ToHalf(m[quant3 + 8]).y;
            let data3 = unpackInt4x8ToInt32(m[quant3 + lid + 9u]);

            let mat = float4x4(
                float4(data0) * scale0,
                float4(data1) * scale1,
                float4(data2) * scale2,
                float4(data3) * scale3,
            );

            sum += mul(mat, vj);
        }
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

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L43-L46
// Contains two `BlockQ8_0(f16, [i8; 32])`.
struct BlockQ8_0x2 {
    uint data[17];
}
