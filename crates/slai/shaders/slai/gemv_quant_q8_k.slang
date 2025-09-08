import stensor.linalg.shape;


static const int WORKGROUP_SIZE = 32;

// TODO: needs optimizations.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<BlockQ8K> m,
    StructuredBuffer<float4> v,
) {
    if (invocation_id.x < shape_m.nrows) {
        let i_ref = shape_v.iv(0u);
        let i_out = shape_out.iv(invocation_id.x);
        var sum = 0.0;

        for (var j = 0u; j < shape_m.ncols; j++) {
            let quant = m[shape_m.im(invocation_id.x, j)];
            let dequant = quant.dequantize();

            // Unroll calculation with all block elements.
            let i_base = i_ref + j * 64u;

            for (var k = 0u; k < 64u; k += 16u) {
                sum += dot(dequant[k + 0u], v[k + i_base + 0u]) +
                    dot(dequant[k + 1u], v[k + i_base + 1u]) +
                    dot(dequant[k + 2u], v[k + i_base + 2u]) +
                    dot(dequant[k + 3u], v[k + i_base + 3u]) +
                    dot(dequant[k + 4u], v[k + i_base + 4u]) +
                    dot(dequant[k + 5u], v[k + i_base + 5u]) +
                    dot(dequant[k + 6u], v[k + i_base + 6u]) +
                    dot(dequant[k + 7u], v[k + i_base + 7u]) +
                    dot(dequant[k + 8u], v[k + i_base + 8u]) +
                    dot(dequant[k + 9u], v[k + i_base + 9u]) +
                    dot(dequant[k + 10u], v[k + i_base + 10u]) +
                    dot(dequant[k + 11u], v[k + i_base + 11u]) +
                    dot(dequant[k + 12u], v[k + i_base + 12u]) +
                    dot(dequant[k + 13u], v[k + i_base + 13u]) +
                    dot(dequant[k + 14u], v[k + i_base + 14u]) +
                    dot(dequant[k + 15u], v[k + i_base + 15u]);
            }
        }

        out[i_out] = sum;
    }
}

// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L161-L165
struct BlockQ8K {
    /// Delta
    float d;
    /// Quants
    uint qs[64]; // [i8; 256]
    /// Sum of quants in groups of 16
    uint bsums[9]; // [i16; 256 / 16]

    func dequantize() -> float4[64] {
        float4 result[64];

        for (var j = 0; j < 64; j++) {
            // TODO: bsums is not used? Is this useful
            //       for optimizing matmul?
            let qs = unpackInt4x8ToInt32(this.qs[j]);
            result[j] = float4(qs) * this.d;
        }

        return result;
    }
}

