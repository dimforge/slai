import stensor.linalg.shape;


static const int WORKGROUP_SIZE = 64;

// TODO: needs a lot of optimizations.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void gemv(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_out,
    ConstantBuffer<Shape> shape_m,
    ConstantBuffer<Shape> shape_v,
    RWStructuredBuffer<float4> out,
    StructuredBuffer<BlockQ5_0x2> m,
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
            let i_base = i_ref + j * 16u;
            sum += dot(dequant[0], v[i_base + 0]) +
                dot(dequant[1], v[i_base + 1]) +
                dot(dequant[2], v[i_base + 2]) +
                dot(dequant[3], v[i_base + 3]) +
                dot(dequant[4], v[i_base + 4]) +
                dot(dequant[5], v[i_base + 5]) +
                dot(dequant[6], v[i_base + 6]) +
                dot(dequant[7], v[i_base + 7]) +
                dot(dequant[8], v[i_base + 8]) +
                dot(dequant[9], v[i_base + 9]) +
                dot(dequant[10], v[i_base + 10]) +
                dot(dequant[11], v[i_base + 11]) +
                dot(dequant[12], v[i_base + 12]) +
                dot(dequant[13], v[i_base + 13]) +
                dot(dequant[14], v[i_base + 14]) +
                dot(dequant[15], v[i_base + 15]);
        }

        out[i_out] = sum;
    }
}



// Contains two `BlockQ5_0(f16, uint, [i8; 16])`.
// See https://github.com/ggerganov/ggml/blob/fca1caafea7de9fbd7efc733b9818f9cf2da3050/src/ggml-quants.h#L26-L30
struct BlockQ5_0x2 {
    uint data[11];

    static func dequantize_part(j0: uint, qh: uint, data: uint, scale: float) -> float4[2] {
        let xh0 = ((qh >> j0) << 4) & 0x10;
        let x0 = int(((data >> (0u * 8u)) & 0x0F) | xh0) - 16;
        let xh1 = (qh >> (j0 + 12)) & 0x10;
        let x1 = int(((data >> (0u * 8u + 4u)) & 0x0F) | xh1) - 16;
        let xh2 = ((qh >> (j0 + 1)) << 4) & 0x10;
        let x2 = int(((data >> (1u * 8u)) & 0x0F) | xh2) - 16;
        let xh3 = (qh >> ((j0 + 1) + 12)) & 0x10;
        let x3 = int(((data >> (1u * 8u + 4u)) & 0x0F) | xh3) - 16;
        let xh4 = ((qh >> (j0 + 2)) << 4) & 0x10;
        let x4 = int(((data >> (2u * 8u)) & 0x0F) | xh4) - 16;
        let xh5 = (qh >> ((j0 + 2) + 12)) & 0x10;
        let x5 = int(((data >> (2u * 8u + 4u)) & 0x0F) | xh5) - 16;
        let xh6 = ((qh >> (j0 + 3)) << 4) & 0x10;
        let x6 = int(((data >> (3u * 8u)) & 0x0F) | xh6) - 16;
        let xh7 = (qh >> ((j0 + 3) + 12)) & 0x10;
        let x7 = int(((data >> (3u * 8u + 4u)) & 0x0F) | xh7) - 16;
        return {
            float4(float(x0), float(x2), float(x4), float(x6)) * scale,
            float4(float(x1), float(x3), float(x5), float(x7)) * scale
        };
    }

    // See https://github.com/ggerganov/ggml/blob/a3c0188a4b5d3dec052ff87c9f773baa53631d70/src/ggml-quants.c#L1556
    func dequantize() -> float4[16] {
        let zero = float4(0.0);
        var result: float4[16];

        // First block.
        let d1 = unpackHalf2x16ToHalf(this.data[0]).x;
        let qh1 = this.data[0] >> 16u | this.data[1] << 16u;

        for (var k = 0u; k < 4u; k++) {
            let data = this.data[k + 1] >> 16u | this.data[k + 2] << 16u;
            let parts = BlockQ5_0x2.dequantize_part(k * 4, qh1, data, d1);
            result[k] = parts[0];
            result[4 + k] = parts[1];
        }

        // Second block.
        let d2 = unpackHalf2x16ToHalf(this.data[5]).y;
        let qh2 = this.data[6];

        for (var k = 0u; k < 4u; k++) {
            let data = this.data[k + 7];
            let parts = BlockQ5_0x2.dequantize_part(k * 4, qh2, data, d2);
            result[8 + k] = parts[0];
            result[12 + k] = parts[1];
        }

        return result;
    }
}