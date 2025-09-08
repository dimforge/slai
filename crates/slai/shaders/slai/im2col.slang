// NOTE: this was ported from llama.cpp `im2col.comp` (vulkan backend)

struct Im2ColParams {
    uint batch_offset;
    uint offset_delta;
    uint IC;
    uint IW;
    uint IH;
    uint OW;
    uint OH;
    uint KW;
    uint KH;
    uint pelements;
    uint CHW;
    int s0;
    int s1;
    int p0;
    int p1;
    int d0;
    int d1;
}

static const uint WORKGROUP_SIZE = 32;
static const uint NUM_ITER = 512 / WORKGROUP_SIZE;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void im2col(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Im2ColParams> params,
    StructuredBuffer<float> in_tensor,
    RWStructuredBuffer<float> out_tensor,
) {
    const uint gidx = invocation_id.x;
    const uint oh = invocation_id.y;
    const uint batch = invocation_id.z / params.IC;
    const uint ic = invocation_id.z % params.IC;

    const uint src_base = ic * params.offset_delta + batch * params.batch_offset;
    const uint dst_base = ((batch * params.OH + oh) * params.OW) * params.CHW + ic * (params.KW * params.KH);
    const int oh_s1 = int(oh) * params.s1;
    const uint ksize = params.OW * (params.KH > 1 ? params.KW : 1);

    const uint base_linear_idx = gidx * NUM_ITER;

    const uint max_ky = ksize / params.OW;

    uint current_kx = base_linear_idx / ksize;
    const uint rem = base_linear_idx - (current_kx * ksize);
    uint current_ky = rem / params.OW;
    uint current_ix = rem % params.OW;

    float values[NUM_ITER];
    uint offset_dst[NUM_ITER];

    [unroll]
    for (uint idx = 0; idx < NUM_ITER; ++idx) {
        values[idx] = 0.0;
    }

    [unroll]
    for (uint idx = 0; idx < NUM_ITER; ++idx) {
        const uint linear_idx = base_linear_idx + idx;

        if (linear_idx >= params.pelements) {
            continue;
        }

        const uint iiw = current_ix * params.s0 + current_kx * params.d0 - params.p0;
        const uint iih = oh_s1 + current_ky * params.d1 - params.p1;

        offset_dst[idx] = dst_base + current_ix * params.CHW + current_ky * params.KW + current_kx;

        if ((iih < params.IH) && (iiw < params.IW)) {
            values[idx] = in_tensor[src_base + iih * params.IW + iiw];
        }

        if (++current_ix == params.OW) {
            current_ix = 0;
            if (++current_ky == max_ky) {
                current_ky = 0;
                current_kx++;
            }
        }
    }

    [unroll]
    for (uint idx = 0; idx < NUM_ITER; ++idx) {

        const uint linear_idx = base_linear_idx + idx;

        if (linear_idx >= params.pelements) {
            continue;
        }

        out_tensor[offset_dst[idx]] = values[idx];
    }

}
