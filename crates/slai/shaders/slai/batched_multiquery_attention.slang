// From https://github.com/seddonm1/web-llm/blob/main/src/shaders/multihead_attn.wgsl (MIT/Apache license).

struct Params {
    uint seq_len;
    uint kv_dim;
    uint kv_mul;
    uint n_heads;
    uint head_size;
    uint pos;
};

static const int WORKGROUP_SIZE = 64;

func div_ceil4(a: uint) -> uint {
    return (a + 3u) / 4u;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void mult_mask_attn(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Params> params,
    RWStructuredBuffer<float> attn,
) {
    let nonzero_len = params.pos + 1u;
    let aligned_len = div_ceil4(params.pos + 1u) * 4u;
    if (invocation_id.x % aligned_len < nonzero_len)  {
        attn[invocation_id.x] /= sqrt(float(params.head_size));
    } else {
        attn[invocation_id.x] = 0.0;
    }
}