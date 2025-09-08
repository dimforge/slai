import stensor.linalg.shape;

struct RmsNormConfig {
    float nudge_factor;
}

static const int WORKGROUP_SIZE = 128;

/*
 * Magnitude.
 */
groupshared float workspace[WORKGROUP_SIZE];

func reduce_sum(index: uint, stride: uint) {
    if (index < stride) {
        workspace[index] += workspace[index + stride];
    }

    GroupMemoryBarrierWithGroupSync();
}


func magnitude_squared(
    thread_id: uint,
    ConstantBuffer<Shape>  shape_v,
    StructuredBuffer<float> v,
) -> float {
    workspace[thread_id] = 0.0;

    for (var i = thread_id; i < shape_v.nrows; i += WORKGROUP_SIZE) {
        let val_i = v[shape_v.iv(i)];
        workspace[thread_id] += val_i * val_i;
    }

    GroupMemoryBarrierWithGroupSync();

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    return workspace[0];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void rms_norm(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_v,
    ConstantBuffer<Shape> shape_w,
    ConstantBuffer<Shape> shape_out,
    StructuredBuffer<float> v,
    StructuredBuffer<float> w,
    RWStructuredBuffer<float> out,
    ConstantBuffer<RmsNormConfig> config,
) {
    let magnitude_sq = magnitude_squared(invocation_id.x, shape_v, v);

    let len = shape_v.nrows;
    let rms = 1.0 / sqrt((magnitude_sq / float(len)) + config.nudge_factor);

    for (var i = invocation_id.x; i < len; i += WORKGROUP_SIZE) {
        out[shape_out.iv(i)] = (v[shape_v.iv(i)] * rms) * w[shape_w.iv(i)];
    }
}
