import stensor.linalg.shape;

static const int WORKGROUP_SIZE = 128;

// TODO: make the epsilon configurable
static const float NUDGE_FACTOR = 1.0e-6;

groupshared float workspace[WORKGROUP_SIZE];
groupshared float the_mean;
groupshared float scale;

func reduce_sum(index: uint, stride: uint) {
    GroupMemoryBarrierWithGroupSync();
    if (index < stride) {
        workspace[index] += workspace[index + stride];
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void layernorm_cols(
    uint3 wid: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> in_shape,
    ConstantBuffer<Shape> out_shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
) {
    let thread_id = local_id.x;

    // Compute the MEAN
    let data_len = in_shape.nrows;
    workspace[thread_id] = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = input[in_shape.it(i, wid.x, wid.y, wid.z)];
        workspace[thread_id] += val_i;
    }

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        the_mean = workspace[0] / float(data_len);
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute the SQUARED NORM
    workspace[thread_id] = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = input[in_shape.it(i, wid.x, wid.y, wid.z)] - the_mean;
        workspace[thread_id] += val_i * val_i;
    }

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        let variance = workspace[0] / float(data_len);
        scale = 1.0 / sqrt(variance + NUDGE_FACTOR);
    }

    GroupMemoryBarrierWithGroupSync();

    // Apply the scale.
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = in_shape.it(i, wid.x, wid.y, wid.z);
        let iout = out_shape.it(i, wid.x, wid.y, wid.z);
        output[iout] = (input[ii] - the_mean) * scale;
    }
}


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void layernorm_rows(
    uint3 wid: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> in_shape,
    ConstantBuffer<Shape> out_shape,
    StructuredBuffer<float> input,
    RWStructuredBuffer<float> output,
) {
    let thread_id = local_id.x;

    // Compute the MEAN
    let data_len = in_shape.ncols;
    workspace[thread_id] = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = input[in_shape.it(wid.x, i, wid.y, wid.z)];
        workspace[thread_id] += val_i;
    }

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        the_mean = workspace[0] / float(data_len);
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute the SQUARED NORM
    workspace[thread_id] = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = input[in_shape.it(wid.x, i, wid.y, wid.z)] - the_mean;
        workspace[thread_id] += val_i * val_i;
    }

    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        let variance = workspace[0] / float(data_len);
        scale = 1.0 / sqrt(variance + NUDGE_FACTOR);
    }

    GroupMemoryBarrierWithGroupSync();

    // Apply the scale.
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = in_shape.it(wid.x, i, wid.y, wid.z);
        let iout = out_shape.it(wid.x, i, wid.y, wid.z);
        output[iout] = (input[ii] - the_mean) * scale;
    }
}
