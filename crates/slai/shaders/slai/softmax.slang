import stensor.linalg.shape;


static const int WORKGROUP_SIZE = 64;

groupshared float workspace[WORKGROUP_SIZE];
groupshared float the_max;
groupshared float denominator;

func reduce_max(index: uint, stride: uint) {
    GroupMemoryBarrierWithGroupSync();
    if (index < stride) {
        workspace[index] = max(workspace[index], workspace[index + stride]);
    }
}

func reduce_sum(index: uint, stride: uint) {
    GroupMemoryBarrierWithGroupSync();
    if (index < stride) {
        workspace[index] += workspace[index + stride];
    }
}

// Runs softmax on each individual column of the tensor.
// To run on the rows, transpose the tensor and then call this kernel.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void softmax_cols(
    uint3 workgroup_id: SV_GroupID,
    uint3 local_id: SV_GroupThreadID,
    ConstantBuffer<Shape> shape,
    RWStructuredBuffer<float> in_out_mat,
) {
    let j = workgroup_id.x;
    let k = workgroup_id.y;
    let l = workgroup_id.z;
    let thread_id = local_id.x;

    // Compute the MAX
    let data_len = shape.nrows;
    var my_max = -1.0e38;

    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let val_i = in_out_mat[shape.it(i, j, k, l)];
        my_max = max(my_max, val_i);
    }

    workspace[thread_id] = my_max;

//    reduce_max(thread_id, 64u);
    reduce_max(thread_id, 32u);
    reduce_max(thread_id, 16u);
    reduce_max(thread_id, 8u);
    reduce_max(thread_id, 4u);
    reduce_max(thread_id, 2u);
    reduce_max(thread_id, 1u);

    if (thread_id == 0) {
        the_max = workspace[0];
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute the denominator (sum of exponentials).
    var my_denominator = 0.0;
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = shape.it(i, j, k, l);
        let val_i = in_out_mat[ii];
        let exp_i = exp(val_i - the_max);
        my_denominator += exp_i;
        in_out_mat[ii] = exp_i;
    }

    workspace[thread_id] = my_denominator;

//    reduce_sum(thread_id, 64u);
    reduce_sum(thread_id, 32u);
    reduce_sum(thread_id, 16u);
    reduce_sum(thread_id, 8u);
    reduce_sum(thread_id, 4u);
    reduce_sum(thread_id, 2u);
    reduce_sum(thread_id, 1u);

    if (thread_id == 0) {
        denominator = workspace[0];
    }

    GroupMemoryBarrierWithGroupSync();

    // Divide by the denominator.
    for (var i = thread_id; i < data_len; i += WORKGROUP_SIZE) {
        let ii = shape.it(i, j, k, l);
        let val_i = in_out_mat[ii];
        in_out_mat[ii] = val_i / denominator;
    }
}
