import stensor.linalg.shape;

static const uint WORKGROUP_SIZE = 128;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void get_rel_pos(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_result,
    ConstantBuffer<Shape> shape_source,
    RWStructuredBuffer<float> result,
    StructuredBuffer<float> source,
) {
    if (invocation_id.x >= shape_result.len()) {
        return;
    }

    let id = shape_result.decompose(invocation_id.x);
    let i = shape_result.it(id);
    let w = shape_result.nrows;
    let pos = (w - id[0] - 1) + id[2];
    let j = shape_source.im(pos, id[1]);

    result[i] = source[j];
}

// TODO PERF: merge both phases into a single kernel.
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void add_rel_pos_phase2(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_src1,
    RWStructuredBuffer<float> dst,
    StructuredBuffer<float> src2,
) {
    if (invocation_id.x >= shape_src1.len()) {
        return;
    }

    let id = shape_src1.decompose(invocation_id.x);
    let jp0 = shape_src1.it(id);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359
    let src2_e = src2[jp0];
    let ne10 = shape_src1.ncols;

    let jdh = jp0 * ne10;
    let jdw = jdh - (ne10 - 1) * id.y;

    for (var j = 0; j < ne10; ++j) {
        dst[jdh + j     ] += src2_e;
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void add_rel_pos_phase1(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_src1,
    RWStructuredBuffer<float> dst,
    StructuredBuffer<float> src1,
) {
    if (invocation_id.x >= shape_src1.len()) {
        return;
    }

    let id = shape_src1.decompose(invocation_id.x);
    let jp0 = shape_src1.it(id);

    // ref: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L357-L359
    let src1_e = src1[jp0];
    let ne10 = shape_src1.ncols;

    let jdh = jp0 * ne10;
    let jdw = jdh - (ne10 - 1) * id.y;

    for (var j = 0; j < ne10; ++j) {
        dst[jdw + j * ne10] += src1_e;
    }
}