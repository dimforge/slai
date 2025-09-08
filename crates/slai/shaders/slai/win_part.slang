import stensor.linalg.shape;

static const uint WORKGROUP_SIZE = 128;

/// source: [R1, C2, M1, T1]
/// result: [R1, W, W, _]
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void win_part(
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

    let w = shape_result.nmats;
    let pad_x = (w - shape_source.nrows % w) % w;
    // NOTE: notation nep0 === npx
    let nep0 = (pad_x + shape_source.nrows) / w;

    // NOTE: id[3] spans [0..nep0*nep1[ by definition of the result tensor.
    let py = id[3] / nep0;
    let px = id[3] - py * nep0;
    let i02 = py * w + id[2];
    let i01 = px * w + id[0];
    let i00 = id[1];

    if (py * w + id[2] >= shape_source.nmats || px * w + id[0] >= shape_source.nrows) {
        result[i] = 0.0;
    } else {
        let j = shape_source.it(i01, i00, i02);
        result[i] = source[j];
    }
}


/// source: [R1, C2, M1, T1]
/// result: [R1, W, W, _]
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void win_unpart(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<uint> w,
    ConstantBuffer<Shape> shape_result,
    ConstantBuffer<Shape> shape_source,
    RWStructuredBuffer<float> result,
    StructuredBuffer<float> source,
) {
    if (invocation_id.x >= shape_result.len()) {
        return;
    }

    let id = shape_result.decompose(invocation_id.x);
    let j = shape_result.it(id);

    let px = (w - shape_result.nrows % w) % w;
    let npx = (px + shape_result.nrows) / w;
    let ip2 = id[2] / w;
    let ip1 = id[0] / w;
    let i = shape_source.it(
        id[0] % w,
        id[1],
        id[2] % w,
        ip2 * npx + ip1
    );

    result[j] = source[i];
}