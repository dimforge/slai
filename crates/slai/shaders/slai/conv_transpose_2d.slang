// NOTE: this was ported from ggml
import stensor.linalg.shape;

static const uint WORKGROUP_SIZE = 64;

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void init_dest(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_dest,
    RWStructuredBuffer<float> dest,
) {
    if (invocation_id.x >= shape_dest.len()) {
        return;
    }

    dest[invocation_id.x] = 0.0;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void init_wdata(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_wdata,
    RWStructuredBuffer<float> wdata,
) {
    if (invocation_id.x >= shape_wdata.len()) {
        return;
    }

    wdata[invocation_id.x] = 0.0;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void init_src0(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_src0,
    StructuredBuffer<float> src0,
    RWStructuredBuffer<float> wdata,
) {
    if (invocation_id.x >= shape_src0.len()) {
        return;
    }

    // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
    let id = shape_src0.decompose(invocation_id.x);
    let id_wdata =
        id.z * shape_src0.nrows * shape_src0.ncols * shape_src0.ncubes
        + id.x * shape_src0.ncols * shape_src0.ncubes
        + id.y * shape_src0.ncubes
        + id.w;
    wdata[id_wdata] = src0[invocation_id.x];
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void init_src1(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_src0,
    ConstantBuffer<Shape> shape_src1,
    StructuredBuffer<float> src1,
    RWStructuredBuffer<float> wdata,
) {
    if (invocation_id.x >= shape_src1.len()) {
        return;
    }

    // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
    let nk = shape_src0.len();
    let id = shape_src1.decompose(invocation_id.x);
    let id_wdata =
        nk
        + id.x * shape_src1.ncols * shape_src1.nmats
        + id.y * shape_src1.nmats
        + id.z;
    wdata[id_wdata] = src1[invocation_id.x];
}


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void conv_transpose_2d_ref(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<uint> stride,
    ConstantBuffer<Shape> shape_src0,
    ConstantBuffer<Shape> shape_src1,
    ConstantBuffer<Shape> shape_dest,
    StructuredBuffer<float> wdata,
    RWStructuredBuffer<float> dest,
) {
    let i2 = invocation_id.x;

    if (i2 >= shape_dest.nmats) {
        return;
    }

    let nk = shape_src0.len();
    let ne0 = shape_dest.ncols;
    let ne1 = shape_dest.nrows;
    let ne2 = shape_dest.nmats;
    let ne3 = shape_dest.ncubes;

    let nb0 = shape_dest.col_stride;
    let nb1 = shape_dest.row_stride;
    let nb2 = shape_dest.mat_stride;
    let nb3 = shape_dest.cube_stride;

    let ne00 = shape_src0.ncols;
    let ne01 = shape_src0.nrows;
    let ne02 = shape_src0.nmats;
    let ne03 = shape_src0.ncubes;
    let ne10 = shape_src1.ncols;
    let ne11 = shape_src1.nrows;
    let ne12 = shape_src1.nmats;
    let ne13 = shape_src1.ncubes;


    for (int i11 = 0; i11 < ne11; i11++) {
        for (int i10 = 0; i10 < ne10; i10++) {
            int i1n = i11*ne10*ne12 + i10*ne12;
            for (int i01 = 0; i01 < ne01; i01++) {
                for (int i00 = 0; i00 < ne00; i00++) {
                    float v = 0.0;

                    for (int k = 0; k < ne03; k++) {
                        v += wdata[nk + i1n + k] * wdata[i2*ne01*ne00*ne03 + i01*ne00*ne03 + i00*ne03 + k];
                    }
                    dest[i2 * nb2 + (i11*stride + i01)*ne0 + i10*stride + i00] += v;
                }
            }
        }
    }
}



//[shader("compute")]
//[numthreads(WORKGROUP_SIZE, 1, 1)]
//void conv_transpose_2d_phase1(
//    uint3 invocation_id: SV_DispatchThreadID,
//    StructuredBuffer<float> src1,
//    StructuredBuffer<float> src0,
//    RWStructuredBuffer<float> dest,
//) {
//    const int ith = params->ith;
//    const int nth = params->nth;
//
//    const int nk = ne00*ne01*ne02*ne03;
//
//    memset(params->wdata, 0, params->wsize);
//    memset(dst->data, 0, ggml_nbytes(dst));
//
//    // permute kernel data (src0) from (Kw x Kh x Cout x Cin) to (Cin x Kw x Kh x Cout)
//    {
//
//        for (int64_t i03 = 0; i03 < ne03; i03++) {
//            for (int64_t i02 = 0; i02 < ne02; i02++) {
//                for (int64_t i01 = 0; i01 < ne01; i01++) {
//                    for (int64_t i00 = 0; i00 < ne00; i00++) {
//                        params->wdata[i02*ne01*ne00*ne03 + i01*ne00*ne03 + i00*ne03 + i03] = src0->data[i03*nb03 + i02*nb02 + i01 * ne00 + i00];
//                    }
//                }
//            }
//        }
//    }
//
//    // permute source data (src1) from (Sw x Sh x Cin) to (Cin x Sw x Sh)
//    {
//        for (int i12 = 0; i12 < ne12; i12++) {
//            for (int i11 = 0; i11 < ne11; i11++) {
//                for (int i10 = 0; i10 < ne10; i10++) {
//                    params->wdata[nk + i11*ne10*ne12 + i10*ne12 + i12] = src1->data[i12*nb12 + i11*nb11 + i10];
//                }
//            }
//        }
//    }
//}


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void conv_transpose_2d(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<uint> stride,
    ConstantBuffer<Shape> shape_src1,
    ConstantBuffer<Shape> shape_src0,
    ConstantBuffer<Shape> shape_dest,
    StructuredBuffer<float> src1,
    StructuredBuffer<float> src0,
    RWStructuredBuffer<float> dest,
) {
    let i2 = invocation_id.x;

    if (i2 >= shape_dest.nmats) {
        return;
    }

    for (int k = 0; k < shape_dest.nrows * shape_dest.ncols; k++) {
        dest[i2 * shape_dest.nmats + k] = 0.0;
    }

    for (int i11 = 0; i11 < shape_src1.nrows; i11++) {
        for (int i10 = 0; i10 < shape_src1.ncols; i10++) {
            for (int i01 = 0; i01 < shape_src0.nrows; i01++) {
                for (int i00 = 0; i00 < shape_src0.ncols; i00++) {
                    float v = 0;

                    for (int k = 0; k < shape_src0.nmats; k++) {
                        v += src1[shape_src1.it(i10, k, i11)] * src0[shape_src0.it(i00, k, i01, i2)];
                    }

                    let dest_id = shape_dest.it((i11*stride + i01), (i10*stride + i00), i2);
                    dest[dest_id] += v;
                }
            }
        }
    }
}
