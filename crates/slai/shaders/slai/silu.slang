import stensor.linalg.shape;

static const int WORKGROUP_SIZE = 64;

// SwiGLU non-linearity.
func swish(x: float, beta: float) -> float {
    // This is the swiglu function from https://youtu.be/Mn_9W1nCFLo?si=LT6puSAfzgpP6ydz&t=3973
    return x / (1.0 + exp(-beta * x));
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void silu(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_a,
    ConstantBuffer<Shape> shape_b,
    RWStructuredBuffer<float> in_out_a,
    StructuredBuffer<float> in_b,
) {
    if (invocation_id.x < shape_a.nrows) {
        let ia = shape_a.iv(invocation_id.x);
        let ib = shape_b.iv(invocation_id.x);
        let lhs = in_out_a[ia];
        let rhs = in_b[ib];
        in_out_a[ia] = rhs * swish(lhs, 1.0);
    }
}
