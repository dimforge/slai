// This was mostly ported from ggml-cuda/unary.cu
// (MIT license).

import stensor.linalg.shape;
import stensor.utils.trig;
import stensor.utils.limits;


static const uint WORKGROUP_SIZE = 64;
// TODO: slang doesn’t have a way to get the total number of workgroups?
//       The max here is based on webgpu’s limitations.
static const uint MAX_NUM_THREADS = MAX_NUM_WORKGROUPS * WORKGROUP_SIZE;


interface IUnOp {
    [ForceInline]
    static func unop(a: float) -> float;
}

interface IUnOpWithArgs {
    [ForceInline]
    static func unop(a: float, args: float4) -> float;
}

struct Abs: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return abs(x);
    }
}

struct Sgn: IUnOp {
    static func unop(x: float) -> float {
        return sign(x);
    }
}

struct Neg: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return -x;
    }
}

struct Step: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        if (x > 0.0) {
            return 1.0;
        } else {
            return 0.0;
        }
    }
}

struct Elu: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        if (x > 0.0) {
            return x;
        } else {
            return exp(x) - 1.0;
        }
    }
}

struct Gelu: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        static const float GELU_COEF_A = 0.044715f;
        static const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
        return 0.5f * x * (1.0f + stable_tanh(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

struct GeluQuick: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        static const float GELU_QUICK_COEF = -1.702f;
        return x * (1.0f / (1.0f + exp(GELU_QUICK_COEF * x)));
    }
}

struct Silu: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return x / (1.0 + exp(-x));
    }
}

struct Tanh: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return stable_tanh(x);
    }
}

struct Relu: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return max(x, 0.0);
    }
}

struct Sigmoid: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return 1.0 / (1.0 + exp(-x));
    }
}

struct HardSigmoid: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return min(1.0, max(0.0, (x + 3.0) / 6.0));
    }
}

struct HardSwish: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return x * min(1.0, max(0.0, (x + 3.0) / 6.0));
    }
}

struct Sqr: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return x * x;
    }
}

struct Sqrt: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return sqrt(x);
    }
}

struct Sin: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return sin(x);
    }
}

struct Cos: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return cos(x);
    }
}

struct Log: IUnOp {
    [ForceInline]
    static func unop(x: float) -> float {
        return log(x);
    }
}

/* Unary ops with extra arguments passed through the args uniform. */


struct LeakyRelu: IUnOpWithArgs {
    [ForceInline]
    static func unop(x: float, args: float4) -> float {
        return max(x, 0.0) + min(x, 0.0) * args.x;
    }
}

struct Clamp: IUnOpWithArgs {
    [ForceInline]
    static func unop(x: float, args: float4) -> float {
        return min(max(args.x, x), args.y);
    }
}

struct Scale: IUnOpWithArgs {
    [ForceInline]
    static func unop(x: float, args: float4) -> float {
        return x * args.x;
    }
}

struct AddScalar: IUnOpWithArgs {
    [ForceInline]
    static func unop(x: float, args: float4) -> float {
        return x + args.x;
    }
}

[ForceInline]
func main<Op: IUnOp>(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_dst,
    ConstantBuffer<Shape> shape_src,
    RWStructuredBuffer<float> dst,
    RWStructuredBuffer<float> src,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_src.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it(id);
        let idst = shape_dst.it(id);
        dst[idst] = Op.unop(src[isrc]);
    }
}

[ForceInline]
func main_with_args<Op: IUnOpWithArgs>(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_dst,
    ConstantBuffer<Shape> shape_src,
    RWStructuredBuffer<float> dst,
    RWStructuredBuffer<float> src,
    ConstantBuffer<float4> args,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_src.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it(id);
        let idst = shape_dst.it(id);
        dst[idst] = Op.unop(src[isrc], args);
    }
}

[ForceInline]
func main_inplace<Op: IUnOp>(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_src,
    RWStructuredBuffer<float> src,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_src.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it(id);
        src[isrc] = Op.unop(src[isrc]);
    }
}

[ForceInline]
func main_with_args_inplace<Op: IUnOpWithArgs>(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_src,
    RWStructuredBuffer<float> src,
    ConstantBuffer<float4> args,
) {
    for (var thread_id = invocation_id.x; thread_id < shape_src.len(); thread_id += MAX_NUM_THREADS) {
        let id = shape_src.decompose(thread_id);
        let isrc = shape_src.it(id);
        src[isrc] = Op.unop(src[isrc], args);
    }
}

/* NOTE: keep all the entrypoints this way or do link-time specialization? */
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
// NOTE: can't call it `abs` because of name clash with the `abs` builtin.
func abs_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Abs>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func abs_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Abs>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sgn_op(
  uint3 invocation_id: SV_DispatchThreadID,
  ConstantBuffer<Shape> shape_dst,
  ConstantBuffer<Shape> shape_src,
  RWStructuredBuffer<float> dst,
  RWStructuredBuffer<float> src,
) {
    main<Sgn>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sgn_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Sgn>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func neg_op(
  uint3 invocation_id: SV_DispatchThreadID,
  ConstantBuffer<Shape> shape_dst,
  ConstantBuffer<Shape> shape_src,
  RWStructuredBuffer<float> dst,
  RWStructuredBuffer<float> src,
) {
    main<Neg>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func neg_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Neg>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func step_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Step>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func step_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Step>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func elu_op(
  uint3 invocation_id: SV_DispatchThreadID,
  ConstantBuffer<Shape> shape_dst,
  ConstantBuffer<Shape> shape_src,
  RWStructuredBuffer<float> dst,
  RWStructuredBuffer<float> src,
) {
    main<Elu>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func elu_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Elu>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func gelu_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Gelu>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func gelu_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Gelu>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func gelu_quick_op(
 uint3 invocation_id: SV_DispatchThreadID,
 ConstantBuffer<Shape> shape_dst,
 ConstantBuffer<Shape> shape_src,
 RWStructuredBuffer<float> dst,
 RWStructuredBuffer<float> src,
) {
    main<GeluQuick>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func gelu_quick_inplace(
      uint3 invocation_id: SV_DispatchThreadID,
      ConstantBuffer<Shape> shape_src,
      RWStructuredBuffer<float> src,
  ) {
    main_inplace<GeluQuick>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func silu_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Silu>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func silu_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Silu>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func tanh_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Tanh>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func tanh_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Tanh>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func relu_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Relu>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func relu_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Relu>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sigmoid_op(
  uint3 invocation_id: SV_DispatchThreadID,
  ConstantBuffer<Shape> shape_dst,
  ConstantBuffer<Shape> shape_src,
  RWStructuredBuffer<float> dst,
  RWStructuredBuffer<float> src,
) {
    main<Sigmoid>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sigmoid_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Sigmoid>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func hard_sigmoid_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<HardSigmoid>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func hard_sigmoid_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<HardSigmoid>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sqr_op(
  uint3 invocation_id: SV_DispatchThreadID,
  ConstantBuffer<Shape> shape_dst,
  ConstantBuffer<Shape> shape_src,
  RWStructuredBuffer<float> dst,
  RWStructuredBuffer<float> src,
) {
    main<Sqr>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sqr_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Sqr>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sqrt_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Sqrt>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sqrt_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Sqrt>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sin_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Sin>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func sin_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Sin>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func cos_op(
   uint3 invocation_id: SV_DispatchThreadID,
   ConstantBuffer<Shape> shape_dst,
   ConstantBuffer<Shape> shape_src,
   RWStructuredBuffer<float> dst,
   RWStructuredBuffer<float> src,
) {
    main<Cos>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func cos_inplace(
        uint3 invocation_id: SV_DispatchThreadID,
        ConstantBuffer<Shape> shape_src,
        RWStructuredBuffer<float> src,
    ) {
    main_inplace<Cos>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func log_op(
  uint3 invocation_id: SV_DispatchThreadID,
  ConstantBuffer<Shape> shape_dst,
  ConstantBuffer<Shape> shape_src,
  RWStructuredBuffer<float> dst,
  RWStructuredBuffer<float> src,
) {
    main<Log>(invocation_id, shape_dst, shape_src, dst, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func log_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
   ) {
    main_inplace<Log>(invocation_id, shape_src, src);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func leaky_relu_op(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_dst,
    ConstantBuffer<Shape> shape_src,
    RWStructuredBuffer<float> dst,
    RWStructuredBuffer<float> src,
    ConstantBuffer<float4> args,
) {
    main_with_args<LeakyRelu>(invocation_id, shape_dst, shape_src, dst, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func leaky_relu_inplace(
      uint3 invocation_id: SV_DispatchThreadID,
      ConstantBuffer<Shape> shape_src,
      RWStructuredBuffer<float> src,
      ConstantBuffer<float4> args,
  ) {
    main_with_args_inplace<LeakyRelu>(invocation_id, shape_src, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func clamp_op(
     uint3 invocation_id: SV_DispatchThreadID,
     ConstantBuffer<Shape> shape_dst,
     ConstantBuffer<Shape> shape_src,
     RWStructuredBuffer<float> dst,
     RWStructuredBuffer<float> src,
     ConstantBuffer<float4> args,
 ) {
    main_with_args<Clamp>(invocation_id, shape_dst, shape_src, dst, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func clamp_inplace(
     uint3 invocation_id: SV_DispatchThreadID,
     ConstantBuffer<Shape> shape_src,
     RWStructuredBuffer<float> src,
     ConstantBuffer<float4> args,
 ) {
    main_with_args_inplace<Clamp>(invocation_id, shape_src, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func scale_op(
     uint3 invocation_id: SV_DispatchThreadID,
     ConstantBuffer<Shape> shape_dst,
     ConstantBuffer<Shape> shape_src,
     RWStructuredBuffer<float> dst,
     RWStructuredBuffer<float> src,
     ConstantBuffer<float4> args,
 ) {
    main_with_args<Scale>(invocation_id, shape_dst, shape_src, dst, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func scale_inplace(
     uint3 invocation_id: SV_DispatchThreadID,
     ConstantBuffer<Shape> shape_src,
     RWStructuredBuffer<float> src,
     ConstantBuffer<float4> args,
 ) {
    main_with_args_inplace<Scale>(invocation_id, shape_src, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func add_scalar_op(
      uint3 invocation_id: SV_DispatchThreadID,
      ConstantBuffer<Shape> shape_dst,
      ConstantBuffer<Shape> shape_src,
      RWStructuredBuffer<float> dst,
      RWStructuredBuffer<float> src,
      ConstantBuffer<float4> args,
  ) {
    main_with_args<AddScalar>(invocation_id, shape_dst, shape_src, dst, src, args);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
func add_scalar_inplace(
       uint3 invocation_id: SV_DispatchThreadID,
       ConstantBuffer<Shape> shape_src,
       RWStructuredBuffer<float> src,
       ConstantBuffer<float4> args,
   ) {
    main_with_args_inplace<AddScalar>(invocation_id, shape_src, src, args);
}