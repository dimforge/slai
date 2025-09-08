import stensor.linalg.shape;


static const int WORKGROUP_SIZE = 64;

struct RoPEConfig {
    uint head_size;
    uint kv_dim;
    uint pos;
    float base_freq;
}

// TODO: import from stensor?
struct Rotation2 {
    float cos;
    float sin;
}

func rot2(angle: float) -> Rotation2 {
    return Rotation2(cos(angle), sin(angle));
}

func rotate2(r: Rotation2, vx: float, vy: float) -> float2 {
    return float2(r.cos * vx - r.sin * vy, r.sin * vx + r.cos * vy);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void rope(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_q,
    ConstantBuffer<Shape> shape_k,
    ConstantBuffer<RoPEConfig> config,
    RWStructuredBuffer<float> in_out_q,
    RWStructuredBuffer<float> in_out_k,
) {
    let i = invocation_id.x;
    let head_dim = float((i * 2) % config.head_size);
    let theta = pow(config.base_freq, -head_dim / float(config.head_size));
    let m_theta = float(config.pos) * theta;
    let rot = rot2(m_theta);

    let iq = shape_q.iv(i * 2);
    let q_rotated = rotate2(rot, in_out_q[iq], in_out_q[iq + 1]);
    in_out_q[iq] = q_rotated.x;
    in_out_q[iq + 1] = q_rotated.y;

    if (i * 2 < config.kv_dim) {
        let ik = shape_k.iv(i * 2);
        let k_rotated = rotate2(rot, in_out_k[ik], in_out_k[ik + 1]);
        in_out_k[ik] = k_rotated.x;
        in_out_k[ik + 1] = k_rotated.y;
    }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void rope_neox(
    uint3 invocation_id: SV_DispatchThreadID,
    ConstantBuffer<Shape> shape_q,
    ConstantBuffer<Shape> shape_k,
    ConstantBuffer<RoPEConfig> config,
    RWStructuredBuffer<float> in_out_q,
    RWStructuredBuffer<float> in_out_k,
) {
    let i = invocation_id.x;
    let head_dim = float((i * 2) % config.head_size);
    let theta = pow(config.base_freq, -head_dim / float(config.head_size));
    let m_theta = float(config.pos) * theta;
    let rot = rot2(m_theta);

    let head_id = (i * 2) / config.head_size;
    let shift = config.head_size / 2;

    let iq = shape_q.iv(i + head_id * config.head_size / 2);
    let q_rotated = rotate2(rot, in_out_q[iq], in_out_q[iq + shift]);
    in_out_q[iq] = q_rotated.x;
    in_out_q[iq + shift] = q_rotated.y;

    if (i * 2 < config.kv_dim) {
        let ik = shape_k.iv(i + head_id * config.head_size / 2);
        let k_rotated = rotate2(rot, in_out_k[ik], in_out_k[ik + shift]);
        in_out_k[ik] = k_rotated.x;
        in_out_k[ik + shift] = k_rotated.y;
    }
}