"""Fused Metal training step for the grouped per-plane probability networks.

One kernel computes the forward pass, the mixture loss and the backward pass
through the network for every sample; a second reduces the weight gradients
over batch chunks with simdgroup matrix products; a third sums the chunks;
a fourth clips each plane's gradient and applies Adam.
Training only: the decoder never runs this code, and models are exported to
the same ProbabilityNet tensors as before.
"""

from __future__ import annotations

import math

import numpy as np
import torch

_LIBRARIES: dict[tuple[int, ...], object] = {}
GROUP = 256
# Batch rows per gradient-reduction chunk; a multiple of the 8-row matrix tile.
CHUNK = 512
# Nets at least this wide keep training's per-sample scratch rows (the
# activations and gradients that pass from forward_backward to reduce) in
# binary16 instead of float: less memory traffic between the two kernels.
# Measured per step, that saved 14% at width 48 (batch 8192), 9% at width 32
# and 3% at width 12.
HALF_SCRATCH_WIDTH = 48
# In binary16 scratch rows, training's per-sample gradients are
# 2^SCRATCH_EXPONENT times their share of the batch mean, which keeps most of
# them out of the subnormal range; the chunk sums undo it exactly. Float rows
# stay unscaled: Metal flushes float subnormals to zero, so a scale there
# changes a few tiny gradients.
SCRATCH_EXPONENT = 6
# Output tiles per reduction simdgroup: BLOCK_ROWS tile rows by STRIP tile columns.
STRIP = 8
BLOCK_ROWS = 2
THREADGROUP_BYTES = 32 * 1024
MAX_HIDDEN = 64
# Search estimates (the evaluation kernels' mixture_bits, and coded_bits in
# neural_probability_probe) spend bits as the integer coder does: its logistic
# is exactly 0 or 1 beyond +-CODER_SATURATION (Distribution::sigmoid in
# nnproto_integer.cpp), a component whose range lies beyond adds no mass, and
# every value keeps one of CODER_TOTAL counts (arithmetic.hpp). EMPTY is the
# log of no mass. Training's objective clamps the mass instead, which in float
# rounds to nothing once a mean lies far below the range, and bits come out
# negative: on the lower half of PIA13757 the searched model was estimated at
# -3,042,707 bytes, and its file was 61,000 bytes larger than another rounding's.
CODER_SATURATION = 16
CODER_TOTAL = 1 << 20
EMPTY = -1e4


def layout(d: int, h: int, k: int) -> dict[str, tuple[int, tuple[int, int]]]:
    """Per-plane parameter block: name -> (offset, shape).

    w1a holds w1 in columns :h and the affine weight in column h, with an
    extra row d for b1 (row d, column h is unused and never updated).
    """
    r = 3 * k
    shapes = {"w1a": (d + 1, h + 1), "w2b": (h + 1, h), "w3b": (h + 1, r)}
    out, offset = {}, 0
    for name, shape in shapes.items():
        out[name] = (offset, shape)
        offset += shape[0] * shape[1]
    out["size"] = (offset, (0, 0))
    return out


def _pad(n: int) -> int:
    return -(-n // 8) * 8


def padded(d: int, h: int, k: int) -> dict[str, int]:
    """Row strides of the per-sample scratch: every reduction operand is 8-aligned."""
    da, ha, hb, ra = _pad(d + 1), _pad(h + 1), _pad(h), _pad(3 * k)
    return {"DA": da, "HA": ha, "HB": hb, "RA": ra, "QT": da * ha + ha * hb + ha * ra}


def units(pads: dict[str, int]) -> int:
    """Simdgroups per (chunk, plane) in the reduction: row blocks times column strips."""
    strips = [-(-pads[name] // 8 // STRIP) for name in ("HA", "HB", "RA")]
    rows = [-(-pads[name] // 8 // BLOCK_ROWS) for name in ("DA", "HA", "HA")]
    return sum(r * s for r, s in zip(rows, strips, strict=True))


def _fits(d: int, h: int, k: int) -> bool:
    # Room for the evaluation kernel's per-simdgroup sums next to the weights.
    return 4 * layout(d, h, k)["size"][0] + 4 * GROUP // 32 <= THREADGROUP_BYTES


# Network code in the kernels. The scalar form reads the parameter block as
# laid out; the vector form keeps it in threadgroup memory with the affine
# column split off, so every hidden and head row is a run of float4 loads.
# Both perform the same multiply-adds into each accumulator in the same order,
# so they compute the same bits; the vector form issues a quarter of the loads.
# First-layer loops stop at the plane's own input count DG: the inputs and
# weights past it are zero, so skipping them leaves every sum unchanged.
_SCALAR = {
    "l1_fb": """    float h1[H];
    for (int j = 0; j < H; j++) h1[j] = w[D * H1 + j];
    float A = 0.0f;
    for (int i = 0; i < DG; i++) {
        half xh = xr[i];
        float xi = float(xh);
        xo[i] = xh;
        A += xi * w[i * H1 + H];
        for (int j = 0; j < H; j++) h1[j] += xi * w[i * H1 + j];
    }""",
    "l2_fb": """    float raw[R];
    for (int c = 0; c < R; c++) raw[c] = w[O3 + H * R + c];
    for (int u = 0; u < H; u++) {
        float s = w[O2 + H * H + u];
        for (int j = 0; j < H; j++) s += h1[j] * w[O2 + j * H + u];
        s = max(s, 0.0f);
        z2o[u] = s;
        for (int c = 0; c < R; c++) raw[c] += s * w[O3 + u * R + c];
    }""",
    "back_fb": """    float back[H];
    for (int j = 0; j < H; j++) back[j] = 0.0f;
    for (int u = 0; u < H; u++) {
        float s = 0.0f;
        for (int c = 0; c < R; c++) s += w[O3 + u * R + c] * draw[c];
        s = z2o[u] > 0.0f ? s : 0.0f;
        d2o[u] = s;
        for (int j = 0; j < H; j++) back[j] += w[O2 + j * H + u] * s;
    }""",
    "l1_sb": """    float h1[H];
    for (int j = 0; j < H; j++) h1[j] = w[D * H1 + j];
    float A = 0.0f;
    for (int i = 0; i < DG; i++) {
        float xi = float(xr[i]);
        A += xi * w[i * H1 + H];
        for (int j = 0; j < H; j++) h1[j] += xi * w[i * H1 + j];
    }""",
    "l2_sb": """    float raw[R];
    for (int c = 0; c < R; c++) raw[c] = w[O3 + H * R + c];
    for (int u = 0; u < H; u++) {
        float s = w[O2 + H * H + u];
        for (int j = 0; j < H; j++) s += h1[j] * w[O2 + j * H + u];
        s = max(s, 0.0f);
        if (STORE) z2o[u] = s;
        for (int c = 0; c < R; c++) raw[c] += s * w[O3 + u * R + c];
    }""",
}
_FIRST = """    float h1[H];
    for (int q = 0; q < H / 4; q++) {
        float4 v = ((T4)(w + D * H))[q];
        h1[4 * q] = v.x; h1[4 * q + 1] = v.y; h1[4 * q + 2] = v.z; h1[4 * q + 3] = v.w;
    }
    float A = 0.0f;
    for (int i = 0; i < DG; i++) {
        %s
        A += xi * w[VA + i];
        T4 r4 = (T4)(w + i * H);
        for (int q = 0; q < H / 4; q++) {
            float4 v = r4[q];
            h1[4 * q] += xi * v.x; h1[4 * q + 1] += xi * v.y;
            h1[4 * q + 2] += xi * v.z; h1[4 * q + 3] += xi * v.w;
        }
    }"""
# With D a multiple of 4, every input row is a run of half4 loads. The loop
# rounds DG up to a multiple of 4; those inputs and their weights are zero.
# It is 5 to 11% faster up to width 32 and 4% slower at width 48.
_FIRST4 = """    float h1[H];
    for (int q = 0; q < H / 4; q++) {
        float4 v = ((T4)(w + D * H))[q];
        h1[4 * q] = v.x; h1[4 * q + 1] = v.y; h1[4 * q + 2] = v.z; h1[4 * q + 3] = v.w;
    }
    float A = 0.0f;
    for (int i4 = 0; i4 < (DG + 3) / 4; i4++) {
        half4 x4 = ((device const half4*)xr)[i4];
        %s
        for (int t = 0; t < 4; t++) {
            int i = 4 * i4 + t;
            float xi = float(x4[t]);
            A += xi * w[VA + i];
            T4 r4 = (T4)(w + i * H);
            for (int q = 0; q < H / 4; q++) {
                float4 v = r4[q];
                h1[4 * q] += xi * v.x; h1[4 * q + 1] += xi * v.y;
                h1[4 * q + 2] += xi * v.z; h1[4 * q + 3] += xi * v.w;
            }
        }
    }"""
FIRST4_MAX_HIDDEN = 32
_SECOND = """    float raw[R];
    for (int c = 0; c < R; c++) raw[c] = w[VW3 + H * R + c];
    for (int u0 = 0; u0 < H; u0 += 4) {
        float4 s4 = *((T4)(w + VW2 + H * H + u0));
        for (int j = 0; j < H; j++) s4 += h1[j] * *((T4)(w + VW2 + j * H + u0));
        for (int t = 0; t < 4; t++) {
            float s = max(s4[t], 0.0f);
            %s
            T4 h4 = (T4)(w + VW3 + (u0 + t) * R);
            for (int q = 0; q < R / 4; q++) {
                float4 v = h4[q];
                raw[4 * q] += s * v.x; raw[4 * q + 1] += s * v.y;
                raw[4 * q + 2] += s * v.z; raw[4 * q + 3] += s * v.w;
            }
        }
    }"""
_VECTOR = {
    "l1_fb": _FIRST % "half xh = xr[i];\n        float xi = float(xh);\n        xo[i] = xh;",
    "l2_fb": _SECOND % "z2o[u0 + t] = s;",
    "back_fb": """    float back[H];
    for (int j = 0; j < H; j++) back[j] = 0.0f;
    for (int u0 = 0; u0 < H; u0 += 4) {
        float s[4];
        for (int t = 0; t < 4; t++) {
            int u = u0 + t;
            float acc = 0.0f;
            T4 h4 = (T4)(w + VW3 + u * R);
            for (int q = 0; q < R / 4; q++) {
                float4 v = h4[q];
                acc += v.x * draw[4 * q]; acc += v.y * draw[4 * q + 1];
                acc += v.z * draw[4 * q + 2]; acc += v.w * draw[4 * q + 3];
            }
            acc = z2o[u] > 0.0f ? acc : 0.0f;
            d2o[u] = acc;
            s[t] = acc;
        }
        for (int j = 0; j < H; j++) {
            float4 v = *((T4)(w + VW2 + j * H + u0));
            back[j] += v.x * s[0]; back[j] += v.y * s[1];
            back[j] += v.z * s[2]; back[j] += v.w * s[3];
        }
    }""",
    "l1_sb": _FIRST % "float xi = float(xr[i]);",
    "l2_sb": _SECOND % "if (STORE) z2o[u0 + t] = s;",
}
# Vector-form threadgroup layout: w1 rows (stride H, bias row last), the affine
# column, w2b and w3b; every block starts on a float4.
_VECTOR_LOAD = """threadgroup float4 w4[VP / 4];
    threadgroup float* w = (threadgroup float*)w4;
    device const float* pg = params + size_t(g) * P;
    for (uint i = lid; i < uint(P); i += lsize) {
        int at;
        if (i < uint(O2)) {
            int r = i / H1, c = i % H1;
            at = c < H ? r * H + c : (r < D ? VA + r : -1);
        } else if (i < uint(O3)) {
            at = VW2 + int(i - O2);
        } else {
            at = VW3 + int(i - O3);
        }
        if (at >= 0) w[at] = pg[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);"""
VECTOR = True


def _vector_size(d: int, h: int, k: int) -> int:
    return (d + 1) * h + _pad4(d) + (h + 1) * h + (h + 1) * 3 * k


def _pad4(n: int) -> int:
    return -(-n // 4) * 4


def _vector(d: int, h: int, k: int) -> bool:
    return (
        VECTOR
        and h % 4 == 0
        and k == 4
        and 4 * _vector_size(d, h, k) + 4 * GROUP // 32 <= THREADGROUP_BYTES
    )


def _source(d: int, h: int, k: int) -> str:
    lay = layout(d, h, k)
    o2, o3, size = lay["w2b"][0], lay["w3b"][0], lay["size"][0]
    pads = padded(d, h, k)
    vector = _vector(d, h, k)
    pieces = dict(_VECTOR if vector else _SCALAR)
    if vector and d % 4 == 0 and h <= FIRST4_MAX_HIDDEN:
        pieces["l1_fb"] = _FIRST4 % "((device half4*)xo)[i4] = x4;"
        pieces["l1_sb"] = _FIRST4 % ""
    head = pieces["l2_sb"].replace("    float raw[R];\n", "", 1)
    if vector:
        layer = "#define W3(u, c) w[VW3 + (u) * R + (c)]\n#define W3B(c) w[VW3 + H * R + (c)]\n"
        layer += "#define WAFF(i) w[VA + (i)]"
    else:
        layer = "#define W3(u, c) w[O3 + (u) * R + (c)]\n#define W3B(c) w[O3 + H * R + (c)]\n"
        layer += "#define WAFF(i) w[(i) * H1 + H]"
    va = (d + 1) * h
    vw2 = va + _pad4(d)
    vw3 = vw2 + (h + 1) * h
    if vector:
        space = "threadgroup"
        load = _VECTOR_LOAD
    elif _fits(d, h, k):
        space = "threadgroup"
        load = """threadgroup float w[P];
    device const float* pg = params + size_t(g) * P;
    for (uint i = lid; i < uint(P); i += lsize) w[i] = pg[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);"""
    else:
        # Too large for threadgroup memory; every thread of a group reads the
        # same addresses, so device reads stay cache broadcasts.
        space = "device"
        load = "device const float* w = params + size_t(g) * P;"
    source = f"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant constexpr int D = {d};
constant constexpr int H = {h};
constant constexpr int K = {k};
constant constexpr int R = 3 * K;
constant constexpr int H1 = H + 1;
constant constexpr int O2 = {o2};
constant constexpr int O3 = {o3};
constant constexpr int P = {size};
constant constexpr int DA = {pads["DA"]}, HA = {pads["HA"]}, HB = {pads["HB"]}, RA = {pads["RA"]};
constant constexpr int Q1 = DA * HA, Q2 = HA * HB, QT = {pads["QT"]};
constant constexpr int CHUNK = {CHUNK};
constant constexpr int VA = {va}, VW2 = {vw2}, VW3 = {vw3}, VP = {_vector_size(d, h, k)};
typedef threadgroup const float4* T4;
#define WSPACE {space}
{layer}

inline float log1p_acc(float e) {{
    float u = 1.0f + e;
    return u == 1.0f ? e : log(u) * e / (u - 1.0f);
}}
inline float expm1_acc(float x) {{
    float u = exp(x);
    if (u == 1.0f) return x;
    float um1 = u - 1.0f;
    if (um1 == -1.0f) return -1.0f;
    return um1 * x / log(u);
}}
inline float logsigmoid(float x) {{ return min(x, 0.0f) - log1p_acc(exp(-fabs(x))); }}
inline float sigm(float x) {{
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    float e = exp(x);
    return e / (1.0f + e);
}}
// log(s(upper) - s(lower)) for upper - lower = width > 0, s the integer coder's
// logistic; EMPTY where both ends saturate alike.
inline float log_sigmoid_difference(float upper, float lower, float width) {{
    const float limit = {CODER_SATURATION}.0f;
    if (lower >= limit || upper <= -limit) return {EMPTY}f;
    bool high = upper >= limit, low = lower <= -limit;
    if (high) return low ? 0.0f : logsigmoid(-lower);
    if (low) return logsigmoid(upper);
    return logsigmoid(upper) + logsigmoid(-lower) + log(max(-expm1_acc(-width), 1e-12f));
}}
// logsigmoid(x), with s = sigm(-x) from the same exponential.
inline float logsigmoid_with(float x, thread float& s) {{
    float e = exp(-fabs(x)), u = 1.0f + e;
    s = (x <= 0.0f ? 1.0f : e) / u;
    return min(x, 0.0f) - (u == 1.0f ? e : log(u) * e / (u - 1.0f));
}}
// -expm1_acc(-v), with e = exp(-v).
inline float one_minus_exp(float v, thread float& e) {{
    e = exp(-v);
    if (e == 1.0f) return v;
    if (e == 0.0f) return 1.0f;
    return (e - 1.0f) * v / log(e);
}}

kernel void forward_backward(
    device const float* params, device const half* x, device const float* y,
    device const float* a, device const int* index, device const int* dims, device half* xb,
    device SCRATCH* e1, device SCRATCH* z1e, device SCRATCH* d2, device SCRATCH* z2e,
    device SCRATCH* dr, device float* bits, constant uint& N, constant uint& B, constant uint& BPAD,
    constant float& scale, uint t [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint lsize [[threads_per_threadgroup]])
{{
    uint g = t / BPAD, b = t % BPAD;
    const int DG = dims[g];
    {load}
    uint b8 = (B + 7) / 8 * 8;
    if (b >= b8) return;
    size_t out = size_t(g) * b8 + b;
    if (b >= B) {{
        // Rows up to the 8-row tile boundary enter the reduction as zeros.
        for (int i = 0; i < DA; i++) xb[out * DA + i] = 0.0h;
        for (int i = 0; i < HA; i++) {{ e1[out * HA + i] = 0.0f; z1e[out * HA + i] = 0.0f; }}
        for (int i = 0; i < HA; i++) z2e[out * HA + i] = 0.0f;
        for (int i = 0; i < HB; i++) d2[out * HB + i] = 0.0f;
        for (int i = 0; i < RA; i++) dr[out * RA + i] = 0.0f;
        return;
    }}
    size_t row = size_t(g) * N + uint(index[b]);
    device const half* xr = x + row * D;
    // Columns DG to D of xb stay zero from allocation.
    device half* xo = xb + out * DA;
{pieces["l1_fb"]}
    xo[D] = 1.0h;
    // Activations go to the scratch rows as soon as they exist and are read
    // back for the ReLU masks, so at most one hidden-width array stays live
    // (fewer registers, more threads in flight).
    device SCRATCH* z1o = z1e + out * HA;
    device SCRATCH* z2o = z2e + out * HA;
    for (int j = 0; j < H; j++) {{
        h1[j] = max(h1[j], 0.0f);
        z1o[j] = h1[j];
    }}
    z1o[H] = 1.0f;
{pieces["l2_fb"]}
    z2o[H] = 1.0f;
    float target = y[row], anchor = a[row];
    float lmax = raw[0];
    for (int i = 1; i < K; i++) lmax = max(lmax, raw[i]);
    float lsum = 0.0f;
    for (int i = 0; i < K; i++) lsum += exp(raw[i] - lmax);
    float lse = lmax + log(lsum);
    // Exponentials are shared where one serves two terms: sigm(-u) with
    // logsigmoid(u), 1 / expm1(inv) = exp(-inv) / em, and the mixture
    // posteriors with their normalizing sums.
    float vmix[K], wmix[K], dum[K], dul[K], dmm[K], dml[K];
    float vmax = -INFINITY, wmax = -INFINITY;
    for (int i = 0; i < K; i++) {{
        float m = raw[K + i] + A;
        float ls = clamp(raw[2 * K + i], -4.0f, 7.0f);
        float inv = exp(-ls);
        float u = (target + 0.5f - m) * inv, l = (target - 0.5f - m) * inv;
        float decay;
        float em = one_minus_exp(inv, decay);
        float gt = 0.0f;
        if (em < 1e-12f) em = 1e-12f; else gt = decay / em;
        float su, sl;
        float lp = logsigmoid_with(u, su) + logsigmoid_with(-l, sl) + log(em);
        float pu = su + gt, pl = -sl - gt;
        dum[i] = -inv * (pu + pl);
        dul[i] = -(u * pu + l * pl);
        float top = (255.5f - anchor - m) * inv, bot = (-0.5f - anchor - m) * inv;
        float st = sigm(top), sb = sigm(bot);
        float mass = st - sb;
        float logmix = raw[i] - lse;
        if (mass < 1e-10f) {{
            mass = 1e-10f;
            dmm[i] = 0.0f; dml[i] = 0.0f;
        }} else {{
            float dt = st * (1.0f - st), db = sb * (1.0f - sb);
            dmm[i] = -inv * (dt - db) / mass;
            dml[i] = -(top * dt - bot * db) / mass;
        }}
        vmix[i] = logmix + lp;
        wmix[i] = logmix + log(mass);
        vmax = max(vmax, vmix[i]);
        wmax = max(wmax, wmix[i]);
    }}
    float ev[K], ew[K];
    float vs = 0.0f, ws = 0.0f;
    for (int i = 0; i < K; i++) {{
        ev[i] = exp(vmix[i] - vmax);
        ew[i] = exp(wmix[i] - wmax);
        vs += ev[i];
        ws += ew[i];
    }}
    float lp_all = vmax + log(vs), normal = wmax + log(ws);
    bits[size_t(g) * B + b] = (normal - lp_all) * 1.4426950408889634f;
    float draw[R];
    float dA = 0.0f;
    float rv = 1.0f / vs, rw = 1.0f / ws;
    for (int i = 0; i < K; i++) {{
        float r = ev[i] * rv, q = ew[i] * rw;
        draw[i] = (q - r) * scale;
        float dm = (q * dmm[i] - r * dum[i]) * scale;
        draw[K + i] = dm;
        dA += dm;
        float lsr = raw[2 * K + i];
        draw[2 * K + i] = (lsr >= -4.0f && lsr <= 7.0f) ? (q * dml[i] - r * dul[i]) * scale : 0.0f;
    }}
    device SCRATCH* dro = dr + out * RA;
    for (int c = 0; c < R; c++) dro[c] = draw[c];
    device SCRATCH* d2o = d2 + out * HB;
    device SCRATCH* e1o = e1 + out * HA;
{pieces["back_fb"]}
    for (int j = 0; j < H; j++) e1o[j] = z1o[j] > 0.0f ? back[j] : 0.0f;
    e1o[H] = dA;
}}

// One sample's bits from its head outputs raw and affine prediction A, as the
// integer coder spends them (CODER_SATURATION).
inline float mixture_bits(thread const float* raw, float A, float target, float anchor)
{{
    float lmax = raw[0];
    for (int i = 1; i < K; i++) lmax = max(lmax, raw[i]);
    float lsum = 0.0f;
    for (int i = 0; i < K; i++) lsum += exp(raw[i] - lmax);
    float lse = lmax + log(lsum);
    float vmix[K], wmix[K];
    float vmax = -INFINITY, wmax = -INFINITY;
    for (int i = 0; i < K; i++) {{
        float m = raw[K + i] + A;
        float inv = exp(-clamp(raw[2 * K + i], -4.0f, 7.0f));
        float u = (target + 0.5f - m) * inv, l = (target - 0.5f - m) * inv;
        float top = (255.5f - anchor - m) * inv, bot = (-0.5f - anchor - m) * inv;
        float logmix = raw[i] - lse;
        vmix[i] = logmix + log_sigmoid_difference(u, l, inv);
        wmix[i] = logmix + log_sigmoid_difference(top, bot, 256.0f * inv);
        vmax = max(vmax, vmix[i]);
        wmax = max(wmax, wmix[i]);
    }}
    float vs = 0.0f, ws = 0.0f;
    for (int i = 0; i < K; i++) {{ vs += exp(vmix[i] - vmax); ws += exp(wmix[i] - wmax); }}
    float normal = wmax + log(ws);
    if (normal < {EMPTY / 2}f) return 8.0f;
    // -log2 of one count plus (TOTAL - 256) counts times the share exp(-d).
    float d = normal - vmax - log(vs);
    float x = d - {math.log(CODER_TOTAL)!r}f, y = {math.log1p(-256 / CODER_TOTAL)!r}f;
    return (d - max(x, y) - log1p_acc(exp(-fabs(x - y)))) * 1.4426950408889634f;
}}

// The second layer and head: raw from the first layer's outputs h1. Each
// second-layer unit goes straight into the head, so only h1 and raw stay live
// (fewer registers, more threads in flight). With STORE, z2o gets the second
// layer's outputs.
template <bool STORE>
inline void head_outputs(WSPACE const float* w, thread const float* h1, thread float* raw,
                         device float* z2o)
{{
{head}
}}

// Bits of one sample under the weights w: forward_backward's forward pass and
// objective without the backward pass. With STORE it writes the layer
// outputs, head outputs and affine prediction instead (the same values).
template <bool STORE>
inline float sample_pass(WSPACE const float* w, device const half* xr, float target, float anchor,
                         int DG, device float* z1o, device float* z2o, device float* rawo,
                         device float* Ao)
{{
{pieces["l1_sb"]}
    for (int j = 0; j < H; j++) {{
        h1[j] = max(h1[j], 0.0f);
        if (STORE) z1o[j] = h1[j];
    }}
    float raw[R];
    head_outputs<STORE>(w, h1, raw, z2o);
    if (STORE) {{
        for (int c = 0; c < R; c++) rawo[c] = raw[c];
        *Ao = A;
        return 0.0f;
    }}
    return mixture_bits(raw, A, target, anchor);
}}

inline float sample_bits(WSPACE const float* w, device const half* xr, float target, float anchor,
                         int DG)
{{
    return sample_pass<false>(w, xr, target, anchor, DG, nullptr, nullptr, nullptr, nullptr);
}}

// partial[block] = summed v over one threadgroup's block of samples.
inline void block_sum(float v, threadgroup float* sums, device float* partial, uint lid,
                      uint lsize, uint sid, uint lane)
{{
    v = simd_sum(v);
    if (lane == 0) sums[sid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {{
        float s = 0.0f;
        for (uint i = 0; i < lsize / 32; i++) s += sums[i];
        *partial = s;
    }}
}}

// partial[c][block] = summed bits of candidate c (weights params[c], inputs of
// plane[c]) over one threadgroup's block of samples.
kernel void evaluate(
    device const float* params, device const int* plane, device const int* dims,
    device const half* x, device const float* y, device const float* a, device float* partial,
    constant uint& N, constant uint& NPAD, uint t [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint lsize [[threads_per_threadgroup]],
    uint tg [[threadgroup_position_in_grid]], uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    uint g = t / NPAD, b = t % NPAD;
    {load}
    float v = 0.0f;
    if (b < N) {{
        size_t row = size_t(plane[g]) * N + b;
        v = sample_bits(w, x + row * D, y[row], a[row], dims[plane[g]]);
    }}
    threadgroup float sums[{GROUP // 32}];
    block_sum(v, sums, partial + tg, lid, lsize, sid, lane);
}}

// Plane G's layer outputs, head outputs and affine predictions under the
// weights params[0], exactly as sample_bits computes them.
kernel void activations(
    device const float* params, device const int* dims, device const half* x, device float* z1o,
    device float* z2o, device float* rawo, device float* Ao, constant uint& N, constant uint& G,
    uint t [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]])
{{
    uint g = 0, b = t;
    {load}
    if (b >= N) return;
    size_t row = size_t(G) * N + b;
    sample_pass<true>(w, x + row * D, 0.0f, 0.0f, dims[G], z1o + size_t(b) * H,
                      z2o + size_t(b) * H, rawo + size_t(b) * R, Ao + b);
}}

// evaluate for candidates params[c] of plane G that equal the activations'
// weights before one stage: 1 (the second layer and head change), 2 (the head)
// or 3 (only the affine weights). Each sample's bits are sample_bits'.
kernel void evaluate_from(
    device const float* params, device const int* dims, device const half* x,
    device const float* y, device const float* a, device const float* z1c,
    device const float* z2c, device const float* rawc, device const float* Ac,
    device float* partial, constant uint& N, constant uint& NPAD, constant uint& G,
    constant uint& stage, uint t [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]], uint lsize [[threads_per_threadgroup]],
    uint tg [[threadgroup_position_in_grid]], uint sid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{{
    uint g = t / NPAD, b = t % NPAD;
    {load}
    float v = 0.0f;
    if (b < N) {{
        size_t row = size_t(G) * N + b;
        float raw[R];
        float A = Ac[b];
        if (stage == 1) {{
            float h1[H];
            for (int j = 0; j < H; j++) h1[j] = z1c[size_t(b) * H + j];
            head_outputs<false>(w, h1, raw, nullptr);
        }} else if (stage == 2) {{
            for (int c = 0; c < R; c++) raw[c] = W3B(c);
            for (int u = 0; u < H; u++) {{
                float s = z2c[size_t(b) * H + u];
                for (int c = 0; c < R; c++) raw[c] += s * W3(u, c);
            }}
        }} else {{
            for (int c = 0; c < R; c++) raw[c] = rawc[size_t(b) * R + c];
            device const half* xr = x + row * D;
            A = 0.0f;
            for (int i = 0; i < dims[G]; i++) A += float(xr[i]) * WAFF(i);
        }}
        v = mixture_bits(raw, A, y[row], a[row]);
    }}
    threadgroup float sums[{GROUP // 32}];
    block_sum(v, sums, partial + tg, lid, lsize, sid, lane);
}}

constant constexpr int S1 = (D + 1) * H1;
constant constexpr int S2 = H1 * H;
constant constexpr int SKIP = D * H1 + H;

constant constexpr int NC = {STRIP}, RB = {BLOCK_ROWS};
constant constexpr int C1 = HA / 8, C2 = HB / 8, C3 = RA / 8;
constant constexpr int N1 = (C1 + NC - 1) / NC, N2 = (C2 + NC - 1) / NC, N3 = (C3 + NC - 1) / NC;
constant constexpr int M1 = (DA / 8 + RB - 1) / RB, M2 = (HA / 8 + RB - 1) / RB;
constant constexpr int U1 = M1 * N1, U2 = M2 * N2, UT = U1 + U2 + M2 * N3;

// One simdgroup accumulates a block of up to RB x NC 8x8 tiles of A^T E over
// batch rows [begin, end). Every tile sees the same multiply-accumulate
// sequence whatever the block shape, so the shape changes only the loads.
// Tile rows in [lo, hi) are all-zero columns of A; they are skipped, and their
// output tiles keep the zeros the partial buffer was allocated with.
template <typename T, typename U>
inline void block(device const T* A, int lda, device const U* E, int lde, device float* out,
                  int ti, int tj, int rows, int count, uint begin, uint end, int lo, int hi)
{{
    bool live[RB];
    bool any = false;
    for (int r = 0; r < RB; r++) {{
        live[r] = r < rows && (ti + r < lo || ti + r >= hi);
        any = any || live[r];
    }}
    if (!any) return;
    simdgroup_float8x8 acc[RB][NC];
    simdgroup_matrix<T, 8, 8> ma[RB];
    simdgroup_matrix<U, 8, 8> me;
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < NC; c++) acc[r][c] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k = begin; k < end; k += 8) {{
        for (int r = 0; r < RB; r++)
            if (live[r])
                simdgroup_load(ma[r], A + size_t(k) * lda + (ti + r) * 8, lda, ulong2(0, 0), true);
        for (int c = 0; c < NC; c++) {{
            if (c < count) {{
                simdgroup_load(me, E + size_t(k) * lde + (tj + c) * 8, lde);
                for (int r = 0; r < RB; r++)
                    if (live[r]) simdgroup_multiply_accumulate(acc[r][c], ma[r], me, acc[r][c]);
            }}
        }}
    }}
    for (int r = 0; r < RB; r++)
        for (int c = 0; c < NC; c++)
            if (live[r] && c < count)
                simdgroup_store(acc[r][c], out + (ti + r) * 8 * lde + (tj + c) * 8, lde);
}}

// partial[chunk][g] holds xb^T e1 (DA x HA), z1e^T d2 (HA x HB) and z2e^T dr
// (HA x RA) over one chunk of batch rows.
kernel void reduce(
    device const half* xb, device const SCRATCH* e1, device const SCRATCH* z1e,
    device const SCRATCH* d2, device const SCRATCH* z2e, device const SCRATCH* dr,
    device const int* dims, device float* partial, constant uint& B, constant uint& G,
    uint tg [[threadgroup_position_in_grid]])
{{
    uint b8 = (B + 7) / 8 * 8, chunks = (b8 + CHUNK - 1) / CHUNK;
    uint unit = tg % UT, rest = tg / UT, chunk = rest % chunks, g = rest / chunks;
    uint begin = chunk * CHUNK, end = min(b8, begin + CHUNK);
    device float* out = partial + (size_t(chunk) * G + g) * QT;
    size_t rows = size_t(g) * b8;
    if (unit < uint(U1)) {{
        int ti = unit / N1 * RB, tj = unit % N1 * NC;
        // Input tiles past the plane's own inputs and before the bias row are zero.
        block(xb + rows * DA, DA, e1 + rows * HA, HA, out, ti, tj, min(RB, DA / 8 - ti),
              min(NC, C1 - tj), begin, end, (dims[g] + 7) / 8, D / 8);
    }} else if (unit < uint(U1 + U2)) {{
        uint u = unit - U1;
        int ti = u / N2 * RB, tj = u % N2 * NC;
        block(z1e + rows * HA, HA, d2 + rows * HB, HB, out + Q1, ti, tj, min(RB, HA / 8 - ti),
              min(NC, C2 - tj), begin, end, 0, 0);
    }} else {{
        uint u = unit - U1 - U2;
        int ti = u / N3 * RB, tj = u % N3 * NC;
        block(z2e + rows * HA, HA, dr + rows * RA, RA, out + Q1 + Q2, ti, tj,
              min(RB, HA / 8 - ti), min(NC, C3 - tj), begin, end, 0, 0);
    }}
}}

inline float grad_at(int i, uint g, device const float* part, uint G, uint chunks) {{
    int q;
    if (i < S1) {{
        if (i == SKIP) return 0.0f;
        q = (i / H1) * HA + i % H1;
    }} else if (i < S1 + S2) {{
        int j = i - S1; q = Q1 + (j / H) * HB + j % H;
    }} else {{
        int j = i - S1 - S2; q = Q1 + Q2 + (j / R) * RA + j % R;
    }}
    float s = 0.0f;
    for (uint c = 0; c < chunks; c++) s += part[(size_t(c) * G + g) * QT + q];
    return s;
}}

kernel void quantize(device const float* p, device float* q, constant float& limit,
                     uint tg [[threadgroup_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint lsize [[threads_per_threadgroup]])
{{
    // Mirrors quantize_model._round on the hidden and head weight tensors;
    // biases (last row) and the affine column pass through unchanged.
    threadgroup float partial[256];
    uint g = tg / 3, t = tg % 3;
    int offset = t == 0 ? 0 : (t == 1 ? O2 : O3);
    int rows = t == 0 ? D + 1 : H1;
    int cols = t == 0 ? H1 : (t == 1 ? H : R);
    int qrows = t == 0 ? D : H, qcols = t == 0 ? H : cols;
    device const float* src = p + size_t(g) * P + offset;
    device float* dst = q + size_t(g) * P + offset;
    float s = 0.0f;
    for (int i = lid; i < rows * cols; i += lsize) {{
        int r = i / cols, c = i % cols;
        if (r < qrows && c < qcols) s = max(s, fabs(src[i]));
    }}
    partial[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize / 2; stride > 0; stride /= 2) {{
        if (lid < stride) partial[lid] = max(partial[lid], partial[lid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float top = partial[0];
    float scale = top > 0.0f ? exp2(floor(log2(limit / top))) : 1.0f;
    for (int i = lid; i < rows * cols; i += lsize) {{
        int r = i / cols, c = i % cols;
        float v = src[i];
        dst[i] = (top > 0.0f && r < qrows && c < qcols) ? rint(v * scale) / scale : v;
    }}
}}

// Every parameter's gradient, summed over the chunks in order, on all GPU cores;
// adam then runs one threadgroup per plane only for the clipping norm.
kernel void gradient_sums(device const float* part, device float* total, constant uint& G,
                          constant uint& chunks, constant float& unscale,
                          uint index [[thread_position_in_grid]])
{{
    if (index >= G * uint(P)) return;
    total[index] = grad_at(int(index % uint(P)), index / uint(P), part, G, chunks) * unscale;
}}

kernel void adam(device float* p, device float* m, device float* v, device const float* total,
                 device const float* keep,
                 constant float& lr, constant float& bc1, constant float& bc2,
                 constant float& maxnorm,
                 uint g [[threadgroup_position_in_grid]],
                 uint lid [[thread_position_in_threadgroup]],
                 uint lsize [[threads_per_threadgroup]])
{{
    threadgroup float partial[1024];
    device const float* grads = total + size_t(g) * P;
    float s = 0.0f;
    for (int i = lid; i < P; i += lsize) {{
        float gr = grads[i]; s += gr * gr;
    }}
    partial[lid] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = lsize / 2; stride > 0; stride /= 2) {{
        if (lid < stride) partial[lid] += partial[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    float factor = min(maxnorm / (sqrt(partial[0]) + 1e-6f), 1.0f);
    float step = lr / bc1, root = sqrt(bc2);
    for (int i = lid; i < P; i += lsize) {{
        size_t at = size_t(g) * P + i;
        // Pruned weights stay zero; their gradients still count in the clipping norm.
        if (keep[at] == 0.0f) continue;
        float gr = grads[i] * factor;
        float mi = 0.9f * m[at] + 0.1f * gr;
        float vi = 0.999f * v[at] + 0.001f * gr * gr;
        m[at] = mi; v[at] = vi;
        p[at] -= step * mi / (sqrt(vi) / root + 1e-8f);
    }}
}}
"""
    # forward_backward and reduce keep binary16 scratch rows (HALF_SCRATCH_WIDTH);
    # their _float copies keep float rows, for narrower nets and the weight
    # search's per-sample gradients.
    copies = (("forward_backward", "// One sample's bits"), ("reduce", "inline float grad_at"))
    for name, until in copies:
        start, end = source.index(f"kernel void {name}("), source.index(until)
        copy = source[start:end].replace(f"{name}(", f"{name}_float(", 1)
        source = source[:end] + copy.replace("SCRATCH", "float") + source[end:]
    return source.replace("SCRATCH", "half")


def supported(d: int, h: int, k: int) -> bool:
    return d <= 128 and h <= MAX_HIDDEN and k <= 4


def library(d: int, h: int, k: int):
    key = (d, h, k, STRIP, BLOCK_ROWS, CHUNK, _vector(d, h, k))
    if key not in _LIBRARIES:
        _LIBRARIES[key] = torch.mps.compile_shader(_source(d, h, k))
    return _LIBRARIES[key]


# Weighted Gram matrices for the encoder's GPTQ curvatures. One simdgroup sums
# a GRAM_BLOCK x GRAM_BLOCK block of 8x8 tiles of the upper triangle for one
# output row over one chunk of samples; each 8-sample step scales the left
# tiles by a diagonal matrix of that row's weights.
GRAM_BLOCK = 3
GRAM_GROUP = 4
# Simdgroups per call to aim for: enough sample chunks to fill the GPU.
GRAM_UNITS = 16384
_GRAM_SOURCE = """
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
constant constexpr int DP = {DP}, C = DP / 8, B = {B}, NB = (C + B - 1) / B;
constant constexpr int U = NB * (NB + 1) / 2, S = {S};

kernel void weighted_gram(device const float* x, device const float* w, device float* out,
                          constant uint& R, constant uint& L, constant uint& count,
                          uint tg [[threadgroup_position_in_grid]],
                          uint sid [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float scratch[S][4 * 64];
    uint unit = tg * S + sid;
    if (unit >= count) return;
    threadgroup float* diag = scratch[sid];
    uint block = unit % U, rest = unit / U, r = rest % R, chunk = rest / R;
    int s = block, bi = 0;
    while (s >= NB - bi) { s -= NB - bi; bi++; }
    int ti = bi * B, tj = (bi + s) * B;
    int rows = min(B, C - ti), cols = min(B, C - tj);
    for (int i = lane; i < 4 * 64; i += 32) diag[i] = 0.0f;
    simdgroup_float8x8 acc[B][B], a[B], b, d;
    for (int p = 0; p < B; p++)
        for (int q = 0; q < B; q++) acc[p][q] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint k0 = chunk * L; k0 < (chunk + 1) * L; k0 += 32) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        diag[(lane / 8) * 64 + (lane % 8) * 9] = w[size_t(k0 + lane) * R + r];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (int kk = 0; kk < 4; kk++) {
            device const float* xk = x + size_t(k0 + kk * 8) * DP;
            simdgroup_load(d, diag + kk * 64, 8);
            for (int p = 0; p < B; p++)
                if (p < rows) {
                    simdgroup_load(a[p], xk + (ti + p) * 8, DP, ulong2(0, 0), true);
                    simdgroup_multiply(a[p], a[p], d);
                }
            for (int q = 0; q < B; q++) {
                if (q < cols) {
                    simdgroup_load(b, xk + (tj + q) * 8, DP);
                    for (int p = 0; p < B; p++)
                        if (p < rows) simdgroup_multiply_accumulate(acc[p][q], a[p], b, acc[p][q]);
                }
            }
        }
    }
    device float* o = out + (size_t(chunk) * R + r) * DP * DP;
    for (int p = 0; p < B; p++)
        for (int q = 0; q < B; q++)
            if (p < rows && q < cols)
                simdgroup_store(acc[p][q], o + (ti + p) * 8 * DP + (tj + q) * 8, DP);
}
"""


def weighted_gram(inputs: torch.Tensor, weights: torch.Tensor, bias: bool) -> torch.Tensor:
    """(rows, m, m) sums of weights[:, r] * x x^T on MPS, where x is a row of
    inputs with a constant 1 appended if bias (m = d + bias)."""
    n, d = inputs.shape
    rows = weights.shape[1]
    m = d + int(bias)
    dp = _pad(m)
    blocks = -(-(dp // 8) // GRAM_BLOCK)
    per_chunk = rows * blocks * (blocks + 1) // 2
    chunks = max(1, min(-(-n // 32), -(-GRAM_UNITS // per_chunk)))
    length = -(-n // (chunks * 32)) * 32
    # Zero padding rows and columns add nothing to any sum.
    x = torch.zeros(chunks * length, dp, device=inputs.device)
    x[:n, :d] = inputs
    if bias:
        x[:n, d] = 1.0
    w = torch.zeros(chunks * length, rows, device=inputs.device)
    w[:n] = weights
    key = ("gram", dp, GRAM_BLOCK, GRAM_GROUP)
    if key not in _LIBRARIES:
        source = _GRAM_SOURCE.replace("{DP}", str(dp))
        source = source.replace("{B}", str(GRAM_BLOCK)).replace("{S}", str(GRAM_GROUP))
        _LIBRARIES[key] = torch.mps.compile_shader(source)
    out = torch.zeros(chunks, rows, dp, dp, device=inputs.device)
    count = chunks * per_chunk
    _LIBRARIES[key].weighted_gram(
        x,
        w,
        out,
        rows,
        length,
        count,
        threads=-(-count // GRAM_GROUP) * GRAM_GROUP * 32,
        group_size=GRAM_GROUP * 32,
    )
    upper = out.sum(0)[:, :m, :m].triu()
    return upper + upper.triu(1).mT


class FusedGroupedTrainer:
    """Adam with per-plane clipping on stacked G-plane networks, fused on MPS."""

    def __init__(
        self, nets, dimensions, x, y, a, lr: float, max_norm: float = 10.0, half_scratch=None
    ):
        self.g = len(nets)
        self.d = max(dimensions)
        self.h = nets[0].hidden[0].out_features
        self.half = self.h >= HALF_SCRATCH_WIDTH if half_scratch is None else half_scratch
        self.exponent = SCRATCH_EXPONENT if self.half else 0
        self.k = nets[0].mixtures
        self.dimensions = dimensions
        self.lay = layout(self.d, self.h, self.k)
        self.size = self.lay["size"][0]
        self.pads = padded(self.d, self.h, self.k)
        self.lib = library(self.d, self.h, self.k)
        # Integer features divided by 32 are exact in binary16; only the guide's
        # fractional-part column rounds. Half the memory of float32 inputs.
        self.x = x.to(torch.float16).contiguous()
        self.y = y.contiguous()
        self.a = a.contiguous()
        self.n = self.x.shape[1]
        self.lr = lr
        self.max_norm = max_norm
        self.params = torch.zeros(self.g, self.size, device=x.device)
        self.total = torch.empty_like(self.params)
        self.m = torch.zeros_like(self.params)
        self.v = torch.zeros_like(self.params)
        # Zero marks a pruned parameter, which Adam no longer updates.
        self.keep = torch.ones_like(self.params)
        self.dims = torch.tensor(dimensions, dtype=torch.int32, device=x.device)
        self.steps = 0
        self._buffers: dict[int, tuple] = {}
        with torch.no_grad():
            for g, (net, dim) in enumerate(zip(nets, dimensions, strict=True)):
                w1a, w2b, w3b = self.views(self.params[g])
                w1a[:dim, : self.h] = net.hidden[0].weight.T
                w1a[:dim, self.h] = net.affine.weight[0]
                w1a[self.d, : self.h] = net.hidden[0].bias
                w2b[: self.h] = net.hidden[2].weight.T
                w2b[self.h] = net.hidden[2].bias
                w3b[: self.h] = net.head.weight.T
                w3b[self.h] = net.head.bias

    def views(self, block):
        out = []
        for name in ("w1a", "w2b", "w3b"):
            offset, shape = self.lay[name]
            out.append(block[offset : offset + shape[0] * shape[1]].view(*shape))
        return out

    def _scratch(self, batch: int):
        if batch not in self._buffers:
            g, dev, p = self.g, self.x.device, self.pads
            rows = _pad(batch)
            chunks = -(-rows // CHUNK)
            scratch = torch.float16 if self.half else torch.float32
            self._buffers[batch] = (
                torch.zeros(g, rows, p["DA"], device=dev, dtype=torch.float16),
                *(
                    torch.zeros(g, rows, width, device=dev, dtype=scratch)
                    for width in (p["HA"], p["HA"], p["HB"], p["HA"], p["RA"])
                ),
                torch.empty(g, batch, device=dev),
                # Zeros: the reduction never writes the tiles of zero input columns.
                torch.zeros(chunks, g, p["QT"], device=dev),
            )
        return self._buffers[batch]

    def _reduce(self, index: torch.Tensor, params: torch.Tensor | None):
        """Forward/backward and chunked gradient reduction; returns (partials, bits)."""
        batch = int(index.numel())
        xb, e1, z1e, d2, z2e, dr, bits, partial = self._scratch(batch)
        bpad = -(-batch // GROUP) * GROUP
        lib = self.lib
        forward_backward = lib.forward_backward if self.half else lib.forward_backward_float
        forward_backward(
            self.params if params is None else params,
            self.x,
            self.y,
            self.a,
            index,
            self.dims,
            xb,
            e1,
            z1e,
            d2,
            z2e,
            dr,
            bits,
            self.n,
            batch,
            bpad,
            2.0**self.exponent / (batch * math.log(2)),
            threads=self.g * bpad,
            group_size=GROUP,
        )
        (lib.reduce if self.half else lib.reduce_float)(
            xb,
            e1,
            z1e,
            d2,
            z2e,
            dr,
            self.dims,
            partial,
            batch,
            self.g,
            threads=units(self.pads) * partial.shape[0] * self.g * 32,
            group_size=32,
        )
        return partial, bits

    def gradients(self, index: torch.Tensor, params: torch.Tensor | None = None):
        """Weight gradients (w1a, w2b, w3b blocks) and per-sample bits for `index`."""
        partial, bits = self._reduce(index, params)
        p, d, h, r = self.pads, self.d, self.h, 3 * self.k
        total = partial.sum(0) * 2.0**-self.exponent
        q1, q2 = p["DA"] * p["HA"], p["HA"] * p["HB"]
        g1 = total[:, :q1].view(self.g, p["DA"], p["HA"])[:, : d + 1, : h + 1]
        g2 = total[:, q1 : q1 + q2].view(self.g, p["HA"], p["HB"])[:, : h + 1, :h]
        g3 = total[:, q1 + q2 :].view(self.g, p["HA"], p["RA"])[:, : h + 1, :r]
        return g1, g2, g3, bits

    def quantized(self, bits: int) -> torch.Tensor:
        """Parameters with hidden/head weights on the post-training precision grid."""
        out = torch.empty_like(self.params)
        self.lib.quantize(
            self.params, out, float(2 ** (bits - 1) - 1), threads=self.g * 3 * 256, group_size=256
        )
        return out

    def state(self):
        return self.params.clone(), self.m.clone(), self.v.clone(), self.keep.clone(), self.steps

    def restore(self, state):
        params, m, v, keep, self.steps = state
        self.params.copy_(params)
        self.m.copy_(m)
        self.v.copy_(v)
        self.keep.copy_(keep)

    @torch.no_grad()
    def prune(self, columns) -> None:
        """Hold the first-layer weights of input columns[g] of each plane g at zero."""
        offset = self.lay["w1a"][0]
        for g, plane in enumerate(columns):
            for j in plane:
                start = offset + j * (self.h + 1)
                for buffer in (self.params, self.m, self.keep):
                    buffer[g, start : start + self.h] = 0.0

    @torch.no_grad()
    def fold(self, column: int, value: float) -> None:
        """Move the weights of an input that always equals value into the first-layer bias."""
        for g in range(self.g):
            w1a, _w2b, _w3b = self.views(self.params[g])
            w1a[self.d, : self.h] += w1a[column, : self.h] * value
        self.prune([[column]] * self.g)

    def step(self, index: torch.Tensor, lr: float, bits: int | None = None):
        """One Adam step; with `bits`, a straight-through quantization-aware step."""
        forward = None if bits is None else self.quantized(bits)
        partial, bits = self._reduce(index, forward)
        self.steps += 1
        bc1 = 1 - 0.9**self.steps
        bc2 = 1 - 0.999**self.steps
        self.lib.gradient_sums(
            partial,
            self.total,
            self.g,
            partial.shape[0],
            2.0**-self.exponent,
            threads=-(-self.g * self.size // 256) * 256,
            group_size=256,
        )
        self.lib.adam(
            self.params,
            self.m,
            self.v,
            self.total,
            self.keep,
            lr,
            bc1,
            bc2,
            self.max_norm,
            threads=self.g * 1024,
            group_size=1024,
        )
        return bits

    @torch.no_grad()
    def export(self, nets, params: torch.Tensor | None = None):
        params = self.params if params is None else params
        for g, (net, dim) in enumerate(zip(nets, self.dimensions, strict=True)):
            w1a, w2b, w3b = self.views(params[g])
            net.hidden[0].weight.copy_(w1a[:dim, : self.h].T)
            net.affine.weight.copy_(w1a[:dim, self.h][None])
            net.hidden[0].bias.copy_(w1a[self.d, : self.h])
            net.hidden[2].weight.copy_(w2b[: self.h].T)
            net.hidden[2].bias.copy_(w2b[self.h])
            net.head.weight.copy_(w3b[: self.h].T)
            net.head.bias.copy_(w3b[self.h])


def pack(state: dict[str, np.ndarray], d: int, h: int, k: int) -> np.ndarray:
    """One plane's ProbabilityNet tensors (float arrays by state_dict name) in layout(d, h, k)."""
    lay = layout(d, h, k)
    block = np.zeros(lay["size"][0], np.float32)

    def view(name):
        offset, shape = lay[name]
        return block[offset : offset + shape[0] * shape[1]].reshape(shape)

    w1a, w2b, w3b = view("w1a"), view("w2b"), view("w3b")
    w0 = state["hidden.0.weight"]
    dim = w0.shape[1]
    w1a[:dim, :h] = w0.T
    w1a[:dim, h] = state["affine.weight"][0]
    w1a[d, :h] = state["hidden.0.bias"]
    w2b[:h] = state["hidden.2.weight"].T
    w2b[h] = state["hidden.2.bias"]
    w3b[:h] = state["head.weight"].T
    w3b[h] = state["head.bias"]
    return block


class FusedEvaluator:
    """Summed bits of many candidate weight sets over every training sample."""

    # Samples per launch, bounding the grid and the partial-sum buffer.
    LAUNCH = 1 << 27

    def __init__(self, x, y, a, hidden: int, mixtures: int, dimensions=None):
        """dimensions[g]: plane g's input count; its columns past that must be zero."""
        self.g, self.n, self.d = x.shape
        self.dims = torch.tensor(
            [self.d] * self.g if dimensions is None else list(dimensions),
            dtype=torch.int32,
            device=x.device,
        )
        self.h, self.k = hidden, mixtures
        self.lib = library(self.d, hidden, mixtures)
        self.x = x.to(torch.float16).contiguous()
        self.y = y.contiguous()
        self.a = a.contiguous()
        self.npad = -(-self.n // GROUP) * GROUP
        self._outputs = None
        self.outputs_key = None

    def bits(self, planes: list[int], blocks: np.ndarray) -> list[float]:
        """Total bits for candidate i: weights blocks[i] (pack layout) on plane planes[i]."""
        out: list[float] = []
        per = max(1, self.LAUNCH // self.npad)
        for start in range(0, len(planes), per):
            count = min(per, len(planes) - start)
            params = torch.from_numpy(np.ascontiguousarray(blocks[start : start + count]))
            plane = torch.tensor(planes[start : start + count], dtype=torch.int32)
            partial = torch.empty(count, self.npad // GROUP, device=self.x.device)
            self.lib.evaluate(
                params.to(self.x.device),
                plane.to(self.x.device),
                self.dims,
                self.x,
                self.y,
                self.a,
                partial,
                self.n,
                self.npad,
                threads=count * self.npad,
                group_size=GROUP,
            )
            out.extend(partial.cpu().double().sum(1).tolist())
        return out

    def activations(self, g: int, block: np.ndarray, key=None) -> None:
        """Keeps plane g's layer outputs under block (pack layout) for bits_from."""
        device = self.x.device
        if self._outputs is None:
            self._outputs = tuple(
                torch.empty(self.n, *width, device=device)
                for width in ((self.h,), (self.h,), (3 * self.k,), ())
            )
        params = torch.from_numpy(np.ascontiguousarray(block[None])).to(device)
        self.lib.activations(
            params,
            self.dims,
            self.x,
            *self._outputs,
            self.n,
            g,
            threads=self.npad,
            group_size=GROUP,
        )
        self.outputs_plane, self.outputs_key = g, key

    def bits_from(self, stage: int, blocks: np.ndarray) -> list[float]:
        """bits() for candidates on the activations' plane that equal their
        weights before stage 1 (second layer), 2 (head) or 3 (affine only)."""
        out: list[float] = []
        per = max(1, self.LAUNCH // self.npad)
        for start in range(0, len(blocks), per):
            count = min(per, len(blocks) - start)
            params = torch.from_numpy(np.ascontiguousarray(blocks[start : start + count]))
            partial = torch.empty(count, self.npad // GROUP, device=self.x.device)
            self.lib.evaluate_from(
                params.to(self.x.device),
                self.dims,
                self.x,
                self.y,
                self.a,
                *self._outputs,
                partial,
                self.n,
                self.npad,
                self.outputs_plane,
                stage,
                threads=count * self.npad,
                group_size=GROUP,
            )
            out.extend(partial.cpu().double().sum(1).tolist())
        return out


def _check():
    """Compare fused gradients with autograd on random data."""
    import neural_probability_probe as trainer

    torch.manual_seed(1)
    dev = "mps"
    dims = [39, 57, 76]
    n, batch = 5000, 3000
    for hidden in (16, 24, 48, 64):
        nets = []
        for d in dims:
            net = trainer.ProbabilityNet(d, hidden, 4, np.random.default_rng(d).normal(0, 0.3, d))
            with torch.no_grad():
                net.head.weight.normal_(0, 0.3)
            nets.append(net.to(dev))
        x = torch.zeros(3, n, max(dims), device=dev)
        for g, d in enumerate(dims):
            x[g, :, :d] = torch.randn(n, d, device=dev) * 0.5
        y = torch.randint(-30, 30, (3, n), device=dev).float()
        a = torch.randint(0, 256, (3, n), device=dev).float()
        y = torch.clamp(y, -a, 255 - a)
        idx = torch.randperm(n, device=dev)[:batch].to(torch.int32)
        fused = FusedGroupedTrainer(nets, dims, x, y, a, lr=0.01)
        g1, g2, g3, bits = fused.gradients(idx)
        grouped = trainer.GroupedNet(nets, dims).to(dev)
        il = idx.long()
        loss_rows = trainer.mixture_bits(*grouped(fused.x[:, il].float()), y[:, il], a[:, il])
        loss = loss_rows.mean(1).sum()
        loss.backward()
        h = hidden
        ref = {
            "w1": grouped.w1.grad,
            "affine": grouped.affine.grad[..., 0],
            "b1": grouped.b1.grad[:, 0],
            "w2": grouped.w2.grad,
            "b2": grouped.b2.grad[:, 0],
            "w3": grouped.w3.grad,
            "b3": grouped.b3.grad[:, 0],
        }
        got = {
            "w1": g1[:, : max(dims), :h],
            "affine": g1[:, : max(dims), h],
            "b1": g1[:, max(dims), :h],
            "w2": g2[:, :h],
            "b2": g2[:, h],
            "w3": g3[:, :h],
            "b3": g3[:, h],
        }
        reference = loss_rows.detach()
        error = float((bits - reference).abs().max())
        print(f"hidden {hidden}: bits max diff {error:.2e} of {float(reference.abs().mean()):.3f}")
        for name in ref:
            r, gg = ref[name].detach(), got[name]
            rel = float((r - gg).norm() / (r.norm() + 1e-12))
            print(f"  {name:7} rel err {rel:.2e}  norm {float(r.norm()):.3e}")


if __name__ == "__main__":
    _check()
