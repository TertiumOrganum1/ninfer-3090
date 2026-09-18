#pragma once
//
// Integer-activation prefill GEMM over groupwise-int RowSplit weights, on sm_86.
//
//   D[n, t] = sum_k W[n, k] * X[k, t]
//
// The weight is read in the artifact's own RowSplit layout: for Q4 the code plane is row-major
// [n][k/2] with two codes per byte, for Q5 an 8-byte-per-group high plane carries each code's fifth
// bit, and both carry one FP16 scale per (row, group of 64). Activations are quantised to s8 with
// one scale per (token, group of 64), so an outlier channel can only spoil its own group -- a
// per-token absmax is set by whichever channel is largest and starves every other one.
//
// This is the schedule the mlp/gate_up and mlp/down routes measured, lifted out of them so the
// mixer projections can have it too: 128x128 output tile, one 64-wide scale group per iteration,
// register-staged prefetch one group ahead, and mma.m16n8k32.s32.s8.s8.s32 on the s8 tensor cores
// at about 4.7x the rate of the bf16 f32-accumulate MMA this hardware offers the A16 routes.
//
// Rescale is exact per group: the s32 dot product for one group is multiplied once by
// (weight scale * activation scale). Accumulation across groups is FP32.
//
// Epilogue decides what happens to the four accumulators a thread owns; it sees the row index
// within the launched range, so a caller can scatter one weight's rows into several destinations.

#include "core/tensor.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::rowsplit_a8 {

constexpr int kBM     = 128; // output rows per block
constexpr int kBN     = 128; // tokens per block
constexpr int kBK     = 64;  // exactly one scale group, so the rescale needs no partial bookkeeping
constexpr int kSRow   = kBK + 16; // padded: 16-byte aligned, 20 words apart so the eight rows a
                                  // warp touches land in eight distinct banks
constexpr int kThreads = 512;
constexpr int kGroup   = 64;

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A uint4 read through a char* is emitted as four 32-bit loads unless the compiler can prove the
// alignment, which it cannot through this pointer arithmetic.
__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(addr));
    return r;
}

// Spread the four low bits of `bits` into the low bit of four bytes: byte j becomes bit j.
__device__ __forceinline__ unsigned spread4(unsigned bits) {
    const unsigned t = bits & 0xfu;
    return ((t) | (t << 7) | (t << 14) | (t << 21)) & 0x01010101u;
}

// Four packed bytes hold eight codes as (low,high) nibble pairs. Codes are two's complement, so
// flipping bit 3 of every nibble and subtracting 8 reproduces the (n^8)-8 the A16 decode uses.
struct Q4Codec {
    static constexpr bool kHasHigh = false;
    __device__ static void unpack(unsigned packed, unsigned /*high*/, unsigned& w0, unsigned& w1) {
        packed ^= 0x88888888u;
        const unsigned mask = 0x0f0f0f0fu;
        const unsigned even = __vsub4(packed & mask, 0x08080808u);
        const unsigned odd  = __vsub4((packed >> 4) & mask, 0x08080808u);
        w0                  = __byte_perm(even, odd, 0x5140);
        w1                  = __byte_perm(even, odd, 0x7362);
    }
};

// Q5 adds the fifth bit from the high plane: a code decodes as ((low4 | hbit << 4) ^ 0x10) - 0x10
// over [-16,15], which int8 still holds exactly. The high bit for code i is bit i of the high byte
// covering codes 8i..8i+7, and it is consumed while unpacking, so it never occupies shared memory.
struct Q5Codec {
    static constexpr bool kHasHigh = true;
    __device__ static void unpack(unsigned packed, unsigned high, unsigned& w0, unsigned& w1) {
        const unsigned mask = 0x0f0f0f0fu;
        const unsigned even = packed & mask;
        const unsigned odd  = (packed >> 4) & mask;
        unsigned lo0        = __byte_perm(even, odd, 0x5140); // codes 0..3
        unsigned lo1        = __byte_perm(even, odd, 0x7362); // codes 4..7
        lo0 |= spread4(high) << 4;
        lo1 |= spread4(high >> 4) << 4;
        w0 = __vsub4(lo0 ^ 0x10101010u, 0x10101010u);
        w1 = __vsub4(lo1 ^ 0x10101010u, 0x10101010u);
    }
};

// One scale per (token, group of 64), written where the GEMM wants it.
template <std::int32_t kCols>
__global__ void quantize_activations(const __nv_bfloat16* __restrict__ x, std::int32_t tokens,
                                     std::int8_t* __restrict__ codes, __half* __restrict__ scales) {
    constexpr std::int32_t kGroups = kCols / kGroup;
    const std::int32_t token       = blockIdx.x;
    if (token >= tokens) { return; }
    for (std::int32_t g = threadIdx.x; g < kGroups; g += blockDim.x) {
        const __nv_bfloat16* src = x + static_cast<std::size_t>(token) * kCols + g * kGroup;
        float amax               = 0.0F;
        for (int j = 0; j < kGroup; ++j) {
            amax = fmaxf(amax, fabsf(__bfloat162float(src[j])));
        }
        amax                                                  = fmaxf(amax, 1.0e-20F);
        scales[static_cast<std::size_t>(token) * kGroups + g] = __float2half(amax / 127.0F);
        const float inv  = 127.0F / amax;
        std::int8_t* dst = codes + static_cast<std::size_t>(token) * kCols +
                           static_cast<std::size_t>(g) * kGroup;
        for (int j = 0; j < kGroup; ++j) {
            const float v = __bfloat162float(src[j]) * inv;
            dst[j]        = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
        }
    }
}

// Writes each result to one destination matrix, offsetting the row. A weight whose rows feed two
// destinations is launched once per contiguous row range.
struct StoreEpilogue {
    __nv_bfloat16* dst;
    std::int32_t dst_rows;
    std::int32_t dst_row_offset;
    __device__ void operator()(std::int32_t row_local, std::int32_t token, float value) const {
        dst[static_cast<std::size_t>(token) * dst_rows + (row_local + dst_row_offset)] =
            __float2bfloat16(value);
    }
};

// residual += W @ x. Each output element belongs to exactly one thread, so the read-add-write is
// safe without any ordering.
struct ResidualAddEpilogue {
    __nv_bfloat16* residual;
    std::int32_t rows;
    __device__ void operator()(std::int32_t row_local, std::int32_t token, float value) const {
        const std::size_t i = static_cast<std::size_t>(token) * rows + row_local;
        residual[i]         = __float2bfloat16(__bfloat162float(residual[i]) + value);
    }
};

// row_begin selects a contiguous range of the weight's rows; the epilogue sees the index within
// that range. Grid is (rows_in_range / kBM, tokens / kBN).
template <class Codec, std::int32_t kCols, class Epilogue>
__global__ __launch_bounds__(kThreads) void a8_mma_kernel(
    const std::uint8_t* __restrict__ w_codes, const std::uint8_t* __restrict__ w_high,
    const __half* __restrict__ w_scales, const std::int8_t* __restrict__ x_codes,
    const __half* __restrict__ x_scales, std::int32_t tokens, std::int32_t row_begin,
    Epilogue epilogue) {
    constexpr std::int32_t kGroups = kCols / kGroup;
    extern __shared__ char smem[];
    std::int8_t* const sa = reinterpret_cast<std::int8_t*>(smem);
    std::int8_t* const sb = sa + kBM * kSRow;
    __half* const sws     = reinterpret_cast<__half*>(sb + kBN * kSRow);
    __half* const sxs     = sws + kBM;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2; // 0..7, selects the row inside a 16-row tile
    const int tig    = lane & 3;  // 0..3, selects the k quarter
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;

    const int row_block = blockIdx.x * kBM;       // within the launched range
    const int col_block = blockIdx.y * kBN;

    const int ld_row = tid >> 2; // 0..127
    const int ld_q   = tid & 3;  // 0..3, sixteen k each
    const int w_row  = row_begin + row_block + ld_row;
    const int x_tok  = col_block + ld_row;

    // Register-staged prefetch: the reads for group g+1 issue before the MMAs for group g, so their
    // latency is covered by compute rather than stalling on the barrier.
    struct Stage {
        uint2 w;
        unsigned high;
        uint4 x;
        __half ws;
        __half xs;
    };
    auto load_stage = [&](int g, Stage& s) {
        s.w = *reinterpret_cast<const uint2*>(w_codes +
                                              static_cast<std::size_t>(w_row) * (kCols / 2) +
                                              static_cast<std::size_t>(g) * (kBK / 2) + ld_q * 8);
        if constexpr (Codec::kHasHigh) {
            s.high = *reinterpret_cast<const std::uint16_t*>(
                w_high + static_cast<std::size_t>(w_row) * (static_cast<std::size_t>(kGroups) * 8) +
                static_cast<std::size_t>(g) * 8 + ld_q * 2);
        } else {
            s.high = 0;
        }
        s.x = *reinterpret_cast<const uint4*>(x_codes + static_cast<std::size_t>(x_tok) * kCols +
                                              static_cast<std::size_t>(g) * kBK + ld_q * 16);
        if (tid < kBM) {
            s.ws = w_scales[static_cast<std::size_t>(row_begin + row_block + tid) * kGroups + g];
        }
        if (tid < kBN) {
            s.xs = x_scales[static_cast<std::size_t>(col_block + tid) * kGroups + g];
        }
    };
    // Shared rows hold k permuted: a thread's four MMA operand chunks for one row are the k
    // quarters [t*4, t*4+16, t*4+32, t*4+48), so storing them adjacent turns fragment assembly into
    // one 128-bit load per row.
    auto store_stage = [&](const Stage& s) {
        unsigned q[4];
        Codec::unpack(s.w.x, s.high & 0xffu, q[0], q[1]);
        Codec::unpack(s.w.y, (s.high >> 8) & 0xffu, q[2], q[3]);
        std::int8_t* wrow = sa + ld_row * kSRow + ld_q * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<unsigned*>(wrow + i * 16) = q[i];
        }
        const unsigned xw[4] = {s.x.x, s.x.y, s.x.z, s.x.w};
        std::int8_t* xrow    = sb + ld_row * kSRow + ld_q * 4;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<unsigned*>(xrow + i * 16) = xw[i];
        }
        if (tid < kBM) { sws[tid] = s.ws; }
        if (tid < kBN) { sxs[tid] = s.xs; }
    };

    float acc[2][4][4];
#pragma unroll
    for (int m = 0; m < 2; ++m)
#pragma unroll
        for (int n = 0; n < 4; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0F;

    Stage cur;
    load_stage(0, cur);
    store_stage(cur);
    __syncthreads();

    for (int g = 0; g < kGroups; ++g) {
        Stage next;
        if (g + 1 < kGroups) { load_stage(g + 1, next); }

        unsigned af[2][2][4];
        unsigned bf[4][2][2];
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int r0   = warp_m * 32 + m * 16 + gid;
            const uint4 lo = lds128(sa + r0 * kSRow + tig * 16);
            const uint4 hi = lds128(sa + (r0 + 8) * kSRow + tig * 16);
            af[m][0][0] = lo.x; af[m][0][1] = hi.x; af[m][0][2] = lo.y; af[m][0][3] = hi.y;
            af[m][1][0] = lo.z; af[m][1][1] = hi.z; af[m][1][2] = lo.w; af[m][1][3] = hi.w;
        }
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const uint4 b = lds128(sb + ((warp_n * 4 + n) * 8 + gid) * kSRow + tig * 16);
            bf[n][0][0] = b.x; bf[n][0][1] = b.y; bf[n][1][0] = b.z; bf[n][1][1] = b.w;
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int sr    = warp_m * 32 + m * 16 + gid;
            const float ws0 = __half2float(sws[sr]);
            const float ws1 = __half2float(sws[sr + 8]);
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
                const int c     = (warp_n * 4 + n) * 8 + tig * 2;
                const float xa0 = __half2float(sxs[c]);
                const float xa1 = __half2float(sxs[c + 1]);
                acc[m][n][0]    = fmaf(static_cast<float>(s[0]), ws0 * xa0, acc[m][n][0]);
                acc[m][n][1]    = fmaf(static_cast<float>(s[1]), ws0 * xa1, acc[m][n][1]);
                acc[m][n][2]    = fmaf(static_cast<float>(s[2]), ws1 * xa0, acc[m][n][2]);
                acc[m][n][3]    = fmaf(static_cast<float>(s[3]), ws1 * xa1, acc[m][n][3]);
            }
        }
        __syncthreads();
        if (g + 1 < kGroups) {
            store_stage(next);
            __syncthreads();
        }
    }

#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const int c0 = col_block + (warp_n * 4 + n) * 8 + tig * 2;
            const int r0 = row_block + warp_m * 32 + m * 16 + gid;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = r0 + half * 8;
                epilogue(row, c0, acc[m][n][half * 2]);
                epilogue(row, c0 + 1, acc[m][n][half * 2 + 1]);
            }
        }
    }
}

// Transient bytes the activation planes need for T in [min,max].
[[nodiscard]] inline std::size_t activation_workspace_bytes(std::int32_t input_rows,
                                                            std::int32_t max_tokens) {
    const std::size_t t      = static_cast<std::size_t>(max_tokens);
    const std::size_t groups = static_cast<std::size_t>(input_rows) / kGroup;
    return ((t * static_cast<std::size_t>(input_rows) + 255) / 256) * 256 +
           ((t * groups * sizeof(__half) + 255) / 256) * 256;
}

[[nodiscard]] inline bool tokens_supported(std::int32_t tokens) {
    return tokens >= kBN && tokens % kBN == 0;
}

[[nodiscard]] inline std::size_t shared_bytes() {
    return static_cast<std::size_t>(kBM) * kSRow + static_cast<std::size_t>(kBN) * kSRow +
           (kBM + kBN) * sizeof(__half);
}

} // namespace ninfer::ops::detail::rowsplit_a8
