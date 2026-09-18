// A Marlin-class W4A8 mainloop over a permuted weight layout, on sm_86.
//
//   C[N,T] = W[N,K] (Q4, group-64 FP16 scales) x X[K,T] (s8, one scale per (token, group))
//   N = 34816, K = 5120 -- the Qwen3.8-27B mlp/gate_up shape.
//
// This is the gate for the whole layout project. The shipped kernel reaches 117 TOP/s at T=1024;
// cuBLAS's int8 GEMM reaches 237.5 on the same card and shape while solving an easier problem
// (tools/int8_gemm_reference_probe.cu). Every knob the shipped structure has is spent -- token tile
// +39%, occupancy +6%, pipeline depth +3.9%, grid swizzle +0.5%, L2 evict-first -0.8%, interleaved
// loads -1.9% (tools/w4a8_rowsplit_probe.cu, which stays as the baseline oracle). What is left is
// structural, and this probe builds all of it at once, because measuring the pieces separately is
// exactly what produced those noise-sized numbers:
//
//   1. the weight permuted so one 8-byte shared load is a lane's whole A fragment for a row,
//   2. LOP3 dequant whose natural output order *is* the MMA's required order,
//   3. a cp.async ring of STAGES buffers with one __syncthreads per stage rather than two per group,
//   4. packed nibbles in shared, which is what frees the shared memory the ring and a wide tile need.
//
// **The tile is the other half of the prize.** Per block a tile streams BM*K/2 bytes of weight and
// BN*K of activations, so over the whole GEMM W traffic is N*T*K/(2*BN) and X traffic is N*T*K/BM.
// At the shipped 64x512, X is 94% of the bytes -- 2,852 MB against 178 MB -- because each of the 544
// row blocks re-reads the activation tile. BM is therefore the dominant lever and it is currently
// pinned low by the registers operand assembly costs:
//
//   BM x BN     W        X        total
//    64 x 512   178 MB   2,852 MB 3,030 MB   (shipped)
//   128 x 256   356 MB   1,426 MB 1,782 MB
//   256 x 256   356 MB     713 MB 1,069 MB
//   256 x 512   178 MB     713 MB   891 MB
//
// ---------------------------------------------------------------------------------------------
// RESULT, 2026-09-18: this did not pay, and the decomposition says why.
//
// Best configuration (128x256, 3-4 stages, 512 threads, permuted layout, LOP3 dequant, one barrier
// per stage): 2,966 us / 123.3 TOP/s against the shipped kernel's 3,113 us / 117.3 -- **1.05x**,
// against a gate of 1,900 us / 192 TOP/s. The layout change is therefore abandoned; see TODO.md.
//
// Ablating the best configuration:
//
//   full kernel                          2,966 us
//   without the per-group rescale        2,744 us   (rescale ~12%)
//   streaming only, no MMAs, no reads    1,376 us   (was 1,723 at the shipped 64x512 tile)
//
// The parts are additive: 1,376 streaming + ~1,230 compute + ~370 rescale. **Nothing overlaps.**
// And the compute component is already at the hardware floor: this GEMM is 44.6M m16n8k32 MMAs,
// which at Ampere's 1,024 int8 MAC/SM/cycle is ~1.28 ms of pure issue across 82 SMs, so our MMA
// stream runs at ~93% of the card's peak int8 rate. cuBLAS finishes everything in 1,532 us, i.e.
// roughly max(streaming, MMA) rather than their sum.
//
// So the entire remaining gap is the overlap of streaming with compute, and none of the things
// Marlin's structure brings moved it: the permuted layout and single-instruction fragment loads
// (1.01x), the ring depth at 3 and 4 stages, one barrier per stage instead of two (+5%), or the
// tile that halves the streamed bytes (the floor fell 1,723 -> 1,376, the total did not follow).
// What that leaves -- and what this probe cannot settle without `ncu` counters, which need
// elevation on this box -- is the hypothesis that both phases contend for the same LSU/MIO pipe,
// in which case the lever is shared-read volume per MMA rather than anything structural.
// ---------------------------------------------------------------------------------------------

// Sweep with -DBM_ROWS -DBN_TOKENS -DSTAGES_N -DTHREADS_N -DWARPS_M_N -DABLATE.
//
// Build:
//   nvcc -O3 -arch=sm_86 -allow-unsupported-compiler w4a8_marlin_probe.cu -o w4a8_marlin_probe

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        cudaError_t e = (x);                                                                       \
        if (e != cudaSuccess) {                                                                    \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);                 \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int N = 34816;
constexpr int K = 5120;
#ifndef T_TOKENS
#define T_TOKENS 1024
#endif
constexpr int T = T_TOKENS;

constexpr int GROUP  = 64; // one FP16 scale per 64 codes; the rescale unit
constexpr int GROUPS = K / GROUP;

#ifndef BM_ROWS
#define BM_ROWS 128
#endif
#ifndef BN_TOKENS
#define BN_TOKENS 256
#endif
#ifndef THREADS_N
#define THREADS_N 512
#endif
#ifndef WARPS_M_N
#define WARPS_M_N 4
#endif
#ifndef STAGES_N
#define STAGES_N 3
#endif
// 0 = full kernel, 1 = MMAs and loads but no per-group rescale (int32 accumulate, converted once),
// 2 = streaming only (no MMAs, no shared reads): the G0a floor.
#ifndef ABLATE
#define ABLATE 0
#endif

constexpr int BM      = BM_ROWS;
constexpr int BN      = BN_TOKENS;
constexpr int THREADS = THREADS_N;
constexpr int WARPS   = THREADS / 32;
constexpr int WARPS_M = WARPS_M_N;
constexpr int WARPS_N = WARPS / WARPS_M;
constexpr int MT      = BM / (WARPS_M * 16); // 16-row m-tiles per warp
constexpr int NT      = BN / (WARPS_N * 8);  // 8-token n-tiles per warp
constexpr int STAGES  = STAGES_N;

static_assert(BM % (WARPS_M * 16) == 0, "BM must divide into 16-row tiles per warp row");
static_assert(BN % (WARPS_N * 8) == 0, "BN must divide into 8-token tiles per warp column");

// A row's 32 packed bytes are 8 shared words, so eight consecutive rows would land on only four
// bank groups. 48 keeps the cp.async destination 16-byte aligned and spreads eight rows over eight.
constexpr int WROW    = 48;
constexpr int WSTAGE  = BM * WROW;
constexpr int XSTAGE  = (BN / 8) * 32 * 16; // fragment order: one contiguous run per n-tile
constexpr int XSSTAGE = BN * 2;
constexpr int STAGE   = WSTAGE + XSTAGE + XSSTAGE;

constexpr int RING_GROUPS = 8;
constexpr int RING_BYTES  = BM * RING_GROUPS * 2;
constexpr int RING_BUFS   = 2;

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 lds128(const void* p) {
    uint4 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "r"(addr));
    return r;
}

__device__ __forceinline__ uint2 lds64(const void* p) {
    uint2 r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "r"(addr));
    return r;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(gmem));
}

// The whole point of the permutation: a packed word's low nibbles are already one MMA A register's
// four codes, and its high nibbles the next one's. Two masks and two vsub4 per word -- four
// instructions for eight codes, against ten for the byte-shuffling the RowSplit order forces.
__device__ __forceinline__ unsigned decode_lo(unsigned w) {
    return __vsub4(w & 0x0f0f0f0fu, 0x08080808u);
}
__device__ __forceinline__ unsigned decode_hi(unsigned w) {
    return __vsub4((w >> 4) & 0x0f0f0f0fu, 0x08080808u);
}

// One scale per (token, group of 64), written in the fragment order the GEMM reads.
__global__ void quantize_activations(const __nv_bfloat16* __restrict__ x, std::int8_t* codes,
                                     __half* scales) {
    const int token  = blockIdx.x;
    const int cb     = token / BN;
    const int tok_in = token % BN;
    const int nt     = tok_in / 8;
    const int gid    = tok_in % 8;
    for (int g = threadIdx.x; g < GROUPS; g += blockDim.x) {
        const __nv_bfloat16* src = x + static_cast<size_t>(token) * K + g * GROUP;
        float amax               = 0.0f;
        for (int j = 0; j < GROUP; ++j) { amax = fmaxf(amax, fabsf(__bfloat162float(src[j]))); }
        amax                                                        = fmaxf(amax, 1e-20f);
        scales[(static_cast<size_t>(cb) * GROUPS + g) * BN + tok_in] = __float2half(amax / 127.0f);
        const float inv  = 127.0f / amax;
        std::int8_t* tile =
            codes + ((static_cast<size_t>(cb) * GROUPS + g) * (BN / 8) + nt) * (32 * 16);
        for (int ks = 0; ks < 2; ++ks) {
            for (int hi = 0; hi < 2; ++hi) {
                for (int tig = 0; tig < 4; ++tig) {
                    std::int8_t quad[4];
                    for (int j = 0; j < 4; ++j) {
                        const float v = __bfloat162float(src[ks * 32 + hi * 16 + tig * 4 + j]) * inv;
                        quad[j] = static_cast<std::int8_t>(max(-127, min(127, __float2int_rn(v))));
                    }
                    *reinterpret_cast<unsigned*>(tile + (gid * 4 + tig) * 16 + ks * 8 + hi * 4) =
                        *reinterpret_cast<const unsigned*>(quad);
                }
            }
        }
    }
}

__global__ __launch_bounds__(THREADS) void w4a8_marlin(const unsigned char* __restrict__ w_perm,
                                                       const __half* __restrict__ w_scales,
                                                       const char* __restrict__ x_perm,
                                                       const __half* __restrict__ x_scales,
                                                       __nv_bfloat16* __restrict__ out) {
    extern __shared__ char smem[];
    char* const s_base = smem;
    char* const s_ring = smem + STAGES * STAGE;

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int gid    = lane >> 2; // row within a 16-row tile
    const int tig    = lane & 3;  // which quarter of the group this lane owns
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int row_block = blockIdx.x * BM;
    const int col_block = blockIdx.y;

    const unsigned char* const w_blk = w_perm + static_cast<size_t>(row_block) * (K / 2);
    const char* const x_blk = x_perm + static_cast<size_t>(col_block) * GROUPS * XSTAGE;
    const char* const xs_blk =
        reinterpret_cast<const char*>(x_scales) + static_cast<size_t>(col_block) * GROUPS * XSSTAGE;
    const char* const ws_blk =
        reinterpret_cast<const char*>(w_scales) + static_cast<size_t>(row_block) * GROUPS * 2;

    auto issue = [&](int g, int buf) {
        char* const dst = s_base + buf * STAGE;
        // W stays packed in shared: half the bytes of the unpacked staging the shipped kernel uses,
        // which is what leaves room for the ring and the wider tile.
#pragma unroll
        for (int c = tid; c < BM * 2; c += THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            cp_async16(dst + row * WROW + half * 16,
                       w_blk + static_cast<size_t>(row) * (K / 2) + g * (GROUP / 2) + half * 16);
        }
#pragma unroll
        for (int c = tid; c < XSTAGE / 16; c += THREADS) {
            cp_async16(dst + WSTAGE + c * 16, x_blk + static_cast<size_t>(g) * XSTAGE + c * 16);
        }
#pragma unroll
        for (int c = tid; c < XSSTAGE / 16; c += THREADS) {
            cp_async16(dst + WSTAGE + XSTAGE + c * 16,
                       xs_blk + static_cast<size_t>(g) * XSSTAGE + c * 16);
        }
        if (g % RING_GROUPS == 0) {
#pragma unroll
            for (int r = tid; r < BM; r += THREADS) {
                cp_async16(s_ring + ((g / RING_GROUPS) % RING_BUFS) * RING_BYTES +
                               r * RING_GROUPS * 2,
                           ws_blk + (static_cast<size_t>(r) * GROUPS + g) * 2);
            }
        }
        asm volatile("cp.async.commit_group;");
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[m][n][j] = 0.0f;

#pragma unroll
    for (int i = 0; i < STAGES - 1; ++i) {
        if (i < GROUPS) { issue(i, i); }
    }

    for (int g = 0; g < GROUPS; ++g) {
        // One barrier per stage, and the order is the whole point. Wait for this stage's data, then
        // barrier once -- which both publishes it and proves everyone has finished reading the slot
        // the next issue is about to overwrite -- then issue and compute with nothing between them,
        // so the copies for stage g+STAGES-1 fly while stage g's MMAs run. The shipped kernel pays
        // two barriers per group and serialises the two phases.
        const int outstanding = (g + STAGES - 1 < GROUPS) ? (STAGES - 2) : 0;
        if (outstanding >= 3) {
            asm volatile("cp.async.wait_group 3;");
        } else if (outstanding == 2) {
            asm volatile("cp.async.wait_group 2;");
        } else if (outstanding == 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
        if (g + STAGES - 1 < GROUPS) { issue(g + STAGES - 1, (g + STAGES - 1) % STAGES); }

        const char* const sa    = s_base + (g % STAGES) * STAGE;
        const char* const sb    = sa + WSTAGE;
        const __half* const sxs = reinterpret_cast<const __half*>(sb + XSTAGE);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / RING_GROUPS) % RING_BUFS) * RING_BYTES);

#if ABLATE != 2
        unsigned af[MT][2][4];
        unsigned bf[NT][2][2];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = (warp_m * MT + m) * 16 + gid;
            // One 8-byte load per row is this lane's whole fragment for both k-halves.
            const uint2 w0 = lds64(sa + r0 * WROW + tig * 8);
            const uint2 w1 = lds64(sa + (r0 + 8) * WROW + tig * 8);
            af[m][0][0] = decode_lo(w0.x); af[m][0][1] = decode_lo(w1.x);
            af[m][0][2] = decode_hi(w0.x); af[m][0][3] = decode_hi(w1.x);
            af[m][1][0] = decode_lo(w0.y); af[m][1][1] = decode_lo(w1.y);
            af[m][1][2] = decode_hi(w0.y); af[m][1][3] = decode_hi(w1.y);
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const uint4 b = lds128(sb + ((warp_n * NT + n) * 32 + lane) * 16);
            bf[n][0][0] = b.x; bf[n][0][1] = b.y; bf[n][1][0] = b.z; bf[n][1][1] = b.w;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int sr    = (warp_m * MT + m) * 16 + gid;
#if ABLATE != 1
            const float ws0 = __half2float(ring[sr * RING_GROUPS + (g % RING_GROUPS)]);
            const float ws1 = __half2float(ring[(sr + 8) * RING_GROUPS + (g % RING_GROUPS)]);
#endif
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
#if ABLATE == 1
                acc[m][n][0] += static_cast<float>(s[0]);
                acc[m][n][1] += static_cast<float>(s[1]);
                acc[m][n][2] += static_cast<float>(s[2]);
                acc[m][n][3] += static_cast<float>(s[3]);
#else
                const int c     = (warp_n * NT + n) * 8 + tig * 2;
                const float xa0 = __half2float(sxs[c]);
                const float xa1 = __half2float(sxs[c + 1]);
                acc[m][n][0]    = fmaf(static_cast<float>(s[0]), ws0 * xa0, acc[m][n][0]);
                acc[m][n][1]    = fmaf(static_cast<float>(s[1]), ws0 * xa1, acc[m][n][1]);
                acc[m][n][2]    = fmaf(static_cast<float>(s[2]), ws1 * xa0, acc[m][n][2]);
                acc[m][n][3]    = fmaf(static_cast<float>(s[3]), ws1 * xa1, acc[m][n][3]);
#endif
            }
        }
#else
        // Streaming floor: keep every copy, barrier and loop, drop the reads and the MMAs.
        if (tid == 0x7fffffff) { acc[0][0][0] += __half2float(sxs[0]) + __half2float(ring[0]); }
#endif
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = row_block + (warp_m * MT + m) * 16 + gid;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int c0 = col_block * BN + (warp_n * NT + n) * 8 + tig * 2;
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = r0 + half * 8;
                out[static_cast<size_t>(row) * T + c0]     = __float2bfloat16(acc[m][n][half * 2]);
                out[static_cast<size_t>(row) * T + c0 + 1] = __float2bfloat16(acc[m][n][half * 2 + 1]);
            }
        }
    }
}

int main() {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, 0));

    const size_t w_bytes  = static_cast<size_t>(N) * K / 2;
    const size_t ws_count = static_cast<size_t>(N) * GROUPS;
    const size_t x_count  = static_cast<size_t>(T) * K;
    const size_t xs_count = static_cast<size_t>(T) * GROUPS;

    std::vector<unsigned char> hw(w_bytes);
    std::vector<__half> hws(ws_count);
    std::vector<signed char> hx(x_count);
    std::vector<__half> hxs(xs_count);
    srand(1234);
    for (size_t i = 0; i < w_bytes; ++i) hw[i] = static_cast<unsigned char>(rand() & 0xff);
    for (size_t i = 0; i < ws_count; ++i) hws[i] = __float2half(0.002f + 0.001f * ((i % 7) / 7.0f));
    for (size_t i = 0; i < x_count; ++i) hx[i] = static_cast<signed char>((rand() % 255) - 127);
    for (size_t i = 0; i < xs_count; ++i)
        hxs[i] = __float2half(0.0031f + 0.0004f * ((i % 5) / 5.0f));

    // The permutation, exactly as the load-time device kernel would apply it: within one row and one
    // group of 64, never across rows, so row slicing and the scale plane are untouched. Lane `tig`
    // owns the 8 bytes at `tig*8`; its first word's low nibbles are k = tig*4 + 0..3 (the MMA's a0)
    // and its high nibbles k = 16 + tig*4 + 0..3 (a2); the second word carries the k+32 half.
    const auto code_at = [&](int row, int k) {
        const unsigned char byte = hw[static_cast<size_t>(row) * (K / 2) + k / 2];
        return static_cast<int>((k % 2 == 0) ? (byte & 0xf) : (byte >> 4));
    };
    std::vector<unsigned char> hwp(w_bytes);
    for (int row = 0; row < N; ++row) {
        for (int g = 0; g < GROUPS; ++g) {
            unsigned char* dst = &hwp[static_cast<size_t>(row) * (K / 2) + g * (GROUP / 2)];
            for (int t4 = 0; t4 < 4; ++t4) {
                for (int word = 0; word < 2; ++word) {
                    for (int j = 0; j < 4; ++j) {
                        const int k_lo = g * GROUP + word * 32 + t4 * 4 + j;
                        const int k_hi = k_lo + 16;
                        dst[t4 * 8 + word * 4 + j] =
                            static_cast<unsigned char>(code_at(row, k_lo) | (code_at(row, k_hi) << 4));
                    }
                }
            }
        }
    }

    // Activations: the quantiser owns this buffer, so it writes fragment order directly.
    std::vector<signed char> hxp(x_count);
    for (int cb = 0; cb < T / BN; ++cb)
        for (int g = 0; g < GROUPS; ++g)
            for (int nt = 0; nt < BN / 8; ++nt)
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    const int col = cb * BN + nt * 8 + gid;
                    signed char* dst =
                        &hxp[((static_cast<size_t>(cb) * GROUPS + g) * (BN / 8) + nt) * 512 +
                             static_cast<size_t>(l) * 16];
                    for (int ks = 0; ks < 2; ++ks)
                        for (int hi = 0; hi < 2; ++hi)
                            for (int j = 0; j < 4; ++j)
                                dst[ks * 8 + hi * 4 + j] =
                                    hx[static_cast<size_t>(col) * K + g * GROUP + ks * 32 +
                                       tg * 4 + hi * 16 + j];
                }
    std::vector<__half> hxsp(xs_count);
    for (int cb = 0; cb < T / BN; ++cb)
        for (int g = 0; g < GROUPS; ++g)
            for (int t = 0; t < BN; ++t)
                hxsp[(static_cast<size_t>(cb) * GROUPS + g) * BN + t] =
                    hxs[static_cast<size_t>(cb * BN + t) * GROUPS + g];

    void *dw, *dws, *dx, *dxs, *dout, *dflush;
    CHECK(cudaMalloc(&dw, w_bytes));
    CHECK(cudaMalloc(&dws, ws_count * sizeof(__half)));
    CHECK(cudaMalloc(&dx, x_count));
    CHECK(cudaMalloc(&dxs, xs_count * sizeof(__half)));
    CHECK(cudaMalloc(&dout, static_cast<size_t>(N) * T * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dflush, 256u << 20));
    CHECK(cudaMemcpy(dw, hwp.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dws, hws.data(), ws_count * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hxp.data(), x_count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxs, hxsp.data(), xs_count * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 grid(N / BM, T / BN);
    const size_t smem = STAGES * STAGE + RING_BUFS * RING_BYTES;
    CHECK(cudaFuncSetAttribute(w4a8_marlin, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem)));
    int blocks_per_sm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, w4a8_marlin, THREADS, smem));
    printf("GPU: %s  tile=%dx%d warps=%dx%d (MT=%d NT=%d) threads=%d stages=%d\n", p.name, BM, BN,
           WARPS_M, WARPS_N, MT, NT, THREADS, STAGES);
    printf("      smem=%zu B  blocks/SM=%d (%d of 48 warps)  grid=(%d,%d)\n", smem, blocks_per_sm,
           blocks_per_sm * WARPS, grid.x, grid.y);
    if (smem > 99u * 1024) {
        printf("  shared memory over the 99 KiB budget; skipping\n");
        return 2;
    }
    if (blocks_per_sm == 0) {
        printf("  configuration does not fit; skipping\n");
        return 2;
    }

    w4a8_marlin<<<grid, THREADS, smem>>>(static_cast<const unsigned char*>(dw),
                                         static_cast<const __half*>(dws),
                                         static_cast<const char*>(dx),
                                         static_cast<const __half*>(dxs),
                                         static_cast<__nv_bfloat16*>(dout));
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hout(static_cast<size_t>(N) * T);
    CHECK(cudaMemcpy(hout.data(), dout, static_cast<size_t>(N) * T * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    double worst = 0.0;
#if ABLATE == 0
    for (int s = 0; s < 64; ++s) {
        const int r = static_cast<int>(static_cast<size_t>(rand()) * 7919 % N);
        const int c = rand() % T;
        double ref  = 0.0;
        for (int g = 0; g < GROUPS; ++g) {
            long long dot = 0;
            for (int j = 0; j < GROUP; ++j) {
                const int k = g * GROUP + j;
                dot += static_cast<long long>(code_at(r, k) - 8) * hx[static_cast<size_t>(c) * K + k];
            }
            ref += static_cast<double>(dot) *
                   static_cast<double>(__half2float(hws[static_cast<size_t>(r) * GROUPS + g])) *
                   static_cast<double>(__half2float(hxs[static_cast<size_t>(c) * GROUPS + g]));
        }
        const double got = static_cast<double>(__bfloat162float(hout[static_cast<size_t>(r) * T + c]));
        worst = std::max(worst, std::abs(got - ref) / std::max(std::abs(ref), 1e-6));
    }
    printf("correctness: 64 sampled outputs, worst relative error %.3e  (%s)\n", worst,
           worst < 5e-3 ? "OK" : "MISMATCH");
    if (worst >= 5e-3) {
        printf("  aborting: a fast wrong kernel is the hazard this probe exists to catch\n");
        return 1;
    }
#else
    printf("correctness: skipped (ablation)\n");
#endif

    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a8_marlin<<<grid, THREADS, smem>>>(static_cast<const unsigned char*>(dw),
                                             static_cast<const __half*>(dws),
                                             static_cast<const char*>(dx),
                                             static_cast<const __half*>(dxs),
                                             static_cast<__nv_bfloat16*>(dout));
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double us  = ms[reps / 2] * 1000.0;
    const double ops = 2.0 * N * K * T;
    printf("\n  %-44s %9.1f us   %6.2f TOP/s\n", "W4A8 permuted layout, Marlin-class mainloop", us,
           ops / (us * 1e-6) / 1e12);
    printf("  %-44s %9.1f us   %6.2f TOP/s\n", "shipped kernel (64x512, 2 stages)", 3113.0,
           ops / (3113.0e-6) / 1e12);
    printf("  %-44s %9.1f us   %6.2f TOP/s\n", "cuBLAS int8 (easier problem, 2x the bytes)", 1531.9,
           ops / (1531.9e-6) / 1e12);
    printf("  %-44s %9.2fx\n", "vs shipped", 3113.0 / us);
    return 0;
}
