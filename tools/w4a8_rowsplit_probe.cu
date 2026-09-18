// W4A8 prefill GEMM that keeps the artifact's own RowSplit weight layout, on sm_86.
//
//   C[N,T] = W[N,K] (Q4, group-64 FP16 scales) x X[K,T] (s8, one scale per (token, group))
//   N = 34816, K = 5120, T = 512 -- the Qwen3.8-27B mlp/gate_up shape.
//
// w4a8_gemm_probe.cu found the fast structure: cp.async, packed nibbles in shared, and the scales
// as their own async plane (1415.2 us, 128.99 TOP/s, 2.37x the A16 route). It reached that by
// pre-permuting *both* operands into MMA-fragment order, which for the weights means a second copy
// of the matrix. Production cannot have that: AGENTS.md rules out runtime weight repacking, and a
// permuted copy of the 27B's two MLP matrices is ~9.7 GB on a 24 GB card.
//
// So this probe asks the question production actually has to answer: how much of that 2.37x
// survives if only the *activations* -- a transient this engine already owns and can write in any
// order -- are permuted, while the weights stay exactly as the artifact stores them?
//
// Three things change against the fragment-layout probe:
//
//   1. W is read from RowSplit, where one row's 64 codes for one group are 32 contiguous bytes.
//      That is still cp.async-able (two 16-byte copies per row), so the staging is unchanged; what
//      changes is the consumer, which now assembles its A fragments from four 2-byte reads per row
//      instead of one 16-byte read. The earlier probe measured that shared *instruction count* is
//      not the lever and shared *bytes* are -- and the bytes are identical, because both hold
//      packed nibbles.
//   2. The weight scales cannot be gathered by cp.async per group: for one group they are strided
//      by the row. But for one row they are contiguous *across* groups, so this reads them eight
//      groups at a time into a double-buffered ring, 16 bytes per row, committed in the same
//      cp.async group as the stage that first needs them. Still no synchronous global read in the
//      prefetch path, which was worth 2.10x -> 2.35x there.
//   3. Activation scales are per (token, group of 64), not per token. That is the production
//      quality contract -- a per-token absmax is set by whichever channel is largest and starves
//      the rest; w4a8_real_weight_probe.cu measured relative L2 rising from 1.2e-2 to 1.29e-1 with
//      outlier depth per token, against 9e-3 to 2.0e-2 per group -- and it costs a scale plane
//      read per stage plus a second multiply in the rescale.
//
// Reference points at this shape, all measured on the same card:
//   A16 route (q4_rowsplit_gemm_mma)                        3357.7 us   54.36 TFLOP/s
//   production q4a8_swiglu (register-staged, no repack)     ~1700 us   ~107 TOP/s
//   fragment-layout probe (both operands permuted)           1415.2 us  128.99 TOP/s
//
// Build:  nvcc -O3 -arch=sm_86 -allow-unsupported-compiler w4a8_rowsplit_probe.cu -o rowsplit_probe

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
            printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__);                  \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int N      = 34816;
constexpr int K      = 5120;
constexpr int T      = 512;
// Warp tile, swept: MT m-tiles of 16 rows and NT n-tiles of 8 tokens per warp, over a fixed 4x4
// warp grid. The accumulator is MT*NT*4 floats per thread and is the only register term big enough
// to move occupancy, which TODO.md's open entry names as the cause of the ~30%-of-ceiling rate.
#ifndef MT
#define MT 2
#endif
#ifndef NT
#define NT 4
#endif
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 4;
constexpr int BM     = WARPS_M * MT * 16;
constexpr int BN     = WARPS_N * NT * 8;
constexpr int BK     = 64; // one Q4 scale group
constexpr int GROUPS = K / BK;
constexpr int MTILES = BM / 16;
constexpr int NTILES = BN / 8;

// A row's 32 packed bytes are 8 shared words, so eight consecutive rows land on only four distinct
// bank groups and the A read is 4-way conflicted. Padding the row stride to 48 bytes (12 words,
// still 16-byte aligned for cp.async) spreads eight rows over eight bank groups.
#ifndef WROW
#define WROW 48
#endif
constexpr int WSTAGE  = BM * WROW;           // packed bytes, RowSplit order, padded rows
constexpr int XSTAGE  = NTILES * 32 * 16;    // 8192 s8 bytes, fragment order
constexpr int XSSTAGE = BN * 2;              // one FP16 activation scale per token, this group
constexpr int STAGE   = WSTAGE + XSTAGE + XSSTAGE;
constexpr int STAGES  = 2;

// Weight scales arrive eight groups at a time: 16 bytes is the widest cp.async, and for one row
// eight consecutive groups are exactly that.
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

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(addr), "l"(gmem));
}

// Two packed bytes hold four consecutive k as (low,high) nibble pairs. Codes are two's complement
// in the low four bits, so centring is a subtract of 8 per nibble. Returns the s8 word an A
// fragment register wants: k, k+1, k+2, k+3 in byte order.
__device__ __forceinline__ unsigned expand16(unsigned pair) {
    const unsigned lo = pair & 0x00ffu;
    const unsigned hi = (pair >> 8) & 0x00ffu;
    const unsigned spread = (lo & 0xfu) | ((lo & 0xf0u) << 4) | ((hi & 0xfu) << 16) |
                            ((hi & 0xf0u) << 20);
    return __vsub4(spread, 0x08080808u);
}

__device__ __forceinline__ unsigned lds16(const void* p) {
    unsigned r;
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(p));
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(r) : "r"(addr));
    return r;
}

__global__ __launch_bounds__(512) void w4a8_rowsplit(const unsigned char* __restrict__ w_codes,
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
    const int gid    = lane >> 2;
    const int tig    = lane & 3;
    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;

    const int row_block = blockIdx.x * BM;
    const int col_block = blockIdx.y * BN;

    const unsigned char* const w_blk = w_codes + static_cast<size_t>(row_block) * (K / 2);
    const char* const x_blk          = x_perm + static_cast<size_t>(blockIdx.y) * GROUPS * XSTAGE;
    const char* const xs_blk =
        reinterpret_cast<const char*>(x_scales) + static_cast<size_t>(blockIdx.y) * GROUPS * XSSTAGE;
    const char* const ws_blk = reinterpret_cast<const char*>(w_scales) +
                               static_cast<size_t>(row_block) * GROUPS * 2;

    auto issue = [&](int g, int buf) {
        char* const dst = s_base + buf * STAGE;
        // W: one row's group is 32 contiguous bytes in RowSplit, so two 16-byte copies per row.
        if (tid < BM * 2) {
            const int row  = tid >> 1;
            const int half = tid & 1;
            cp_async16(dst + row * WROW + half * 16,
                       w_blk + static_cast<size_t>(row) * (K / 2) + g * (BK / 2) + half * 16);
        }
        if (tid < XSTAGE / 16) {
            cp_async16(dst + WSTAGE + tid * 16, x_blk + static_cast<size_t>(g) * XSTAGE + tid * 16);
        }
        if (tid < XSSTAGE / 16) {
            cp_async16(dst + WSTAGE + XSTAGE + tid * 16,
                       xs_blk + static_cast<size_t>(g) * XSSTAGE + tid * 16);
        }
        // The scale ring rides the commit group of the stage that first reads it.
        if (g % RING_GROUPS == 0 && tid < BM) {
            char* const ring = s_ring + ((g / RING_GROUPS) % RING_BUFS) * RING_BYTES;
            cp_async16(ring + tid * RING_GROUPS * 2,
                       ws_blk + (static_cast<size_t>(tid) * GROUPS + g) * 2);
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
        const int buf = g % STAGES;
        if (g + STAGES - 1 < GROUPS) { issue(g + STAGES - 1, (g + STAGES - 1) % STAGES); }
        const int issued  = (g + STAGES < GROUPS) ? (g + STAGES) : GROUPS;
        const int allowed = issued - (g + 1);
        if (allowed >= 2) {
            asm volatile("cp.async.wait_group 2;");
        } else if (allowed == 1) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const char* const sa  = s_base + buf * STAGE;
        const char* const sb  = sa + WSTAGE;
        const __half* const sxs = reinterpret_cast<const __half*>(sb + XSTAGE);
        const __half* const ring =
            reinterpret_cast<const __half*>(s_ring + ((g / RING_GROUPS) % RING_BUFS) * RING_BYTES);

        unsigned af[MT][2][4];
        unsigned bf[NT][2][2];
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int r0 = (warp_m * MT + m) * 16 + gid;
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const int off = ks * 16 + tig * 2;
                af[m][ks][0]  = expand16(lds16(sa + r0 * WROW + off));
                af[m][ks][1]  = expand16(lds16(sa + (r0 + 8) * WROW + off));
                af[m][ks][2]  = expand16(lds16(sa + r0 * WROW + off + 8));
                af[m][ks][3]  = expand16(lds16(sa + (r0 + 8) * WROW + off + 8));
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const uint4 b = lds128(sb + ((warp_n * NT + n) * 32 + lane) * 16);
            bf[n][0][0]   = b.x;
            bf[n][0][1]   = b.y;
            bf[n][1][0]   = b.z;
            bf[n][1][1]   = b.w;
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
            const int sr    = (warp_m * MT + m) * 16 + gid;
            const float ws0 = __half2float(ring[sr * RING_GROUPS + (g % RING_GROUPS)]);
            const float ws1 = __half2float(ring[(sr + 8) * RING_GROUPS + (g % RING_GROUPS)]);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                int s[4] = {0, 0, 0, 0};
#pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    mma_s8(s[0], s[1], s[2], s[3], af[m][ks][0], af[m][ks][1], af[m][ks][2],
                           af[m][ks][3], bf[n][ks][0], bf[n][ks][1]);
                }
                const int c     = (warp_n * NT + n) * 8 + tig * 2;
                const float xa0 = __half2float(sxs[c]);
                const float xa1 = __half2float(sxs[c + 1]);
                acc[m][n][0]    = fmaf(static_cast<float>(s[0]), ws0 * xa0, acc[m][n][0]);
                acc[m][n][1]    = fmaf(static_cast<float>(s[1]), ws0 * xa1, acc[m][n][1]);
                acc[m][n][2]    = fmaf(static_cast<float>(s[2]), ws1 * xa0, acc[m][n][2]);
                acc[m][n][3]    = fmaf(static_cast<float>(s[3]), ws1 * xa1, acc[m][n][3]);
            }
        }
        // Second barrier: the next iteration's issue writes the buffer this one just read, and a
        // thread that has not finished reading it must not be overtaken. cp.async lands whenever it
        // lands, so wait_group alone does not order this.
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < MT; ++m) {
        const int r0 = row_block + (warp_m * MT + m) * 16 + gid;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            const int c0 = col_block + (warp_n * NT + n) * 8 + tig * 2;
            out[static_cast<size_t>(r0) * T + c0]           = __float2bfloat16(acc[m][n][0]);
            out[static_cast<size_t>(r0) * T + c0 + 1]       = __float2bfloat16(acc[m][n][1]);
            out[static_cast<size_t>(r0 + 8) * T + c0]       = __float2bfloat16(acc[m][n][2]);
            out[static_cast<size_t>(r0 + 8) * T + c0 + 1]   = __float2bfloat16(acc[m][n][3]);
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
    for (size_t i = 0; i < xs_count; ++i) hxs[i] = __float2half(0.0031f + 0.0004f * ((i % 5) / 5.0f));

    // The activations are this engine's own transient, so the quantizer writes them in fragment
    // order for free. The weights are used exactly as the artifact stores them.
    std::vector<signed char> hxp(x_count);
    for (int cb = 0; cb < T / BN; ++cb)
        for (int g = 0; g < GROUPS; ++g)
            for (int nt = 0; nt < NTILES; ++nt)
                for (int l = 0; l < 32; ++l) {
                    const int gid = l >> 2, tg = l & 3;
                    const int col = cb * BN + nt * 8 + gid;
                    signed char* dst =
                        &hxp[((static_cast<size_t>(cb) * GROUPS + g) * NTILES + nt) * 512 +
                             static_cast<size_t>(l) * 16];
                    for (int ks = 0; ks < 2; ++ks)
                        for (int hi = 0; hi < 2; ++hi)
                            for (int j = 0; j < 4; ++j)
                                dst[ks * 8 + hi * 4 + j] =
                                    hx[static_cast<size_t>(col) * K + g * BK + ks * 32 + tg * 4 +
                                       hi * 16 + j];
                }
    // Activation scales likewise: [col_block][group][token].
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
    CHECK(cudaMemcpy(dw, hw.data(), w_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dws, hws.data(), ws_count * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dx, hxp.data(), x_count, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dxs, hxsp.data(), xs_count * sizeof(__half), cudaMemcpyHostToDevice));

    dim3 grid(N / BM, T / BN);
    const size_t smem = STAGES * STAGE + RING_BUFS * RING_BYTES;
    CHECK(cudaFuncSetAttribute(w4a8_rowsplit, cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(smem)));
    int blocks_per_sm = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, w4a8_rowsplit, 512, smem));
    printf("GPU: %s   grid=(%d,%d) block=512  tile=%dx%d  warp=%dx%d  smem=%zu B  "
           "blocks/SM=%d (%d of 48 warps)\n",
           p.name, grid.x, grid.y, BM, BN, MT, NT, smem, blocks_per_sm, blocks_per_sm * 16);

    w4a8_rowsplit<<<grid, 512, smem>>>(static_cast<const unsigned char*>(dw),
                                       static_cast<const __half*>(dws), static_cast<const char*>(dx),
                                       static_cast<const __half*>(dxs),
                                       static_cast<__nv_bfloat16*>(dout));
    CHECK(cudaDeviceSynchronize());

    std::vector<__nv_bfloat16> hout(static_cast<size_t>(N) * T);
    CHECK(cudaMemcpy(hout.data(), dout, static_cast<size_t>(N) * T * sizeof(__nv_bfloat16),
                     cudaMemcpyDeviceToHost));
    double worst = 0.0;
    for (int s = 0; s < 64; ++s) {
        const int r = static_cast<int>(static_cast<size_t>(rand()) * 7919 % N);
        const int c = rand() % T;
        double ref  = 0.0;
        for (int g = 0; g < GROUPS; ++g) {
            long long dot = 0;
            for (int j = 0; j < BK; ++j) {
                const int k       = g * BK + j;
                const unsigned by = hw[static_cast<size_t>(r) * (K / 2) + k / 2];
                const int code =
                    static_cast<int>((k % 2 == 0) ? (by & 0xf) : ((by >> 4) & 0xf)) - 8;
                dot += static_cast<long long>(code) * hx[static_cast<size_t>(c) * K + k];
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
        printf("  aborting: kernel is wrong, timing would be meaningless\n");
        return 1;
    }

    const int reps = 12;
    std::vector<float> ms(reps);
    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int i = 0; i < reps; ++i) {
        CHECK(cudaMemsetAsync(dflush, i & 0xff, 256u << 20));
        CHECK(cudaEventRecord(a));
        w4a8_rowsplit<<<grid, 512, smem>>>(
            static_cast<const unsigned char*>(dw), static_cast<const __half*>(dws),
            static_cast<const char*>(dx), static_cast<const __half*>(dxs),
            static_cast<__nv_bfloat16*>(dout));
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        CHECK(cudaEventElapsedTime(&ms[i], a, b));
    }
    std::sort(ms.begin(), ms.end());
    const double median_us = ms[reps / 2] * 1000.0;
    const double ops       = 2.0 * N * K * T;
    printf("\n  %-40s %9.1f us   %6.2f TOP/s\n", "W4A8 RowSplit weights, permuted activations",
           median_us, ops / (median_us * 1e-6) / 1e12);
    printf("  %-40s %9.1f us   %6.2f TOP/s\n", "both operands permuted (probe)", 1415.2,
           ops / (1415.2e-6) / 1e12);
    printf("  %-40s %9.1f us   %6.2f TOP/s\n", "production, register-staged", 1700.9,
           ops / (1700.9e-6) / 1e12);
    printf("  %-40s %9.1f us   %6.2f TFLOP/s\n", "A16 route (measured)", 3357.696,
           ops / (3357.696e-6) / 1e12);
    printf("  %-40s %9.2fx  (vs A16)   %5.2fx (vs production)\n", "speedup", 3357.696 / median_us,
           1700.9 / median_us);
    return 0;
}
