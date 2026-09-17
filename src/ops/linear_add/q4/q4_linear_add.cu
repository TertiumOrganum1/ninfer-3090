#include "ops/linear_add/q4/q4_linear_add_dispatch.h"

#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_mma.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

struct ResidualEpilogue {
    __device__ __forceinline__ void operator()(__nv_bfloat16* destination, float value) const {
        *destination = __float2bfloat16_rn(__bfloat162float(*destination) + value);
    }
};

struct GemvResidualEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16*, int row,
                                               float value) const {
        static_assert(!SplitOutput);
        ResidualEpilogue{}(out + row, value);
    }
};

struct KSplitResidualEpilogue {
    __nv_bfloat16* residual;
    std::int32_t tokens;

    template <int Capacity>
    __device__ __forceinline__ void store(int row, int col, float4 value) const {
        if (col < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row, value.x);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col) * 5120 + row + 8, value.z);
        }
        if (col + 1 < tokens) {
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row, value.y);
            ResidualEpilogue{}(residual + static_cast<std::int64_t>(col + 1) * 5120 + row + 8,
                               value.w);
        }
    }
};

using GemvR1W8 =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 6144 / 64, 1>;
using MmaR32C32  = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64  = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C128 = Q4RowSplitMmaGemmSchedule<64, 128, 64, 64, 32, 2, 1, Q4FragmentPipeline::Serial,
                                             Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
// The 64-row tiles the plain `linear` sweep found for this same geometry. Named rather than
// anonymous so bench/ops/dense_linear_add_schedule_bench.cu can time them against the routed
// choice without reimplementing the residual epilogue.
using MmaR64C48 = Q4RowSplitMmaGemmSchedule<64, 48, 64, 16, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C64 = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C80 = Q4RowSplitMmaGemmSchedule<64, 80, 64, 16, 40, 2, 1, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C96 = Q4RowSplitMmaGemmSchedule<64, 96, 64, 32, 16, 2, 1, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C112 = Q4RowSplitMmaGemmSchedule<64, 112, 64, 32, 16, 2, 1,
                                             Q4FragmentPipeline::Serial, Cache::cg, Cache::cg,
                                             Q4ScaleLoad::Pair32>;

template <int Capacity>
void launch_ksplit(const Tensor& x, const Weight& w, Tensor& residual, cudaStream_t stream) {
    using Geometry = Q4LinearGeometry<5120, 6144>;
    auto* output   = static_cast<__nv_bfloat16*>(residual.data);
    q4_ksplit_mma_kernel<Geometry, (Capacity + 7) / 8 * 8, Capacity, KSplitResidualEpilogue,
                         Q4KSplitIdentityRows, true>
        <<<5120 / Q4KSplitMmaSchedule::kRowsPerCta, Q4KSplitMmaSchedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), output,
            KSplitResidualEpilogue{output, x.ne[1]}, {}, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_add_gemv_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_gemv<GemvR1W8, GemvResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_ksplit4_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<4>(x, w, r, s);
}
void q4_linear_add_ksplit8_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<8>(x, w, r, s);
}
void q4_linear_add_ksplit16_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<16>(x, w, r, s);
}
void q4_linear_add_ksplit24_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<24>(x, w, r, s);
}
void q4_linear_add_ksplit32_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_ksplit<32>(x, w, r, s);
}
void q4_linear_add_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR32C32, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR32C64, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c48_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C48, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C64, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c80_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C80, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& r, cudaStream_t s) {
    launch_q4_mma<MmaR64C96, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c112_launch(const Tensor& x, const Weight& w, Tensor& r,
                                       cudaStream_t s) {
    launch_q4_mma<MmaR64C112, ResidualEpilogue>(x, w, r, s);
}
void q4_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& r,
                                       cudaStream_t s) {
    launch_q4_mma<MmaR64C128, ResidualEpilogue>(x, w, r, s);
}

Q4LinearAddLaunch select_q4_linear_add(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    if (rows != 5120 || k != 6144 || tokens <= 0) {
        throw std::invalid_argument("q4 linear_add: unsupported shape or token extent");
    }
    if (tokens == 1) return q4_linear_add_gemv_launch;
    if (tokens <= 4) return q4_linear_add_ksplit4_launch;
    if (tokens <= 8) return q4_linear_add_ksplit8_launch;
    if (tokens <= 16) return q4_linear_add_ksplit16_launch;
    if (tokens <= 24) return q4_linear_add_ksplit24_launch;
    if (tokens <= 32) return q4_linear_add_ksplit32_launch;
    if (tokens <= 96) return q4_linear_add_mma_r32_c32_launch;
    if (tokens <= 192) return q4_linear_add_mma_r32_c64_launch;
    return q4_linear_add_mma_r64_c128_launch;
}

} // namespace ninfer::ops::detail
