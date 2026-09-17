#include "ops/linear/q8/q8_shapes.h"
#include "ops/linear/q8/q8_ksplit_launch.cuh"
#if !defined(NINFER_SM8X_COMPAT)
#include "ops/linear/q8/q8_ksplit_grouped_mma.cuh"
#endif

namespace ninfer::ops::detail {
namespace {
using Geometry = Q8N2048K16384;
using Access   = Q8KSplitScaleAccess;
using Stage    = Q8KSplitActivationStage;
// Four K warps on sm_8x (q8_ksplit_config.h): the sixteen- and eight-warp tiles from C16 and C48
// up do not fit its static shared memory.
template <int KWarps, int TileTokens, int MinBlocks, Access A, Cache AC, Cache WC, Stage S>
using Tile = Q8KSplitSm8xFourWarpSchedule<KWarps, TileTokens, MinBlocks, A, AC, WC, S>;
using C4   = Tile<16, 8, 1, Access::Direct, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C8   = Tile<16, 8, 1, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C16  = Tile<16, 16, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C24  = Tile<16, 24, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C32  = Tile<16, 32, 1, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C40  = Tile<16, 40, 1, Access::Shared, Cache::cg, Cache::cg, Stage::RuntimeActive>;
using C48  = Tile<8, 48, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C56  = Tile<8, 56, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;
using C64  = Tile<8, 64, 2, Access::Shared, Cache::cg, Cache::cg, Stage::ActiveOnly>;

#if defined(NINFER_SM8X_COMPAT)
// The grouped medium kernels stage 128 * KSplits * TileCols bytes of static shared activations:
// 81,920 for C80 and at least 49,152 for C96/C128, all past sm_8x's 49,152-byte cap. As the W8
// route did on this card, T=49..128 runs as 32-column capacity tiles plus a capacity or GEMV tail.
Q8Launch select_small(std::int32_t tokens);

void launch_composite(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    std::int32_t offset = 0;
    while (x.ne[1] - offset >= 32) {
        const Tensor x_slice = x.slice(1, offset, 32);
        Tensor out_slice     = out.slice(1, offset, 32);
        launch_q8_ksplit<Geometry, 32, C32>(x_slice, weight, out_slice, stream);
        offset += 32;
    }
    const std::int32_t tail = x.ne[1] - offset;
    if (tail > 0) {
        const Tensor x_slice = x.slice(1, offset, tail);
        Tensor out_slice     = out.slice(1, offset, tail);
        select_small(tail)(x_slice, weight, out_slice, stream);
    }
}
#else
template <int Capacity, int KWarps, int TokenGroups>
void launch_grouped(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.padded_shape[1] != Geometry::kInputRows) {
        throw std::invalid_argument(
            "q8 grouped K-split: padded K differs from registered geometry");
    }
    const Q8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    q8_ksplit_grouped_mma_kernel<Geometry::kInputRows, Capacity, KWarps, TokenGroups, 1>
        <<<Geometry::kOutputRows / 16, KWarps * TokenGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
#endif

Q8Launch select_small(std::int32_t tokens) {
    if (tokens == 1) return launch_q8_gemv_n2048_k16384;
    if (tokens <= 4) return launch_q8_ksplit<Geometry, 4, C4>;
    if (tokens <= 8) return launch_q8_ksplit<Geometry, 8, C8>;
    if (tokens <= 16) return launch_q8_ksplit<Geometry, 16, C16>;
    if (tokens <= 24) return launch_q8_ksplit<Geometry, 24, C24>;
    if (tokens <= 32) return launch_q8_ksplit<Geometry, 32, C32>;
    if (tokens <= 40) return launch_q8_ksplit<Geometry, 40, C40>;
    return launch_q8_ksplit<Geometry, 48, C48>;
}
} // namespace

Q8Launch select_q8_n2048_k16384(std::int32_t tokens) {
    if (tokens <= 48) return select_small(tokens);
#if defined(NINFER_SM8X_COMPAT)
    if (tokens <= 128) return launch_composite;
#else
    if (tokens <= 56) return launch_q8_ksplit<Geometry, 56, C56>;
    if (tokens <= 64) return launch_q8_ksplit<Geometry, 64, C64>;
    if (tokens <= 80) return launch_grouped<80, 8, 2>;
    if (tokens <= 96) return launch_grouped<96, 4, 6>;
    if (tokens <= 128) return launch_grouped<128, 4, 8>;
#endif

    // Broad throughput regions; each selected MMA handles its own complete and partial tiles.
    if (tokens <= 384) return launch_q8_mma_r32_c64;
    if (tokens <= 480) return launch_q8_mma_r32_c96;
    if (tokens <= 640) return launch_q8_mma_r32_c128;
    if (tokens <= 704) return launch_q8_mma_r48_c64;
    if (tokens <= 960) return launch_q8_mma_r64_c96;
    if (tokens <= 1344) return launch_q8_mma_r128_c64;
    if (tokens <= 1680) return launch_q8_mma_r128_c80;
    if (tokens <= 2016) return launch_q8_mma_r64_c96;
    if (tokens <= 2112) return launch_q8_mma_r96_c96;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
