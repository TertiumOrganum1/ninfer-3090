// Route-boundary sweep for the dense `linear_add` tables the 2026-09-17 catch-up brought in
// (schedule_sweep.cuh). Companion to q5_linear_add_schedule_bench.cu, which covers the Q5 tables.
//
// Two tables, both new here and both measured on an RTX 5090 only:
//
//   Q8, rows = 5120:  {1,64} SplitKMmaCapacity   {65,kAny} GroupedSplitK
//   Q4, rows = 5120, k = 6144:
//       1 gemv / <=4,8,16,24,32 K-split capacities / <=96 r32_c32 / <=192 r32_c64 / r64_c128
//
// The Q8 table is two entries wide where the same Op's 2048-row tables are thirty-three, which is
// the shape of a table nobody has swept rather than one that came out simple. The Q4 table is the
// same geometry the plain `linear` Q4 sweep found 1.7x on, using the same 32-row tiles that lost
// there, so it is the first thing to check.
//
// Domains: the Q8 decode kernel is a one-column route and the exact-T split-K is registered
// through 48 columns; both are capped rather than left to fault.

#include "ninfer/ops/linear_add.h"

#include "ops/linear_add/q4/q4_linear_add_dispatch.h"
#include "ops/linear_add/q8/q8_linear_add_kernels.h"
#include "ops/linear_add/q8/q8_linear_add_plan.h"
#include "quantized_weight.cuh"
#include "schedule_sweep.cuh"

#include <cstdio>
#include <string>
#include <vector>

namespace {

using ninfer::DType;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
namespace detail = ninfer::ops::detail;

constexpr std::int32_t kRows = 5120;

// The Q8 MMA launches take a `full` flag the dispatcher computes from the problem; a bench that
// guessed it would be timing a different kernel than production picks, so it is computed the same
// way here: complete row blocks and a column extent that fills the tile.
using Q8Mma = void (*)(bool, const Tensor&, const Weight&, Tensor&, cudaStream_t);

void sweep_q8(std::int32_t hidden, const ninfer::bench::SweepOptions& base) {
    const std::int32_t max_tokens = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q8_G32_FP16, kRows, hidden, hidden, {0x31, 0x00, 0x3c00});
    ninfer::DeviceBuffer input(static_cast<std::size_t>(hidden) * max_tokens * 2);
    ninfer::DeviceBuffer residual(static_cast<std::size_t>(kRows) * max_tokens * 2);

    const auto tensors = [&](std::int32_t tokens, Tensor& x, Tensor& out) {
        x   = Tensor(input.p, DType::BF16, {hidden, tokens});
        out = Tensor(residual.p, DType::BF16, {kRows, tokens});
    };
    const auto direct = [&](void (*launch)(const Tensor&, const Weight&, Tensor&, cudaStream_t)) {
        return [&, launch](std::int32_t tokens, cudaStream_t stream) {
            Tensor x, out;
            tensors(tokens, x, out);
            launch(x, packed.weight, out, stream);
        };
    };
    const auto tiled = [&](Q8Mma launch, std::int32_t tile_cols) {
        return [&, launch, tile_cols](std::int32_t tokens, cudaStream_t stream) {
            const bool full = (kRows % 64) == 0 && (tokens % tile_cols) == 0;
            Tensor x, out;
            tensors(tokens, x, out);
            launch(full, x, packed.weight, out, stream);
        };
    };

    std::vector<ninfer::bench::SweepEntry> schedules{
        {"decode_r16", direct(&detail::q8_linear_add_decode_r16_launch), 1},
        {"splitk_exact_t", direct(&detail::q8_linear_add_splitk_mma_launch), 48},
        {"splitk_capacity", direct(&detail::q8_linear_add_splitk_capacity_launch), 64},
        {"medium_splitk", direct(&detail::q8_linear_add_medium_splitk_launch), 0},
        {"grouped_splitk", direct(&detail::q8_linear_add_grouped_launch), 0},
        {"mma_r32_c64", tiled(&detail::q8_linear_add_mma_r32_c64_launch, 64), 0},
        {"mma_r32_c96", tiled(&detail::q8_linear_add_mma_r32_c96_launch, 96), 0},
        {"mma_r32_c128", tiled(&detail::q8_linear_add_mma_r32_c128_launch, 128), 0},
        {"mma_r48_c64", tiled(&detail::q8_linear_add_mma_r48_c64_launch, 64), 0},
        {"mma_r48_c96", tiled(&detail::q8_linear_add_mma_r48_c96_launch, 96), 0},
        {"mma_r64_c64", tiled(&detail::q8_linear_add_mma_r64_c64_launch, 64), 0},
        {"mma_r64_c96", tiled(&detail::q8_linear_add_mma_r64_c96_launch, 96), 0},
        {"mma_r64_c112", tiled(&detail::q8_linear_add_mma_r64_c112_launch, 112), 0},
        {"mma_r64_c128", tiled(&detail::q8_linear_add_mma_r64_c128_launch, 128), 0},
        {"mma_r128_c64", tiled(&detail::q8_linear_add_mma_r128_c64_launch, 64), 0},
        {"mma_r128_c80", tiled(&detail::q8_linear_add_mma_r128_c80_launch, 80), 0},
    };

    const std::string title = "q8 dense linear_add n=5120 k=" + std::to_string(hidden);
    ninfer::bench::SweepOptions options = base;
    options.title                       = title.c_str();
    std::string routed;
    options.routed_name = [hidden, &routed](std::int32_t tokens) -> const char* {
        const detail::Q8LinearAddProblem problem{kRows, hidden, hidden, tokens};
        routed = detail::q8_linear_add_schedule_name(
            detail::q8_linear_add_resolve_plan(problem).schedule);
        return routed.c_str();
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x, out;
        tensors(tokens, x, out);
        detail::q8_linear_add_dispatch(x, packed.weight, out, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

void sweep_q4(const ninfer::bench::SweepOptions& base) {
    constexpr std::int32_t kHidden = 6144;
    const std::int32_t max_tokens  = *std::max_element(base.tokens.begin(), base.tokens.end());
    ninfer::bench::PackedQuantizedWeight packed = ninfer::bench::make_row_split_weight(
        QType::Q4_G64_FP16, kRows, kHidden, kHidden, {0x31, 0xa5, 0x3c00});
    ninfer::DeviceBuffer input(static_cast<std::size_t>(kHidden) * max_tokens * 2);
    ninfer::DeviceBuffer residual(static_cast<std::size_t>(kRows) * max_tokens * 2);

    const auto make = [&](detail::Q4LinearAddLaunch launch) {
        return [&, launch](std::int32_t tokens, cudaStream_t stream) {
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor out(residual.p, DType::BF16, {kRows, tokens});
            launch(x, packed.weight, out, stream);
        };
    };

    const std::vector<std::pair<const char*, detail::Q4LinearAddLaunch>> named{
        {"gemv", &detail::q4_linear_add_gemv_launch},
        {"ksplit_c4", &detail::q4_linear_add_ksplit4_launch},
        {"ksplit_c8", &detail::q4_linear_add_ksplit8_launch},
        {"ksplit_c16", &detail::q4_linear_add_ksplit16_launch},
        {"ksplit_c24", &detail::q4_linear_add_ksplit24_launch},
        {"ksplit_c32", &detail::q4_linear_add_ksplit32_launch},
        {"mma_r32_c32", &detail::q4_linear_add_mma_r32_c32_launch},
        {"mma_r32_c64", &detail::q4_linear_add_mma_r32_c64_launch},
        {"mma_r64_c48", &detail::q4_linear_add_mma_r64_c48_launch},
        {"mma_r64_c64", &detail::q4_linear_add_mma_r64_c64_launch},
        {"mma_r64_c80", &detail::q4_linear_add_mma_r64_c80_launch},
        {"mma_r64_c96", &detail::q4_linear_add_mma_r64_c96_launch},
        {"mma_r64_c112", &detail::q4_linear_add_mma_r64_c112_launch},
        {"mma_r64_c128", &detail::q4_linear_add_mma_r64_c128_launch},
    };
    // The K-split capacities are masked to their compile-time extent, and the GEMV is a
    // one-column kernel; both silently do less work rather than failing past their domain.
    const std::int32_t caps[] = {1, 4, 8, 16, 24, 32, 0, 0, 0, 0, 0, 0, 0, 0};

    std::vector<ninfer::bench::SweepEntry> schedules;
    for (std::size_t i = 0; i < named.size(); ++i) {
        schedules.push_back({named[i].first, make(named[i].second), caps[i]});
    }

    ninfer::bench::SweepOptions options = base;
    options.title                       = "q4 dense linear_add n=5120 k=6144";
    options.routed_name                 = [&named](std::int32_t tokens) -> const char* {
        const detail::Q4LinearAddLaunch chosen = detail::select_q4_linear_add(kRows, kHidden, tokens);
        for (const auto& entry : named) {
            if (entry.second == chosen) { return entry.first; }
        }
        return "(unswept)";
    };
    options.public_op = [&](std::int32_t tokens, cudaStream_t stream) {
        Tensor x(input.p, DType::BF16, {kHidden, tokens});
        Tensor out(residual.p, DType::BF16, {kRows, tokens});
        detail::select_q4_linear_add(kRows, kHidden, tokens)(x, packed.weight, out, stream);
    };
    ninfer::bench::run_sweep(options, schedules);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    ninfer::bench::SweepOptions options;
    options.tokens = {1,  2,  4,   8,   12,  16,  24,  32,  40,  48,  56,  64,
                      72, 80, 96,  112, 128, 160, 192, 224, 256, 384, 512, 1024};
    const bool q4  = argc > 1 && std::string(argv[1]) == "--q4";
    if (q4) {
        --argc;
        ++argv;
    }
    if (!ninfer::bench::parse_sweep_args(argc, argv, options)) {
        std::fprintf(stderr,
                     "usage: %s [--q4] [--tokens T,...] [--repeat N] [--warmup N] [--spread]\n",
                     argv[0]);
        return 2;
    }
    if (q4) {
        sweep_q4(options);
    } else {
        sweep_q8(6144, options);
        sweep_q8(17408, options);
    }
    return 0;
}
