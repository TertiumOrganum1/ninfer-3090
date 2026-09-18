// Integer-activation route for the 27B GDN input projection, on sm_86.
//
// The groupwise-int artifact binds this projection as two parents over one input:
//
//   query_key  Q4_G64_FP16 [4096, 5120]  -> qkv rows [0, 4096)
//   value_z    Q5_G64_FP16 [12288, 5120] -> rows [0, 6144)     -> qkv rows [4096, 10240)
//                                           rows [6144, 12288) -> z rows [0, 6144)
//
// Both read the same x, so the activation quantisation happens once and all three launches share
// it: the quantiser is O(K*T) against the GEMMs' O(N*K*T), and at this shape that is well under a
// percent of the work. The three launches differ only in which weight they read and where their
// rows land, which is what rowsplit_a8::StoreEpilogue carries.
//
// 48 of the 27B's 64 layers are GDN, so this is the largest single projection in a prefill: an
// Nsight profile of a 4,096-token prefill put it at 749 ms of 3,280 ms with the A16 route.

#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"

#include "core/device.h"
#include "ops/common/rowsplit_a8_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

namespace a8 = rowsplit_a8;

constexpr std::int32_t kHidden    = 5120;
constexpr std::int32_t kQkRows    = 4096;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kZRows     = 6144;
constexpr std::int32_t kQkvRows   = kQkRows + kValueRows;
constexpr std::int32_t kParentRows = kValueRows + kZRows;

template <class Codec>
void launch_range(const Weight& weight, const std::int8_t* codes, const __half* scales,
                  std::int32_t tokens, std::int32_t row_begin, std::int32_t row_count,
                  a8::StoreEpilogue epilogue, cudaStream_t stream) {
    const dim3 grid(row_count / a8::kBM, tokens / a8::kBN);
    a8::a8_mma_kernel<Codec, kHidden, a8::StoreEpilogue>
        <<<grid, a8::kThreads, a8::shared_bytes(), stream>>>(
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const __half*>(weight.scales), codes, scales, tokens, row_begin, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool q4_q5_gdn_input_a8_supported(const Weight& qk, const Weight& value_z, std::int32_t tokens) {
    return a8::tokens_supported(tokens) && qk.qtype == QType::Q4_G64_FP16 &&
           qk.layout == QuantLayout::RowSplit && qk.n == kQkRows && qk.k == kHidden &&
           qk.group == a8::kGroup && qk.qdata != nullptr && qk.scales != nullptr &&
           value_z.qtype == QType::Q5_G64_FP16 && value_z.layout == QuantLayout::RowSplit &&
           value_z.n == kParentRows && value_z.k == kHidden && value_z.group == a8::kGroup &&
           value_z.qdata != nullptr && value_z.qhigh != nullptr && value_z.scales != nullptr;
}

std::size_t q4_q5_gdn_input_a8_workspace_capacity_bytes(std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("q4_q5 gdn_input a8 workspace: invalid token interval");
    }
    return a8::activation_workspace_bytes(kHidden, max_tokens);
}

void q4_q5_gdn_input_a8_launch(const Tensor& x, const Weight& qk, const Weight& value_z, Tensor& qkv,
                               Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q4_q5_gdn_input_a8_supported(qk, value_z, tokens)) {
        throw std::invalid_argument("q4_q5 gdn_input a8: unsupported profile");
    }

    auto scope = workspace.scope();
    const DeviceSpan codes =
        workspace.alloc_bytes(static_cast<std::size_t>(tokens) * kHidden);
    const DeviceSpan scales = workspace.alloc_bytes(
        static_cast<std::size_t>(tokens) * (kHidden / a8::kGroup) * sizeof(__half));

    a8::quantize_activations<kHidden><<<tokens, 128, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data), tokens,
        reinterpret_cast<std::int8_t*>(codes.data), reinterpret_cast<__half*>(scales.data));
    CUDA_CHECK(cudaGetLastError());

    const auto* x_codes  = reinterpret_cast<const std::int8_t*>(codes.data);
    const auto* x_scales = reinterpret_cast<const __half*>(scales.data);
    auto* qkv_data       = static_cast<__nv_bfloat16*>(qkv.data);
    auto* z_data         = static_cast<__nv_bfloat16*>(z.data);

    launch_range<a8::Q4Codec>(qk, x_codes, x_scales, tokens, 0, kQkRows,
                              a8::StoreEpilogue{qkv_data, kQkvRows, 0}, stream);
    launch_range<a8::Q5Codec>(value_z, x_codes, x_scales, tokens, 0, kValueRows,
                              a8::StoreEpilogue{qkv_data, kQkvRows, kQkRows}, stream);
    launch_range<a8::Q5Codec>(value_z, x_codes, x_scales, tokens, kValueRows, kZRows,
                              a8::StoreEpilogue{z_data, kZRows, 0}, stream);
}

} // namespace ninfer::ops::detail
