#include "runtime/engine/context_cost.h"

#include <array>
#include <vector>

namespace ninfer::runtime {

const std::array<ContextTransferCost, 3>& generic_context_transfer_cost() {
    // Conservative but useful on an unmeasured machine. These values preserve numerical ranking;
    // they are not a switch that disables the cost model.
    static constexpr std::array<ContextTransferCost, 3> value{
        ContextTransferCost{
            .batch_ns = 100'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 50'000, .operation_ns = 12'000, .ns_per_byte_q32 = 107'374'182},
        ContextTransferCost{
            .batch_ns = 25'000, .operation_ns = 12'000, .ns_per_byte_q32 = 42'949'673},
    };
    return value;
}

const ContextPrefillCost& generic_context_prefill_cost() {
    // The slower measured 27B profile is the conservative generic estimate. An unknown model still
    // receives continuous recomputation costs instead of falling back to a discrete tuple.
    static constexpr ContextPrefillCost value{
        .chunk_ns              = 40'813'570,
        .token_ns_q32          = 1'012'273'154'411'951,
        .attention_pair_ns_q32 = 30'497'396'515,
        .vision_item_ns        = 5'986'585,
        .vision_patch_ns_q32   = 23'710'212'854'694,
    };
    return value;
}

// Accepted project defaults live only in this table and are compiled into the binary. Runtime JSON
// presets are independent local-machine overrides and never become a build dependency.
const std::vector<ContextCostMachinePreset>& compiled_context_cost_defaults() {
    static const std::vector<ContextCostMachinePreset> defaults{
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-5090-sm120",
            // Unified from the measured 2026-08-24 transfer corpus. A fresh machine-only transfer
            // calibration can replace this whole array without loading a model.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 48'655, .operation_ns = 7'433, .ns_per_byte_q32 = 91'692'315},
                    ContextTransferCost{
                        .batch_ns = 0, .operation_ns = 8'457, .ns_per_byte_q32 = 83'354'284},
                    ContextTransferCost{
                        .batch_ns = 3'343, .operation_ns = 9'520, .ns_per_byte_q32 = 2'658'314},
                },
            .prefill =
                {
                    ContextPrefillPreset{
                        .model_id   = "qwen3.6-27b",
                        .weights_id = "groupwise-int",
                        .cost =
                            {
                                .chunk_ns              = 40'813'570,
                                .token_ns_q32          = 1'012'273'154'411'951,
                                .attention_pair_ns_q32 = 30'497'396'515,
                                .vision_item_ns        = 5'986'585,
                                .vision_patch_ns_q32   = 23'710'212'854'694,
                            },
                    },
                    ContextPrefillPreset{
                        .model_id   = "qwen3.8-27b",
                        .weights_id = "nvfp4",
                        .cost =
                            {
                                .chunk_ns              = 14'672'989,
                                .token_ns_q32          = 375'800'765'711'778,
                                .attention_pair_ns_q32 = 8'200'474'657,
                                .vision_item_ns        = 5'860'255,
                                .vision_patch_ns_q32   = 23'832'529'381'413,
                            },
                    },
                },
        },
        ContextCostMachinePreset{
            .hardware_class = "nvidia-geforce-rtx-3090-sm86",
            // Measured 2026-09-14 on the fork host (Windows 11, CUDA 12.8, 315 W cap) with
            // ninfer_context_cost_bench, prefill at --max-context 65536 for each groupwise-int
            // artifact. The generic prefill model predicted 26.0 s for a 55,000-token Qwen3.8-27B
            // prefill that took 57 s on this card; this fit predicts 58.1 s.
            .transfer =
                std::array{
                    ContextTransferCost{
                        .batch_ns = 54'593, .operation_ns = 6'305, .ns_per_byte_q32 = 185'859'373},
                    ContextTransferCost{
                        .batch_ns = 48'436, .operation_ns = 6'878, .ns_per_byte_q32 = 172'624'952},
                    ContextTransferCost{
                        .batch_ns = 338, .operation_ns = 6'742, .ns_per_byte_q32 = 10'830'495},
                },
            .prefill =
                {
                    ContextPrefillPreset{
                        .model_id   = "qwen3.8-27b",
                        .weights_id = "groupwise-int",
                        .cost =
                            {
                                .chunk_ns              = 42'827'781,
                                .token_ns_q32          = 3'084'986'452'624'619,
                                .attention_pair_ns_q32 = 46'293'818'703,
                                .vision_item_ns        = 6'744'019,
                                .vision_patch_ns_q32   = 109'358'098'255'328,
                            },
                    },
                    ContextPrefillPreset{
                        .model_id   = "qwen3.6-27b",
                        .weights_id = "groupwise-int",
                        .cost =
                            {
                                .chunk_ns              = 42'711'733,
                                .token_ns_q32          = 3'058'624'051'907'175,
                                .attention_pair_ns_q32 = 49'575'941'648,
                                .vision_item_ns        = 7'025'741,
                                .vision_patch_ns_q32   = 109'178'678'749'881,
                            },
                    },
                    ContextPrefillPreset{
                        .model_id   = "qwen3.6-35b-a3b",
                        .weights_id = "groupwise-int",
                        .cost =
                            {
                                .chunk_ns              = 32'306'259,
                                .token_ns_q32          = 671'808'709'554'317,
                                .attention_pair_ns_q32 = 17'853'060'462,
                                .vision_item_ns        = 6'937'447,
                                .vision_patch_ns_q32   = 106'078'200'821'712,
                            },
                    },
                },
        },
    };
    return defaults;
}

} // namespace ninfer::runtime
