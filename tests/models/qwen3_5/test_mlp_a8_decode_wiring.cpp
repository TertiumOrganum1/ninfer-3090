// Public wiring for --mlp-a8-decode: which phase may take the integer route, and whether target
// planning reserves what that route allocates.
//
// The FP64 oracle in tests/ops/linear_swiglu/test_q4a8_int.cpp invokes the schedule directly, so it
// says nothing about either. Both were wrong once and neither failure is visible from a kernel
// test: the flag reached prefill, which quantised the tail chunk of a scoring run that documents
// itself as unaffected, and the planner passed A16Only, whose capacity across 16..32 columns is
// zero because that band routes to SmallTTiled -- so the shortfall was the whole allocation rather
// than a margin.

#include "guarded_main.h"
#include "models/qwen3_5/execution/ffn.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

using ninfer::QType;
using ninfer::models::qwen3_5::execution::DenseParameters;
using ninfer::models::qwen3_5::execution::ffn_workspace_bytes;
using ninfer::ops::LinearPolicy;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

// Planning reads only qtype, shape and policy, so the parameters need no device memory. The
// policies are what execution::Parameters assigns to the 27B groupwise-int MLP pair on sm_86.
DenseParameters parameters(bool a8_decode) {
    DenseParameters out;
    out.gate_up.weight.qtype = QType::Q4_G64_FP16;
    out.gate_up.weight.n     = 34816;
    out.gate_up.weight.k     = 5120;
    out.down.weight.qtype    = QType::Q5_G64_FP16;
    out.down.weight.n        = 5120;
    out.down.weight.k        = 17408;
#if defined(NINFER_SM8X_COMPAT)
    out.gate_up.policy = LinearPolicy::AllowA8Int;
    out.down.policy    = LinearPolicy::AllowA8Int;
#endif
    out.verify_gate_up_policy = a8_decode && out.gate_up.policy == LinearPolicy::AllowA8Int
                                    ? LinearPolicy::AllowA8IntDecode
                                    : out.gate_up.policy;
    return out;
}

void run_planning() {
    const DenseParameters on  = parameters(true);
    const DenseParameters off = parameters(false);
    for (const std::int32_t width : {16, 24, 32}) {
        const std::size_t prefill = ffn_workspace_bytes(on, width, width, false, false);
        const std::size_t verify  = ffn_workspace_bytes(on, width, width, false, true);
        // The flag must not change what a prefill (or scoring) call reserves or runs.
        check(prefill == ffn_workspace_bytes(off, width, width, false, false),
              "prefill planning must ignore --mlp-a8-decode at T=" + std::to_string(width));
        check(verify >= prefill, "verify planning must not undercut prefill at T=" +
                                     std::to_string(width) + " (" + std::to_string(verify) + " < " +
                                     std::to_string(prefill) + ")");
#if defined(NINFER_SM8X_COMPAT)
        // The route stages one s8 code per (token, hidden) plus an FP16 scale per 64-k group, over
        // the padded tile width. Anything less than that and the arena can throw mid-round.
        const std::size_t codes = static_cast<std::size_t>(width) * 5120;
        check(verify >= prefill + codes,
              "verify planning must cover the integer scratch at T=" + std::to_string(width) +
                  " (needs at least " + std::to_string(codes) + " more than prefill's " +
                  std::to_string(prefill) + ", got " + std::to_string(verify) + ")");
#endif
        check(ffn_workspace_bytes(off, width, width, false, true) == prefill,
              "without the flag, verify planning must match prefill at T=" +
                  std::to_string(width));
    }

    // Outside the route's band the two phases have nothing to differ about.
    for (const std::int32_t width : {1, 8, 64}) {
        check(ffn_workspace_bytes(on, width, width, false, true) ==
                  ffn_workspace_bytes(on, width, width, false, false),
              "planning must match across phases at T=" + std::to_string(width) +
                  ", outside the route's 16..32 band");
    }
}

int run() {
    run_planning();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " --mlp-a8-decode phase gate and workspace planning\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
