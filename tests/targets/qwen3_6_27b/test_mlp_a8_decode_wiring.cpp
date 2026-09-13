// Public wiring for --mlp-a8-decode: which phase may take the integer route, and whether target
// planning reserves what that route allocates.
//
// The FP64 oracle in tests/ops/linear_swiglu/test_q4a8_int.cpp invokes the schedule directly, so it
// says nothing about either. Both were wrong once and neither failure is visible from a kernel
// test: the flag reached prefill, which quantised the tail chunk of a scoring run that documents
// itself as unaffected, and the planner passed A16Only, whose capacity across 16..32 columns is
// zero because that band routes to SmallTTiled -- so the shortfall was the whole allocation rather
// than a margin.

#include "targets/guarded_main.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

using ninfer::QType;
using ninfer::Weight;
using ninfer::targets::qwen3_6_27b::detail::DensePostMixerPayload;
using ninfer::targets::qwen3_6_27b::detail::Variant;
using ninfer::targets::qwen3_6_27b::detail::mlp_policy;
using Phase = ninfer::targets::qwen3_6::TextPhase;

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL " << what << '\n';
        ++failures;
    }
}

// mlp_policy reads only qtype and the registered shape, so the payload needs no device memory.
DensePostMixerPayload payload(bool a8_decode) {
    DensePostMixerPayload out;
    out.gate_up.qtype = QType::Q4G64_F16S;
    out.gate_up.n     = 34816;
    out.gate_up.k     = 5120;
    out.down.qtype    = QType::Q5G64_F16S;
    out.down.n        = 5120;
    out.down.k        = 17408;
    out.a8_decode     = a8_decode;
    return out;
}

const char* name(ninfer::ops::LinearPolicy policy) {
    switch (policy) {
    case ninfer::ops::LinearPolicy::A16Only:
        return "A16Only";
    case ninfer::ops::LinearPolicy::AllowA8:
        return "AllowA8";
    case ninfer::ops::LinearPolicy::AllowA4:
        return "AllowA4";
    case ninfer::ops::LinearPolicy::AllowA8Int:
        return "AllowA8Int";
    case ninfer::ops::LinearPolicy::AllowA8IntDecode:
        return "AllowA8IntDecode";
    }
    return "unknown";
}

void run_phase_gate() {
#if defined(NINFER_SM8X_COMPAT)
    constexpr auto kBase = ninfer::ops::LinearPolicy::AllowA8Int;
#else
    constexpr auto kBase = ninfer::ops::LinearPolicy::A16Only;
#endif
    const DensePostMixerPayload on  = payload(true);
    const DensePostMixerPayload off = payload(false);

    // The whole point of the flag's name. A prefill chunk of 16..32 columns is reachable from an
    // ordinary prompt's ragged tail and from causal_score's remainder, so this is the guard that
    // keeps a decode-time trade out of a scored measurement.
    const auto prefill_on = mlp_policy(on, Phase::Prefill);
    check(prefill_on == kBase,
          std::string("prefill with the flag set must stay ") + name(kBase) + ", got " +
              name(prefill_on));

    const auto verify_off = mlp_policy(off, Phase::Verify);
    check(verify_off == kBase, std::string("verify without the flag must stay ") + name(kBase) +
                                   ", got " + name(verify_off));

    const auto verify_on = mlp_policy(on, Phase::Verify);
#if defined(NINFER_SM8X_COMPAT)
    check(verify_on == ninfer::ops::LinearPolicy::AllowA8IntDecode,
          std::string("verify with the flag set must widen to AllowA8IntDecode, got ") +
              name(verify_on));
#else
    // No integer route off sm_86; the flag must not conjure one.
    check(verify_on == kBase, std::string("verify off sm_86 must stay ") + name(kBase) + ", got " +
                                  name(verify_on));
#endif

    // A shape the integer route is not registered for must be untouched whatever the flag says.
    DensePostMixerPayload other = payload(true);
    other.gate_up.n             = 4096;
    check(mlp_policy(other, Phase::Verify) == ninfer::ops::LinearPolicy::A16Only,
          "an unregistered gate_up shape must stay A16Only even with the flag set");
}

void run_planning() {
    // The planner cannot see the runtime flag, so it reserves the superset for the phase that can
    // take the route. Verify must therefore plan at least as much as prefill across the band the
    // route covers, and strictly more where prefill's A16 capacity is zero.
    for (const std::int32_t width : {16, 24, 32}) {
        const std::size_t prefill = Variant::post_mixer_workspace_capacity_bytes(
            Variant::WeightsProfile::Qwen38GroupwiseInt, Phase::Prefill, width, width);
        const std::size_t verify = Variant::post_mixer_workspace_capacity_bytes(
            Variant::WeightsProfile::Qwen38GroupwiseInt, Phase::Verify, width, width);
        check(verify >= prefill, "verify planning must not undercut prefill at T=" +
                                     std::to_string(width) + " (" + std::to_string(verify) + " < " +
                                     std::to_string(prefill) + ")");
#if defined(NINFER_SM8X_COMPAT)
        // The route stages one s8 code per (token, hidden) plus an FP16 scale per 64-k group, over
        // the padded tile width. Anything less than that and the arena can throw mid-round.
        const std::size_t codes  = static_cast<std::size_t>(width) * 5120;
        check(verify >= prefill + codes,
              "verify planning must cover the integer scratch at T=" + std::to_string(width) +
                  " (needs at least " + std::to_string(codes) + " more than prefill's " +
                  std::to_string(prefill) + ", got " + std::to_string(verify) + ")");
#endif
    }

    // Outside the route's band the two phases have nothing to differ about.
    for (const std::int32_t width : {1, 8, 64}) {
        const std::size_t prefill = Variant::post_mixer_workspace_capacity_bytes(
            Variant::WeightsProfile::Qwen38GroupwiseInt, Phase::Prefill, width, width);
        const std::size_t verify = Variant::post_mixer_workspace_capacity_bytes(
            Variant::WeightsProfile::Qwen38GroupwiseInt, Phase::Verify, width, width);
        check(verify == prefill, "planning must match across phases at T=" +
                                     std::to_string(width) + ", outside the route's 16..32 band");
    }
}

int run() {
    run_phase_gate();
    run_planning();
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " --mlp-a8-decode phase gate and workspace planning\n";
    return failures == 0 ? 0 : 1;
}

} // namespace

NINFER_GUARDED_TEST_MAIN(run)
