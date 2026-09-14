#include "targets/guarded_main.h"
#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/transcode.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace {

std::filesystem::path artifact_path() {
    if (const char* env = std::getenv("NINFER_QWEN3_6_35B_A3B_WEIGHTS");
        env != nullptr && *env != '\0') {
        return env;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out/qwen3_6_35b_a3b.ninfer";
}

ninfer::targets::qwen3_6::StartupFeatures load_features(bool vision,
                                                        ninfer::SpeculativeBackend speculative) {
    return {
        .vision        = vision,
        .speculative   = speculative,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

const ninfer::artifact::DeviceMaterialization*
device_object(const ninfer::artifact::MaterializationPlan& plan,
              ninfer::artifact::ObjectHandle handle) {
    for (const auto& object : plan.device_objects) {
        if (object.object.index == handle.index) { return &object; }
    }
    return nullptr;
}

// --mtp-experts-q4 must reach the binding: the two routed MTP expert blocks are planned in their
// transcoded encodings, the plan records the formats load_moe will build weights with, and the
// arena shrinks by exactly the bytes those encodings save.
int check_mtp_experts_q4(const ninfer::artifact::Reader& reader) {
    using ninfer::artifact::DeviceTranscode;
    using ninfer::artifact::NumericFormat;
    namespace bindings = ninfer::targets::qwen3_6_35b_a3b::detail;

    auto features = load_features(false, ninfer::SpeculativeBackend::Mtp);
    ninfer::artifact::Binder native_binder(reader);
    const auto native = bindings::bind_artifact(native_binder, features);
    features.mtp_experts_q4 = true;
    ninfer::artifact::Binder transcoded_binder(reader);
    const auto transcoded = bindings::bind_artifact(transcoded_binder, features);

    constexpr std::array<std::uint64_t, 2> gate_up_shape = {262144, 2048};
    constexpr std::array<std::uint64_t, 2> down_shape    = {524288, 512};
    const auto w8_gate_up = ninfer::artifact::row_split_geometry(NumericFormat::W8G32_F16S, gate_up_shape);
    const auto w8_down    = ninfer::artifact::row_split_geometry(NumericFormat::W8G32_F16S, down_shape);
    const auto q4_gate_up = ninfer::artifact::row_split_geometry(NumericFormat::Q4G64_F16S, gate_up_shape);
    const auto q6_down    = ninfer::artifact::row_split_geometry(NumericFormat::Q6G64_F16S, down_shape);

    const auto& native_moe      = native.bindings.mtp.moe;
    const auto& transcoded_moe  = transcoded.bindings.mtp.moe;
    const auto* native_gate_up  = device_object(native.materialization, native_moe.routed_gate_up);
    const auto* native_down     = device_object(native.materialization, native_moe.routed_down);
    const auto* gate_up         = device_object(transcoded.materialization, transcoded_moe.routed_gate_up);
    const auto* down            = device_object(transcoded.materialization, transcoded_moe.routed_down);
    const std::uint64_t saved   = (w8_gate_up.encoded_bytes - q4_gate_up.encoded_bytes) +
                                (w8_down.encoded_bytes - q6_down.encoded_bytes);
    const std::uint64_t shrink  = native.materialization.device_capacity_bytes -
                                 transcoded.materialization.device_capacity_bytes;
    const bool ok =
        native_gate_up != nullptr && native_down != nullptr && gate_up != nullptr && down != nullptr &&
        native_gate_up->transcode == DeviceTranscode::None &&
        native_down->transcode == DeviceTranscode::None &&
        native_moe.routed_gate_up_format == NumericFormat::W8G32_F16S &&
        native_moe.routed_down_format == NumericFormat::W8G32_F16S &&
        gate_up->transcode == DeviceTranscode::W8G32ToQ4G64 &&
        gate_up->bytes == q4_gate_up.encoded_bytes &&
        down->transcode == DeviceTranscode::W8G32ToQ6G64 && down->bytes == q6_down.encoded_bytes &&
        transcoded_moe.routed_gate_up_format == NumericFormat::Q4G64_F16S &&
        transcoded_moe.routed_down_format == NumericFormat::Q6G64_F16S &&
        // Alignment padding between objects can move by less than one tensor alignment each.
        shrink + 512 >= saved && shrink <= saved + 512;
    if (!ok) {
        std::cerr << "--mtp-experts-q4 did not plan the MTP experts as Q4G64/Q6G64: shrink=" << shrink
                  << " expected~" << saved << '\n';
        return 1;
    }
    return 0;
}

// --embedding-q4/--embedding-q6 must reach the binding: the token embedding is planned in the
// transcoded encoding and the plan records the format the loaded Weight is built with.
int check_embedding_transcodes(const ninfer::artifact::Reader& reader) {
    using ninfer::artifact::DeviceTranscode;
    using ninfer::artifact::NumericFormat;
    namespace bindings = ninfer::targets::qwen3_6_35b_a3b::detail;

    constexpr std::array<std::uint64_t, 2> shape = {248320, 2048};
    struct Case {
        bool q4;
        bool q6;
        DeviceTranscode transcode;
        NumericFormat format;
        const char* name;
    };
    constexpr std::array<Case, 3> cases{{
        {false, false, DeviceTranscode::None, NumericFormat::W8G32_F16S, "native"},
        {true, false, DeviceTranscode::W8G32ToQ4G64, NumericFormat::Q4G64_F16S, "--embedding-q4"},
        {false, true, DeviceTranscode::W8G32ToQ6G64, NumericFormat::Q6G64_F16S, "--embedding-q6"},
    }};
    for (const Case& test : cases) {
        auto features         = load_features(false, ninfer::SpeculativeBackend::Mtp);
        features.embedding_q4 = test.q4;
        features.embedding_q6 = test.q6;
        ninfer::artifact::Binder binder(reader);
        const auto plan    = bindings::bind_artifact(binder, features);
        const auto* object = device_object(plan.materialization, plan.bindings.token_embedding);
        const auto bytes   = ninfer::artifact::row_split_geometry(test.format, shape).encoded_bytes;
        if (object == nullptr || object->transcode != test.transcode || object->bytes != bytes ||
            plan.bindings.token_embedding_format != test.format) {
            std::cerr << test.name << " did not plan the token embedding as expected\n";
            return 1;
        }
    }
    return 0;
}

} // namespace

int run_dflash_load_plan_checks() {
    const std::filesystem::path path = artifact_path();
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "skip: real 35B artifact is unavailable at " << path << '\n';
        return 77;
    }

    ninfer::artifact::Reader reader(path);
    if (const int result = check_mtp_experts_q4(reader); result != 0) { return result; }
    if (const int result = check_embedding_transcodes(reader); result != 0) { return result; }
    {
        ninfer::artifact::Binder binder(reader);
        // These counts pin a DFlash-carrying artifact. A compact artifact without the DFlash bundle
        // is a different variant on disk, not a regression, and it trips every pinned number at
        // once -- so say so and skip instead of failing. The numbers that actually move when
        // bindings change are device_objects and device_capacity_bytes; object_count and the dflash
        // handles only reflect which variant was loaded.
        if (!binder.has_object("dflash/feature_projection")) {
            std::cout << "skip: this artifact carries no DFlash bundle, so the pinned "
                         "DFlash-enabled plan cannot be checked against it\n";
            return 77;
        }
        const auto plan = ninfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(
            binder, load_features(true, ninfer::SpeculativeBackend::Mtp));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 883 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 22'360'207'360ULL ||
            plan.bindings.dflash.feature_projection.index != 889 ||
            plan.bindings.dflash.final_norm.index != 939) {
            // Print the actuals: this test pins exact plan numbers, so when it trips the only
            // useful next question is which of them moved and by how much.
            std::cerr << "MTP+Vision materialization plan changed resident weights\n"
                      << "  object_count=" << plan.materialization.object_count << " (want 940)\n"
                      << "  device_objects=" << plan.materialization.device_objects.size()
                      << " (want 883)\n"
                      << "  host_objects=" << plan.materialization.host_objects.size()
                      << " (want 6)\n"
                      << "  device_bytes=" << plan.materialization.device_capacity_bytes
                      << " (want 22360207360)\n"
                      << "  dflash.feature_projection=" << plan.bindings.dflash.feature_projection.index
                      << " (want 889)\n"
                      << "  dflash.final_norm=" << plan.bindings.dflash.final_norm.index
                      << " (want 939)\n";
            return 1;
        }
    }
    {
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(
            binder, load_features(false, ninfer::SpeculativeBackend::DFlash));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 586 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 21'591'653'888ULL) {
            std::cerr << "DFlash-only materialization plan is incomplete: device_objects="
                      << plan.materialization.device_objects.size()
                      << " device_bytes=" << plan.materialization.device_capacity_bytes << '\n';
            return 1;
        }
    }
    {
        ninfer::artifact::Binder binder(reader);
        const auto plan = ninfer::targets::qwen3_6_35b_a3b::detail::bind_artifact(
            binder, load_features(true, ninfer::SpeculativeBackend::DFlash));
        if (plan.materialization.object_count != 940 ||
            plan.materialization.device_objects.size() != 919 ||
            plan.materialization.host_objects.size() != 6 ||
            plan.materialization.device_capacity_bytes != 21'872'326'656ULL) {
            std::cerr << "DFlash+Vision materialization plan is incomplete: device_objects="
                      << plan.materialization.device_objects.size()
                      << " device_bytes=" << plan.materialization.device_capacity_bytes << '\n';
            return 1;
        }
    }
    return 0;
}

NINFER_GUARDED_TEST_MAIN(run_dflash_load_plan_checks)
