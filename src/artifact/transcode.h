#pragma once

#include "artifact/reader.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::artifact {

// A device tensor may be materialized in a narrower grouped format than the artifact stores. The
// plan reserves the target encoding's size, the materializer writes target bytes, and every
// consumer (bindings, the overlay eviction mirror, arena accounting) sees only the target format.
// This trades weight precision for device memory; it is selected per tensor by the target.
enum class DeviceTranscode : std::uint8_t {
    None,
    W8G32ToQ4G64,
    W8G32ToQ6G64,
};

[[nodiscard]] NumericFormat transcode_source_format(DeviceTranscode transcode);
[[nodiscard]] NumericFormat transcode_target_format(DeviceTranscode transcode);

// Requantizes a row-split-k128-v1 W8G32_F16S payload to the target format's row-split-k128-v1
// encoding. Each 64-column group takes, of 25 clipping ratios of absmax/qmax in [0.70, 1.18], the
// fp16 scale whose round-to-nearest codes minimise the group's squared error against the W8
// values; plain absmax/qmax is one candidate. Groups are independent, so the result does not depend
// on the worker count.
void transcode_row_split(DeviceTranscode transcode, std::span<const std::uint64_t> shape,
                         std::span<const std::byte> source, std::span<std::byte> destination);

} // namespace ninfer::artifact
