#pragma once
/// @file GraphYolo11Partition.h  The YOLO11 weight blob in a flash data partition (ESP32 only).
///
/// The blob is written by tools/yolo/yolo_export.py (`<name>_board.bin`) and flashed to the "model"
/// partition: the float parameters, then the int8 weights, then their scales. The float parameters
/// are read in place through the flash mapping; the int8 weights and the scales are copied into
/// PSRAM (the flash cache reads them slowly). Keep this object alive while the detector is used.

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"

#if defined(ESP32)
#include <string.h>

#include "esp_partition.h"

namespace tinyyolo {

class GraphYolo11Partition {
 public:
  GraphYolo11Partition() {}
  ~GraphYolo11Partition() {
    if (handle_ != 0) esp_partition_munmap(handle_);
  }
  GraphYolo11Partition(const GraphYolo11Partition&) = delete;
  GraphYolo11Partition& operator=(const GraphYolo11Partition&) = delete;

  /// Maps the data partition `label` and copies the int8 data to PSRAM. The counts are the sizes of
  /// the export (for the nano model: 2616232 floats, 2606816 int8 weights, 8136 scales).
  Status begin(const char* label, size_t float_count, size_t int8_count, size_t scale_count) {
    const esp_partition_t* part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, label);
    if (part == nullptr) {
      TINYYOLO_LOGE("GraphYolo11Partition", "no '%s' partition", label);
      return Status::InvalidArgument;
    }
    const size_t float_bytes = float_count * sizeof(float);
    const size_t blob_bytes = float_bytes + int8_count + scale_count * sizeof(float);
    if (blob_bytes > part->size) {
      TINYYOLO_LOGE("GraphYolo11Partition", "the partition is smaller than the blob");
      return Status::InvalidArgument;
    }
    const void* mapped = nullptr;
    if (esp_partition_mmap(part, 0, blob_bytes, ESP_PARTITION_MMAP_DATA, &mapped, &handle_) != ESP_OK) {
      TINYYOLO_LOGE("GraphYolo11Partition", "cannot map '%s'", label);
      handle_ = 0;
      return Status::InvalidArgument;
    }
    const uint8_t* src = static_cast<const uint8_t*>(mapped);
    weights_ = reinterpret_cast<const float*>(src);
    TINYYOLO_CHECK(qweights_.allocate(int8_count, MemoryPlacement::PsramFirst));
    TINYYOLO_CHECK(qscales_.allocate(scale_count, MemoryPlacement::PsramFirst));
    memcpy(qweights_.data(), src + float_bytes, int8_count);
    memcpy(qscales_.data(), src + float_bytes + int8_count, scale_count * sizeof(float));
    return Status::Ok;
  }

  const float* weights() const { return weights_; }
  const int8_t* qweights() const { return qweights_.data(); }
  const float* qscales() const { return qscales_.data(); }

 private:
  esp_partition_mmap_handle_t handle_ = 0;
  const float* weights_ = nullptr;
  Buffer<int8_t> qweights_;
  Buffer<float> qscales_;
};

}  // namespace tinyyolo

#endif  // ESP32
