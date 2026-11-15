#pragma once
/// @file ModelFilePartition.h  A single-file model (docs/ModelFormat.md) in a flash data partition (ESP32).
///
/// The file is flashed to the partition (write_flash at the partition offset). The float parameters
/// are read in place through the flash mapping, and the int8 weights and their scales are copied
/// into PSRAM (the flash cache reads them slowly). Keep this object alive while the model is used.

#include "TinyYolo/app/ModelFile.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"

#if defined(ESP32)
#include <string.h>

#include "esp_partition.h"

namespace tinyyolo {

class ModelFilePartition {
 public:
  ModelFilePartition() {}
  ~ModelFilePartition() {
    if (handle_ != 0) esp_partition_munmap(handle_);
  }
  ModelFilePartition(const ModelFilePartition&) = delete;
  ModelFilePartition& operator=(const ModelFilePartition&) = delete;

  /// Maps the data partition `label`, checks the model file in it, and copies the int8 data to PSRAM.
  Status begin(const char* label) {
    const esp_partition_t* part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, label);
    if (part == nullptr) {
      TINYYOLO_LOGE("ModelFilePartition", "no '%s' partition", label);
      return Status::WeightError;
    }
    const void* mapped = nullptr;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &mapped, &handle_) != ESP_OK) {
      TINYYOLO_LOGE("ModelFilePartition", "cannot map '%s'", label);
      handle_ = 0;
      return Status::WeightError;
    }
    const uint8_t* data = static_cast<const uint8_t*>(mapped);
    const size_t size = tym::ModelFile::fileSize(data, part->size);
    if (size == 0) {
      TINYYOLO_LOGE("ModelFilePartition", "'%s' holds no model file", label);
      return Status::InvalidModel;
    }
    const Status status = file_.begin(data, size);
    if (status != Status::Ok) return status;
    if (file_.int8Count() == 0 || file_.scaleCount() == 0) return Status::InvalidModel;
    if (int8_.allocate(file_.int8Count(), MemoryPlacement::PsramFirst) != Status::Ok ||
        scales_.allocate(file_.scaleCount(), MemoryPlacement::PsramFirst) != Status::Ok) {
      return Status::OutOfMemory;
    }
    memcpy(int8_.data(), file_.int8Weights(), file_.int8Count());
    memcpy(scales_.data(), file_.int8Scales(), file_.scaleCount() * sizeof(float));
    return Status::Ok;
  }

  const tym::ModelFile& file() const { return file_; }
  const GraphModel& model() const { return file_.model(); }
  const float* floatWeights() const { return file_.floatWeights(); }   ///< in flash, read in place
  const int8_t* int8Weights() const { return int8_.data(); }           ///< in PSRAM
  const float* int8Scales() const { return scales_.data(); }           ///< in PSRAM

 private:
  esp_partition_mmap_handle_t handle_ = 0;
  tym::ModelFile file_;
  Buffer<int8_t> int8_;
  Buffer<float> scales_;
};

}  // namespace tinyyolo

#else
#error "ModelFilePartition.h is for the ESP32 only"
#endif
