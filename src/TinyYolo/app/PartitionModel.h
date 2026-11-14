#pragma once
/// @file PartitionModel.h  A model blob in a flash data partition, read in place (ESP32 only).
///
/// The blob holds the weights (see ModelBlob.h, written with tinyyolo_export_blob and esptool).
/// The network description (cfg and names) comes from program memory, as in
/// examples/Yolov4CameraStream/Yolov4Model.h. The partition is mapped on begin() and unmapped when
/// this object is destroyed, so keep it alive while the detector is in use.

#include "TinyYolo/Detector.h"
#include "TinyYolo/core/BufferStream.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/model/ModelBlob.h"
#include "TinyYolo/model/NetworkDescription.h"
#include "TinyYolo/platform/Features.h"

#if defined(ESP32)
#include "esp_partition.h"

namespace tinyyolo {

class PartitionModel {
 public:
  PartitionModel() {}
  ~PartitionModel() {
    if (handle_ != 0) esp_partition_munmap(handle_);
  }
  PartitionModel(const PartitionModel&) = delete;
  PartitionModel& operator=(const PartitionModel&) = delete;

  /// Maps the data partition `label`, reads its blob and builds `detector` from it. `cfg` and
  /// `names` are the network description and the class names (text in program memory).
  /// `copy_weights` copies the int8 weights into RAM (PSRAM first) instead of reading them
  /// through the flash cache, which is slower for large filters.
  Status begin(Detector& detector, const void* cfg, size_t cfg_size, const void* names,
               size_t names_size, const DetectorConfig& config, Backend* backend,
               const char* label = "model", bool copy_weights = false) {
    const esp_partition_t* part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, label);
    if (part == nullptr) {
      TINYYOLO_LOGE("PartitionModel", "no '%s' data partition: flash the weights first", label);
      return Status::NotInitialized;
    }
    // The flash cache maps only the first 16 MB of flash (the 24-bit window): the mapping stops there,
    // so a partition that starts low but is larger than that is mapped up to the window's end.
    constexpr size_t kFlashWindowEnd = 0x1000000;
    size_t map_bytes = part->size;
    if (part->address < kFlashWindowEnd && part->address + map_bytes > kFlashWindowEnd) {
      map_bytes = kFlashWindowEnd - part->address;
    }
    const void* mapped = nullptr;
    if (esp_partition_mmap(part, 0, map_bytes, ESP_PARTITION_MMAP_DATA, &mapped, &handle_) != ESP_OK) {
      TINYYOLO_LOGE("PartitionModel", "cannot map the '%s' partition", label);
      handle_ = 0;
      return Status::WeightError;
    }
    weights_ = ModelBlobSource(static_cast<const uint8_t*>(mapped), map_bytes);
    if (copy_weights) weights_.setCopyWeights(true);
    TINYYOLO_CHECK(weights_.open());

    BufferStream cfg_stream(cfg, cfg_size);
    BufferStream names_stream(names, names_size);
    TINYYOLO_CHECK(network_.parseCfg(cfg_stream));
    TINYYOLO_CHECK(network_.parseNames(names_stream));
    return detector.begin(network_.spec(), weights_, config, backend);
  }

 private:
  esp_partition_mmap_handle_t handle_ = 0;
  ModelBlobSource weights_{nullptr, 0};
  NetworkDescription network_;
};

}  // namespace tinyyolo

#endif  // ESP32
