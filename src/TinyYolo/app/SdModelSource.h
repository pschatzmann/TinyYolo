#pragma once
/// @file SdModelSource.h  A model read from the SD card: <base>.cfg, <base>.names, <base>.weights.
///
/// Requires the SD library. Include this header after <SD.h>.

#if !__has_include(<SD.h>)
#error "TinyYolo/app/SdModelSource.h needs the SD library: it ships with the Arduino cores, or remove this include."
#endif

#include <SD.h>
#include <stdio.h>

#include "TinyYolo/app/ModelSource.h"
#include "TinyYolo/core/Logger.h"

namespace tinyyolo {

class SdModelSource : public ModelSource {
 public:
  /// base: path without extension, e.g. "/model". chip_select: the SD card's chip-select pin.
  SdModelSource(const char* base, int chip_select) : base_(base), chip_select_(chip_select) {}

  bool open() override {
    if (!SD.begin(chip_select_)) {
      TINYYOLO_LOGE("SdModel", "SD card not found (chip select %d)", chip_select_);
      return false;
    }
    cfg_ = openWithSuffix(".cfg");
    names_ = openWithSuffix(".names");
    weights_ = openWithSuffix(".weights");
    if (!cfg_ || !names_ || !weights_) {
      TINYYOLO_LOGE("SdModel", "missing %s.cfg, .names or .weights", base_);
      close();
      return false;
    }
    return true;
  }

  void close() override {
    if (cfg_) cfg_.close();
    if (names_) names_.close();
    if (weights_) weights_.close();
  }

  Stream& cfg() override { return cfg_; }
  Stream& names() override { return names_; }
  Stream& weights() override { return weights_; }

 private:
  File openWithSuffix(const char* suffix) {
    char path[64];
    snprintf(path, sizeof(path), "%s%s", base_, suffix);
    return SD.open(path);
  }

  const char* base_;
  int chip_select_;
  File cfg_;
  File names_;
  File weights_;
};

}  // namespace tinyyolo
