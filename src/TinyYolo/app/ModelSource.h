#pragma once
/// @file ModelSource.h  Where a Darknet model's three files come from: flash or the SD card.

#include <stddef.h>
#include <stdint.h>

#include "TinyYolo/core/BufferStream.h"
#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

/// Provides the .cfg, .names and .weights streams of one model. TinyYoloModel::begin() calls
/// open() before reading and close() after the model is loaded.
class ModelSource {
 public:
  virtual ~ModelSource() {}
  /// Prepares the streams. Returns false if they are not available.
  virtual bool open() = 0;
  /// Releases the streams.
  virtual void close() = 0;
  virtual Stream& cfg() = 0;
  virtual Stream& names() = 0;
  virtual Stream& weights() = 0;
};

/// Model embedded in the program image. The arrays come from tools/bin2header.py.
class FlashModelSource : public ModelSource {
 public:
  template <size_t CfgSize, size_t NamesSize, size_t WeightsSize>
  FlashModelSource(const uint8_t (&cfg)[CfgSize], const uint8_t (&names)[NamesSize],
                   const uint8_t (&weights)[WeightsSize])
      : cfg_(cfg, CfgSize), names_(names, NamesSize), weights_(weights, WeightsSize) {}

  bool open() override {
    cfg_.rewind();
    names_.rewind();
    weights_.rewind();
    return true;
  }
  void close() override {}
  Stream& cfg() override { return cfg_; }
  Stream& names() override { return names_; }
  Stream& weights() override { return weights_; }

 private:
  BufferStream cfg_;
  BufferStream names_;
  BufferStream weights_;
};

}  // namespace tinyyolo
