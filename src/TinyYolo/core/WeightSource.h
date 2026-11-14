#pragma once
/// @file WeightSource.h  Where trained parameters come from.

#include <string.h>

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

/// Destination of one convolution's parameters.
struct ConvWeights {
  float* weights;       ///< OIHW, weight_count values
  size_t weight_count;
  float* bias;          ///< out_channels values
  size_t out_channels;
  bool batch_norm;      ///< the source stores batch-norm parameters (Darknet)
};

/// Supplies parameters in the order the layers request them.
/// Default format: little-endian float32, conv weights (OIHW) then conv bias, layers in
/// network order. Sources that store batch-norm override readConv() and fold it.
class WeightSource {
 public:
  virtual ~WeightSource() {}
  /// Reads exactly `count` values into `dst`.
  virtual Status read(float* dst, size_t count) = 0;

  /// Returns a view of one convolution's int8 weights, scales and biases without copying them
  /// (for read-only memory such as flash). Returns false if the source cannot do that; the
  /// source must then leave its position unchanged.
  /// Copies one convolution's int8 weights, scales and biases into the given arrays. Returns
  /// false if the source has no int8 data for it; the position must then stay unchanged.
  virtual bool readConvInt8(size_t out_channels, size_t weight_count, int8_t* weights,
                            float* scales, float* bias) {
    (void)out_channels;
    (void)weight_count;
    (void)weights;
    (void)scales;
    (void)bias;
    return false;
  }

  virtual bool viewConvInt8(size_t out_channels, size_t weight_count, const int8_t*& weights,
                            const float*& scales, const float*& bias) {
    (void)out_channels;
    (void)weight_count;
    (void)weights;
    (void)scales;
    (void)bias;
    return false;
  }

  /// Reads one convolution's parameters.
  virtual Status readConv(const ConvWeights& target) {
    TINYYOLO_CHECK(read(target.weights, target.weight_count));
    return read(target.bias, target.out_channels);
  }
};

/// Weights held in a RAM array (e.g. a generated header).
class ArrayWeightSource : public WeightSource {
 public:
  ArrayWeightSource(const float* values, size_t count) : values_(values), count_(count) {}

  Status read(float* dst, size_t count) override {
    if (dst == nullptr) {
      TINYYOLO_LOGE("Weights", "read: destination is null");
      return Status::InvalidArgument;
    }
    if (count > count_ - position_) {
      TINYYOLO_LOGE("Weights", "array exhausted: need %lu values, %lu left", (unsigned long)count,
                    (unsigned long)(count_ - position_));
      return Status::WeightError;
    }
    memcpy(dst, values_ + position_, count * sizeof(float));
    position_ += count;
    return Status::Ok;
  }

  size_t remaining() const { return count_ - position_; }

 private:
  const float* values_;
  size_t count_;
  size_t position_ = 0;
};

/// Weights read sequentially from any Arduino Stream (File on SD/LittleFS, Serial, ...).
class StreamWeightSource : public WeightSource {
 public:
  explicit StreamWeightSource(Stream& stream) : stream_(stream) {}

  Status read(float* dst, size_t count) override {
    if (dst == nullptr) {
      TINYYOLO_LOGE("Weights", "read: destination is null");
      return Status::InvalidArgument;
    }
    const size_t bytes = count * sizeof(float);
    const size_t got = stream_.readBytes((char*)dst, bytes);
    if (got != bytes) {
      TINYYOLO_LOGE("Weights", "stream ended after %lu of %lu bytes", (unsigned long)got,
                    (unsigned long)bytes);
      return Status::WeightError;
    }
    return Status::Ok;
  }

 private:
  Stream& stream_;
};

}  // namespace tinyyolo
