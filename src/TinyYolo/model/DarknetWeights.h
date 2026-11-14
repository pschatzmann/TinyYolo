#pragma once
/// @file DarknetWeights.h  Reads Darknet .weights files (YOLOv2/v3/v4, tiny variants).

#include <math.h>

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/WeightSource.h"
#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

/// Darknet weight file reader. Stream-based: works with File (SD card), BufferStream
/// (flash) or any other Stream.
///
/// Layout: a header (major, minor, revision, "seen"), then for every convolution either
/// [bias] [weights] or, with batch-norm, [beta] [gamma] [mean] [variance] [weights].
/// Batch-norm is folded into the weights and bias while reading.
class DarknetWeightSource : public WeightSource {
 public:
  explicit DarknetWeightSource(Stream& stream) : stream_(stream) {}

  /// Reads and checks the header. Call once before the network is loaded.
  Status open() {
    uint32_t head[3];
    TINYYOLO_CHECK(readRaw(head, sizeof(head)));
    const uint32_t major = head[0];
    const uint32_t minor = head[1];
    // Darknet stores "seen" as 64-bit from version 0.2 on, and as 32-bit before.
    if (major * 10 + minor >= 2 && major < 1000 && minor < 1000) {
      uint64_t seen;
      TINYYOLO_CHECK(readRaw(&seen, sizeof(seen)));
    } else {
      uint32_t seen;
      TINYYOLO_CHECK(readRaw(&seen, sizeof(seen)));
    }
    if (major > 1000 || minor > 1000) {
      TINYYOLO_LOGE("Darknet", "unrecognised weight file version %lu.%lu", (unsigned long)major,
                    (unsigned long)minor);
      return Status::WeightError;
    }
    TINYYOLO_LOGI("Darknet", "weights file version %lu.%lu", (unsigned long)major,
                  (unsigned long)minor);
    return Status::Ok;
  }

  /// Plain float read (used for layers without batch-norm handling).
  Status read(float* dst, size_t count) override { return readFloats(dst, count); }

  Status readConv(const ConvWeights& target) override {
    const size_t out = target.out_channels;
    if (!target.batch_norm) {
      TINYYOLO_CHECK(readFloats(target.bias, out));
      return readFloats(target.weights, target.weight_count);
    }

    // beta is read straight into the bias; gamma, mean and variance go to a temporary buffer.
    Buffer<float> stats;
    TINYYOLO_CHECK(stats.allocate(3 * out));
    float* gamma = stats.data();
    float* mean = gamma + out;
    float* variance = mean + out;
    TINYYOLO_CHECK(readFloats(target.bias, out));
    TINYYOLO_CHECK(readFloats(gamma, out));
    TINYYOLO_CHECK(readFloats(mean, out));
    TINYYOLO_CHECK(readFloats(variance, out));
    TINYYOLO_CHECK(readFloats(target.weights, target.weight_count));

    // Fold: y = gamma * (x - mean) / sqrt(var + eps) + beta  ==  s * x + (beta - s * mean)
    const size_t per_output = target.weight_count / out;
    for (size_t o = 0; o < out; ++o) {
      const float s = gamma[o] / (float)sqrt(variance[o] + 1e-5f);
      target.bias[o] = target.bias[o] - s * mean[o];
      float* w = target.weights + o * per_output;
      for (size_t i = 0; i < per_output; ++i) w[i] *= s;
    }
    return Status::Ok;
  }

  /// Call after the whole network has been loaded. Warns if the file has unread data,
  /// which usually means the .cfg does not match the .weights file.
  Status finish() {
    char extra;
    if (stream_.readBytes(&extra, 1) == 1) {
      TINYYOLO_LOGW("Darknet", "weight file has unread data; the .cfg may not match the .weights");
    }
    return Status::Ok;
  }

 private:
  Status readFloats(float* dst, size_t count) { return readRaw(dst, count * sizeof(float)); }

  Status readRaw(void* dst, size_t bytes) {
    if (bytes == 0) return Status::Ok;
    const size_t got = stream_.readBytes(static_cast<char*>(dst), bytes);
    if (got != bytes) {
      TINYYOLO_LOGE("Darknet", "weight stream ended after %lu of %lu bytes", (unsigned long)got,
                    (unsigned long)bytes);
      return Status::WeightError;
    }
    return Status::Ok;
  }

  Stream& stream_;
};

}  // namespace tinyyolo
