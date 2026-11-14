#pragma once
/// @file ModelBlob.h  Model parameters in one binary blob that can stay in read-only memory
/// (flash). The layout is little-endian:
///
///   magic "TYQ1", uint32 conv_count
///   per convolution (in network order):
///     uint32 kind (0 = float32, 1 = int8), uint32 out_channels, uint32 weight_count
///     kind 0: weight_count float32 weights, out_channels float32 biases
///     kind 1: weight_count int8 weights, zero padding to a multiple of 4 bytes,
///             out_channels float32 scales, out_channels float32 biases
///
/// The blob must start on a 4-byte boundary. Writer: writeModelBlob() (any byte sink).
/// Reader: ModelBlobSource (a WeightSource that can hand out int8 weights without copying).

#include <string.h>

#include "TinyYolo/core/WeightSource.h"
#include "TinyYolo/model/Model.h"

namespace tinyyolo {

namespace blob {

constexpr uint32_t kKindFloat = 0;
constexpr uint32_t kKindInt8 = 1;

inline size_t align4(size_t n) { return (n + 3) & ~(size_t)3; }

/// Bytes of the weights part of a record.
inline size_t weightsBytes(uint32_t kind, size_t weight_count) {
  return kind == kKindInt8 ? align4(weight_count) : weight_count * sizeof(float);
}

}  // namespace blob

/// Writes the blob of a loaded model. `Sink` has `size_t write(const void* data, size_t bytes)`.
/// Float32 and int8 convolutions are both written; the model decides per layer.
template <typename Sink>
Status writeModelBlob(const Model& model, Sink& sink) {
  uint32_t conv_count = 0;
  for (int i = 0; i < model.layerCount(); ++i) {
    if (model.layer(i).asConv() != nullptr) ++conv_count;
  }
  const char magic[4] = {'T', 'Y', 'Q', '1'};
  sink.write(magic, 4);
  sink.write(&conv_count, 4);

  for (int i = 0; i < model.layerCount(); ++i) {
    const ConvLayer* conv = model.layer(i).asConv();
    if (conv == nullptr) continue;
    const bool int8 = conv->precision() == WeightPrecision::Int8;
    if (int8 && conv->int8Weights() == nullptr) {
      TINYYOLO_LOGE("Blob", "layer %d has no weights; load the model first", i);
      return Status::NotInitialized;
    }
    if (!int8 && conv->floatWeights() == nullptr) {
      TINYYOLO_LOGE("Blob", "layer %d has no weights; load the model first", i);
      return Status::NotInitialized;
    }
    const uint32_t kind = int8 ? blob::kKindInt8 : blob::kKindFloat;
    const uint32_t out = (uint32_t)conv->outChannels();
    const uint32_t wc = (uint32_t)conv->weightCount();
    sink.write(&kind, 4);
    sink.write(&out, 4);
    sink.write(&wc, 4);
    if (int8) {
      sink.write(conv->int8Weights(), wc);
      const uint8_t zero[4] = {0, 0, 0, 0};
      const size_t pad = blob::align4(wc) - wc;
      if (pad > 0) sink.write(zero, pad);
      sink.write(conv->scales(), out * sizeof(float));
    } else {
      sink.write(conv->floatWeights(), (size_t)wc * sizeof(float));
    }
    sink.write(conv->bias(), out * sizeof(float));
  }
  return Status::Ok;
}

/// Reads a blob written by writeModelBlob(). The convolutions must be loaded in the same order
/// as they were written. int8 records are handed out as views when the model is int8; a float
/// model reads them dequantized.
class ModelBlobSource : public WeightSource {
 public:
  /// `blob` must be 4-byte aligned and must outlive the model built from it.
  ModelBlobSource(const uint8_t* blob, size_t size) : blob_(blob), size_(size) {}

  /// Checks the header. Call once before the model is loaded.
  Status open() {
    if (blob_ == nullptr || size_ < 8 || memcmp(blob_, "TYQ1", 4) != 0) {
      TINYYOLO_LOGE("Blob", "not a model blob (missing TYQ1 header)");
      return Status::WeightError;
    }
    memcpy(&conv_count_, blob_ + 4, 4);
    position_ = 8;
    return Status::Ok;
  }

  /// Copy mode: int8 convolutions are copied into RAM instead of being read in place. Reading
  /// in place goes through the flash cache, which is slow for large filters on the ESP32-P4.
  void setCopyWeights(bool copy) { copy_ = copy; }

  bool readConvInt8(size_t out_channels, size_t weight_count, int8_t* weights, float* scales,
                    float* bias) override {
    Record r;
    if (!peek(r) || r.kind != blob::kKindInt8 || r.out != out_channels || r.weight_count != weight_count) {
      return false;
    }
    const uint8_t* w = blob_ + r.payload;
    const uint8_t* s = blob_ + r.payload + blob::align4(r.weight_count);
    memcpy(weights, w, r.weight_count);
    memcpy(scales, s, r.out * sizeof(float));
    memcpy(bias, s + r.out * sizeof(float), r.out * sizeof(float));
    position_ = r.end;
    return true;
  }

  bool viewConvInt8(size_t out_channels, size_t weight_count, const int8_t*& weights,
                    const float*& scales, const float*& bias) override {
    Record r;
    if (copy_ || !peek(r) || r.kind != blob::kKindInt8 || r.out != out_channels || r.weight_count != weight_count) {
      return false;
    }
    weights = reinterpret_cast<const int8_t*>(blob_ + r.payload);
    scales = reinterpret_cast<const float*>(blob_ + r.payload + blob::align4(r.weight_count));
    bias = reinterpret_cast<const float*>(blob_ + r.payload + blob::align4(r.weight_count) +
                                          r.out * sizeof(float));
    position_ = r.end;
    return true;
  }

  Status readConv(const ConvWeights& target) override {
    Record r;
    if (!peek(r)) return Status::WeightError;
    if (r.out != target.out_channels || r.weight_count != target.weight_count) {
      TINYYOLO_LOGE("Blob", "record is %lu x %lu weights, the layer expects %lu x %lu",
                    (unsigned long)r.out, (unsigned long)r.weight_count,
                    (unsigned long)target.out_channels, (unsigned long)target.weight_count);
      return Status::WeightError;
    }
    if (r.kind == blob::kKindFloat) {
      memcpy(target.weights, blob_ + r.payload, r.weight_count * sizeof(float));
    } else {
      // Dequantize: w = q * scale, per output channel.
      const int8_t* q = reinterpret_cast<const int8_t*>(blob_ + r.payload);
      const float* scales = reinterpret_cast<const float*>(blob_ + r.payload + blob::align4(r.weight_count));
      const size_t per_output = r.weight_count / r.out;
      for (size_t o = 0; o < r.out; ++o) {
        for (size_t i = 0; i < per_output; ++i) {
          target.weights[o * per_output + i] = q[o * per_output + i] * scales[o];
        }
      }
    }
    const size_t bias_at = r.payload + blob::weightsBytes(r.kind, r.weight_count) +
                           (r.kind == blob::kKindInt8 ? r.out * sizeof(float) : 0);
    memcpy(target.bias, blob_ + bias_at, r.out * sizeof(float));
    position_ = r.end;
    return Status::Ok;
  }

  Status read(float*, size_t) override {
    TINYYOLO_LOGE("Blob", "raw reads are not supported; the blob holds convolution records");
    return Status::Unsupported;
  }

 private:
  struct Record {
    uint32_t kind = 0;
    uint32_t out = 0;
    uint32_t weight_count = 0;
    size_t payload = 0;  // offset of the weights
    size_t end = 0;      // offset just past the record
  };

  /// Reads the record at the current position. Fails if there is none left or it is damaged.
  bool peek(Record& r) const {
    if (position_ + 12 > size_) return false;
    memcpy(&r.kind, blob_ + position_, 4);
    memcpy(&r.out, blob_ + position_ + 4, 4);
    memcpy(&r.weight_count, blob_ + position_ + 8, 4);
    if (r.kind > blob::kKindInt8 || r.out == 0 || r.weight_count == 0) return false;
    r.payload = position_ + 12;
    r.end = r.payload + blob::weightsBytes(r.kind, r.weight_count) +
            (r.kind == blob::kKindInt8 ? r.out * sizeof(float) : 0) + r.out * sizeof(float);
    return r.end <= size_;
  }

  bool copy_ = false;
  const uint8_t* blob_;
  size_t size_;
  size_t position_ = 0;
  uint32_t conv_count_ = 0;
};

}  // namespace tinyyolo
