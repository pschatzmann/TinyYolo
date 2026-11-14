#pragma once
/// @file QTensor.h  int8 activation tensor (HWC layout) with one symmetric float scale.
///
/// A value v is stored as q = round(v / scale), clamped to [-127, 127], and read back as
/// v ~= q * scale. Quantized activations let the convolution run on integer dot products
/// (int8 x int8 accumulated in int32), which is what SIMD kernels such as ESP-NN need.

#include "TinyYolo/platform/KernelOptimize.h"
#include <utility>

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Tensor.h"

namespace tinyyolo {

/// Rounds v * inv_scale to the nearest int8 value (symmetric, clamped to +-127).
inline int8_t quantizeValue(float v, float inv_scale) {
  const float q = v * inv_scale;
  int i = (int)(q >= 0.0f ? q + 0.5f : q - 0.5f);
  if (i > 127) i = 127;
  if (i < -127) i = -127;
  return (int8_t)i;
}

/// Dense int8 tensor in HWC layout with a single scale for all values.
class QTensor {
 public:
  QTensor() {}
  QTensor(const QTensor&) = delete;
  QTensor& operator=(const QTensor&) = delete;
  QTensor(QTensor&& other) noexcept
      : shape_(other.shape_), scale_(other.scale_), data_(other.data_), owned_(std::move(other.owned_)) {
    other.shape_ = Shape();
    other.data_ = nullptr;
  }
  QTensor& operator=(QTensor&& other) noexcept {
    if (this != &other) {
      shape_ = other.shape_;
      scale_ = other.scale_;
      data_ = other.data_;
      owned_ = std::move(other.owned_);
      other.shape_ = Shape();
      other.data_ = nullptr;
    }
    return *this;
  }

  /// Owns a new zero-initialised buffer. `scale` must be positive. `placement` chooses internal
  /// RAM or PSRAM first (large activations are best kept in PSRAM).
  Status allocate(const Shape& shape, float scale, MemoryPlacement placement = MemoryPlacement::InternalFirst) {
    if (!shape.isValid()) {
      TINYYOLO_LOGE("QTensor", "invalid shape %dx%dx%d", shape.h, shape.w, shape.c);
      return Status::InvalidArgument;
    }
    if (!(scale > 0.0f)) {
      TINYYOLO_LOGE("QTensor", "scale must be positive");
      return Status::InvalidArgument;
    }
    // 16 spare bytes, so that the data can start on a 16-byte boundary (the P4 kernels need it;
    // PSRAM allocations are only 4- or 8-byte aligned).
    TINYYOLO_CHECK(owned_.allocate(shape.elements() + 16, placement));
    uint8_t* base = (uint8_t*)owned_.data();
    shape_ = shape;
    scale_ = scale;
    data_ = (int8_t*)(base + ((16 - ((uintptr_t)base & 15)) & 15));
    return Status::Ok;
  }

  /// Frees the data (the tensor becomes empty).
  void release() {
    owned_.release();
    data_ = nullptr;
  }

  /// Makes this tensor a view of `storage` (shape.elements() values, not owned). The storage
  /// must outlive the view. Used to share one buffer between several layers.
  void attachView(int8_t* storage, const Shape& shape, float scale) {
    owned_.release();
    shape_ = shape;
    scale_ = scale;
    data_ = storage;
  }

  const Shape& shape() const { return shape_; }
  float scale() const { return scale_; }
  void setScale(float scale) { scale_ = scale; }
  size_t size() const { return data_ != nullptr ? shape_.elements() : 0; }
  int8_t* data() { return data_; }
  const int8_t* data() const { return data_; }

  int8_t& at(int y, int x, int c) { return data_[index(y, x, c)]; }
  const int8_t& at(int y, int x, int c) const { return data_[index(y, x, c)]; }

 private:
  size_t index(int y, int x, int c) const {
    return ((size_t)y * (size_t)shape_.w + (size_t)x) * (size_t)shape_.c + (size_t)c;
  }

  Shape shape_;
  float scale_ = 1.0f;
  int8_t* data_ = nullptr;
  Buffer<int8_t> owned_;
};

/// Largest absolute value in a float tensor (0 for an empty one).
inline float maxAbsValue(const Tensor& t) {
  float m = 0.0f;
  const float* d = t.data();
  for (size_t i = 0; i < t.size(); ++i) {
    const float a = d[i] < 0.0f ? -d[i] : d[i];
    if (a > m) m = a;
  }
  return m;
}

/// Quantizes `in` into `out` with `scale`. `out` must have the same shape as `in`.
inline Status quantizeTensor(const Tensor& in, float scale, QTensor& out) {
  if (out.shape() != in.shape() || !(scale > 0.0f)) {
    TINYYOLO_LOGE("QTensor", "quantize: shape or scale mismatch");
    return Status::ShapeMismatch;
  }
  out.setScale(scale);
  const float inv = 1.0f / scale;
  const float* src = in.data();
  int8_t* dst = out.data();
  for (size_t i = 0; i < in.size(); ++i) dst[i] = quantizeValue(src[i], inv);
  return Status::Ok;
}

/// Converts `in` back to float32 into `out` (same shape required).
inline Status dequantizeTensor(const QTensor& in, Tensor& out) {
  if (out.shape() != in.shape()) {
    TINYYOLO_LOGE("QTensor", "dequantize: shape mismatch");
    return Status::ShapeMismatch;
  }
  const float s = in.scale();
  const int8_t* src = in.data();
  float* dst = out.data();
  for (size_t i = 0; i < in.size(); ++i) dst[i] = (float)src[i] * s;
  return Status::Ok;
}

}  // namespace tinyyolo
