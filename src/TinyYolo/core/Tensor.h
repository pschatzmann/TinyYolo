#pragma once
/// @file Tensor.h  Shape and float32 activation tensor (HWC layout).

#include <utility>

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"

namespace tinyyolo {

/// Height x width x channels.
struct Shape {
  int h;
  int w;
  int c;

  Shape() : h(0), w(0), c(0) {}
  Shape(int height, int width, int channels) : h(height), w(width), c(channels) {}

  bool isValid() const { return h > 0 && w > 0 && c > 0; }
  size_t elements() const { return (size_t)h * (size_t)w * (size_t)c; }
  bool operator==(const Shape& o) const { return h == o.h && w == o.w && c == o.c; }
  bool operator!=(const Shape& o) const { return !(*this == o); }
};

/// Dense float32 tensor in HWC layout (channel is the fastest-varying index). Either owns
/// its memory (allocate) or is a view into memory owned elsewhere (attachView).
class Tensor {
 public:
  Tensor() {}
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept
      : shape_(other.shape_), data_(other.data_), owned_(std::move(other.owned_)) {
    other.shape_ = Shape();
    other.data_ = nullptr;
  }
  Tensor& operator=(Tensor&& other) noexcept {
    if (this != &other) {
      shape_ = other.shape_;
      data_ = other.data_;
      owned_ = std::move(other.owned_);
      other.shape_ = Shape();
      other.data_ = nullptr;
    }
    return *this;
  }

  /// Owns a new zero-initialised buffer of the given shape.
  Status allocate(const Shape& shape) {
    if (!shape.isValid()) {
      TINYYOLO_LOGE("Tensor", "invalid shape %dx%dx%d", shape.h, shape.w, shape.c);
      return Status::InvalidArgument;
    }
    TINYYOLO_CHECK(owned_.allocate(shape.elements()));
    shape_ = shape;
    data_ = owned_.data();
    return Status::Ok;
  }

  /// Makes this tensor a view of `storage` (shape.elements() floats, not owned). Any owned
  /// memory is released. The storage must outlive the view.
  void attachView(float* storage, const Shape& shape) {
    owned_.release();
    shape_ = shape;
    data_ = storage;
  }

  /// Frees the memory of an owned tensor (or drops a view). The tensor becomes empty.
  void release() {
    owned_.release();
    shape_ = Shape();
    data_ = nullptr;
  }

  const Shape& shape() const { return shape_; }
  size_t size() const { return data_ != nullptr ? shape_.elements() : 0; }
  float* data() { return data_; }
  const float* data() const { return data_; }

  float& at(int y, int x, int c) { return data_[index(y, x, c)]; }
  const float& at(int y, int x, int c) const { return data_[index(y, x, c)]; }

  void fill(float value) {
    for (size_t i = 0; i < size(); ++i) data_[i] = value;
  }

 private:
  size_t index(int y, int x, int c) const {
    return ((size_t)y * (size_t)shape_.w + (size_t)x) * (size_t)shape_.c + (size_t)c;
  }

  Shape shape_;
  float* data_ = nullptr;
  Buffer<float> owned_;
};

}  // namespace tinyyolo
