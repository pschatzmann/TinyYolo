#pragma once
/// @file Detection.h  Detection result type and a fixed-capacity container for results.

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"

namespace tinyyolo {

struct Detection {
  float x;       ///< left edge, source-image pixels
  float y;       ///< top edge, source-image pixels
  float width;   ///< box width, source-image pixels
  float height;  ///< box height, source-image pixels
  float score;   ///< objectness * class probability
  int class_id;
};

inline float intersectionOverUnion(const Detection& a, const Detection& b) {
  const float ix1 = a.x > b.x ? a.x : b.x;
  const float iy1 = a.y > b.y ? a.y : b.y;
  const float ax2 = a.x + a.width, bx2 = b.x + b.width;
  const float ay2 = a.y + a.height, by2 = b.y + b.height;
  const float ix2 = ax2 < bx2 ? ax2 : bx2;
  const float iy2 = ay2 < by2 ? ay2 : by2;
  const float iw = ix2 - ix1;
  const float ih = iy2 - iy1;
  if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
  const float inter = iw * ih;
  const float uni = a.width * a.height + b.width * b.height - inter;
  return uni > 0.0f ? inter / uni : 0.0f;
}

/// Fixed-capacity list of detections. Never reallocates; push() reports overflow.
class Detections {
 public:
  Detections() {}

  Status reserve(int capacity) {
    if (capacity <= 0) {
      TINYYOLO_LOGE("Detections", "capacity must be positive, got %d", capacity);
      return Status::InvalidArgument;
    }
    TINYYOLO_CHECK(items_.allocate((size_t)capacity));
    size_ = 0;
    return Status::Ok;
  }

  void clear() { size_ = 0; }

  /// Returns false when the container is full.
  bool push(const Detection& d) {
    if (size_ >= items_.size()) return false;
    items_[size_++] = d;
    return true;
  }

  int size() const { return (int)size_; }
  int capacity() const { return (int)items_.size(); }
  const Detection& operator[](int i) const { return items_[(size_t)i]; }
  Detection& operator[](int i) { return items_[(size_t)i]; }
  const Detection* begin() const { return items_.data(); }
  const Detection* end() const { return items_.data() + size_; }

 private:
  Buffer<Detection> items_;
  size_t size_ = 0;
};

}  // namespace tinyyolo
