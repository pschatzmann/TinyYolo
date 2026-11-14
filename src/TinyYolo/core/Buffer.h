#pragma once
/// @file Buffer.h  Owning fixed-size array with explicit allocation.

#include <stdlib.h>

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/platform/Memory.h"

namespace tinyyolo {

/// Owning array. Only for trivially constructible element types (float, uint8_t,
/// int, plain structs). Memory is obtained with calloc so it works on every core.
template <typename T>
class Buffer {
 public:
  Buffer() {}
  ~Buffer() { release(); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  Buffer(Buffer&& other) noexcept
      : data_(other.data_), size_(other.size_), in_psram_(other.in_psram_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.in_psram_ = false;
  }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      size_ = other.size_;
      in_psram_ = other.in_psram_;
      other.data_ = nullptr;
      other.size_ = 0;
      other.in_psram_ = false;
    }
    return *this;
  }

  /// Replaces the content with `count` zero-initialised elements. Large buffers may be
  /// placed in PSRAM (see platform/Memory.h).
  Status allocate(size_t count) {
    return allocate(count, memoryPlacement());
  }

  /// Same, with the placement given for this buffer (PsramFirst for large, rarely used-hot data).
  Status allocate(size_t count, MemoryPlacement first) {
    release();
    if (count == 0) return Status::Ok;
    data_ = static_cast<T*>(allocateMemory(count * sizeof(T), true, in_psram_, first));
    if (data_ == nullptr) {
      TINYYOLO_LOGE("Buffer", "out of memory: %lu bytes requested",
                    (unsigned long)(count * sizeof(T)));
      return Status::OutOfMemory;
    }
    size_ = count;
    return Status::Ok;
  }

  void release() {
    releaseMemory(data_, in_psram_);
    data_ = nullptr;
    size_ = 0;
    in_psram_ = false;
  }

  T* data() { return data_; }
  const T* data() const { return data_; }
  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
  bool in_psram_ = false;
};

}  // namespace tinyyolo
