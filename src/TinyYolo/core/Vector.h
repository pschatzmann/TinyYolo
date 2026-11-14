#pragma once
/// @file Vector.h  Growable array for trivially copyable element types.

#include <stdlib.h>
#include <string.h>

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"

namespace tinyyolo {

template <typename T>
class Vector {
 public:
  Vector() {}
  ~Vector() { free(data_); }
  Vector(const Vector&) = delete;
  Vector& operator=(const Vector&) = delete;

  Status reserve(size_t capacity) {
    if (capacity <= capacity_) return Status::Ok;
    T* grown = static_cast<T*>(realloc(data_, capacity * sizeof(T)));
    if (grown == nullptr) {
      TINYYOLO_LOGE("Vector", "out of memory: %lu bytes requested", (unsigned long)(capacity * sizeof(T)));
      return Status::OutOfMemory;
    }
    data_ = grown;
    capacity_ = capacity;
    return Status::Ok;
  }

  Status push_back(const T& value) {
    if (size_ == capacity_) TINYYOLO_CHECK(reserve(capacity_ == 0 ? 8 : capacity_ * 2));
    data_[size_++] = value;
    return Status::Ok;
  }

  void clear() { size_ = 0; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }
  bool empty() const { return size_ == 0; }
  T* data() { return data_; }
  const T* data() const { return data_; }
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

 private:
  T* data_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

}  // namespace tinyyolo
