#pragma once
/// @file BufferStream.h  Read-only Stream over a byte array.
///
/// Use it for models embedded in the program image (flash), e.g.
///   static const uint8_t kWeights[] PROGMEM = {...};
///   BufferStream weights(kWeights, sizeof(kWeights));
/// On the boards TinyYolo supports (SAMD, RP2040, ESP32, nRF52) const data is read directly,
/// and PROGMEM is a no-op. The same class also works for RAM buffers.

#include <string.h>

#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

class BufferStream : public Stream {
 public:
  BufferStream(const void* data, size_t size)
      : data_(static_cast<const uint8_t*>(data)), size_(size) {}

  /// Copies up to `length` bytes from the current position.
  size_t readBytes(char* buffer, size_t length) {
    const size_t n = length < remaining() ? length : remaining();
    if (n > 0) memcpy(buffer, data_ + position_, n);
    position_ += n;
    return n;
  }

  // Stream-compatible members. Non-virtual on the generic platform, overrides on Arduino.
  int available() { return (int)remaining(); }
  int read() {
    if (remaining() == 0) return -1;
    return data_[position_++];
  }
  int peek() { return remaining() == 0 ? -1 : data_[position_]; }
  size_t write(uint8_t) { return 0; }

  size_t remaining() const { return size_ - position_; }
  size_t position() const { return position_; }
  void rewind() { position_ = 0; }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t position_ = 0;
};

}  // namespace tinyyolo
