#pragma once
/// @file Io.h  Output and input interfaces used by the logger and weight sources.
///
/// On Arduino builds, tinyyolo::Print and tinyyolo::Stream are aliases of the core's
/// Print and Stream, so Serial, File, SD and WiFi clients can be passed directly.
/// On every other platform (desktop, generic C++ targets) the library provides its own
/// classes with the same member API.

#if defined(ARDUINO)

#include <Arduino.h>

namespace tinyyolo {
using Print = ::Print;
using Stream = ::Stream;
}  // namespace tinyyolo

#else

#include <stddef.h>
#include <stdint.h>

namespace tinyyolo {

/// Generic byte sink with the Arduino Print API (println appends "\r\n").
class Print {
 public:
  virtual ~Print() {}
  virtual size_t write(uint8_t c) = 0;
  virtual void flush() {}

  size_t print(const char* s) {
    size_t n = 0;
    if (s != nullptr) {
      for (; *s != 0; ++s) n += write((uint8_t)*s);
    }
    return n;
  }

  size_t println(const char* s) {
    size_t n = print(s);
    n += write('\r');
    n += write('\n');
    return n;
  }
};

/// Generic byte source with the Arduino Stream API.
class Stream : public Print {
 public:
  virtual size_t readBytes(char* buffer, size_t length) = 0;
};

}  // namespace tinyyolo

#endif
