#pragma once
/// @file Clock.h  A monotonic microsecond clock for profiling (ESP32 and hosts).

#include <stdint.h>

#if defined(ESP32)
#include <esp_timer.h>
#else
#include <chrono>
#endif

namespace tinyyolo {

/// Microseconds since an arbitrary start (wraps after about 71 minutes as a uint32).
inline uint32_t nowMicros() {
#if defined(ESP32)
  return (uint32_t)esp_timer_get_time();
#else
  return (uint32_t)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
#endif
}

}  // namespace tinyyolo
