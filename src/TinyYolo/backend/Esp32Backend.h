#pragma once
/// @file Esp32Backend.h  DotProductBackend with esp-dsp SIMD dot products and both cores.
///
/// esp-dsp ships with the ESP32 Arduino core (3.x), so nothing has to be installed. Without
/// it the backend uses the scalar dot product and still runs.
///
/// Note: dsps_dotprod_f32() overwrites its destination (it does not accumulate), so the
/// result goes into a fresh variable and is added afterwards.

#include "TinyYolo/backend/DotProductBackend.h"
#include "TinyYolo/platform/Features.h"

#if defined(TINYYOLO_HAS_ESP_DSP)
#include <dsps_dotprod.h>
#endif

namespace tinyyolo {

class Esp32Backend : public DotProductBackend {
 public:
  /// workers = 0 uses both cores (2).
  explicit Esp32Backend(int workers = 0) : DotProductBackend(workers) {}

  const char* name() const override {
#if defined(TINYYOLO_HAS_ESP_DSP)
    return "esp32-dsp";
#else
    return "esp32-scalar";
#endif
  }

 protected:
  float dot(const float* a, const float* b, int n) const override {
#if defined(TINYYOLO_HAS_ESP_DSP)
    float result = 0.0f;
    dsps_dotprod_f32(a, b, &result, n);
    return result;
#else
    return DotProductBackend::dot(a, b, n);
#endif
  }
};

}  // namespace tinyyolo
