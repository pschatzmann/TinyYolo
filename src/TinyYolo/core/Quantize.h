#pragma once
/// @file Quantize.h  Weight precision and symmetric int8 quantization.
///
/// Weights-only quantization: each output channel's weights become int8 values with one
/// float scale, w ~= q * scale. Activations stay float32, so the accuracy loss is limited
/// to the weights. Memory for weights drops to about a quarter.

#include <stddef.h>
#include <stdint.h>

namespace tinyyolo {

enum class WeightPrecision : uint8_t {
  Float32,  ///< weights stored as float32 (default)
  Int8      ///< weights stored as int8 with one float scale per output channel
};

/// Quantizes `rows` rows of `row_len` values each. Row r gets scale = max|w| / 127 and
/// q = round(w / scale) clamped to [-127, 127]. An all-zero row gets scale 1.
inline void quantizeRows(const float* w, size_t rows, size_t row_len, int8_t* q, float* scales) {
  for (size_t r = 0; r < rows; ++r) {
    const float* row = w + r * row_len;
    float max_abs = 0.0f;
    for (size_t i = 0; i < row_len; ++i) {
      const float a = row[i] < 0.0f ? -row[i] : row[i];
      if (a > max_abs) max_abs = a;
    }
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    scales[r] = scale;
    for (size_t i = 0; i < row_len; ++i) {
      const float v = row[i] / scale;
      int iv = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
      if (iv > 127) iv = 127;
      if (iv < -127) iv = -127;
      q[r * row_len + i] = (int8_t)iv;
    }
  }
}

}  // namespace tinyyolo
