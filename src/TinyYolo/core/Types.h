#pragma once
/// @file Types.h  Shared enums and small math helpers.

#include <math.h>
#include <stdint.h>

namespace tinyyolo {

/// Activation applied to the output of a convolution.
enum class Activation : uint8_t { Linear = 0, Leaky, ReLU, SiLU, Mish };

inline float activate(Activation activation, float x) {
  switch (activation) {
    case Activation::Leaky: return x > 0.0f ? x : 0.1f * x;
    case Activation::ReLU: return x > 0.0f ? x : 0.0f;
    case Activation::SiLU: return x / (1.0f + (float)exp(-x));
    case Activation::Mish:
      // x * tanh(softplus(x)); softplus is taken as x for large x to avoid overflow.
      return x > 20.0f ? x : x * (float)tanh(log(1.0f + (float)exp(x)));
    case Activation::Linear: break;
  }
  return x;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + (float)exp(-x)); }

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

/// Output positions along one axis of a sliding window, or -1 for invalid parameters.
inline int windowOutputSize(int input, int kernel, int stride, int pad) {
  if (input <= 0 || kernel <= 0 || stride <= 0 || pad < 0) return -1;
  const int span = input + 2 * pad - kernel;
  if (span < 0) return -1;
  return span / stride + 1;
}

/// Output positions of a pooling window with `pad_total` padding split across both sides
/// (Darknet convention), or -1 for invalid parameters.
inline int poolOutputSize(int input, int kernel, int stride, int pad_total) {
  if (input <= 0 || kernel <= 0 || stride <= 0 || pad_total < 0) return -1;
  const int span = input + pad_total - kernel;
  if (span < 0) return -1;
  return span / stride + 1;
}

}  // namespace tinyyolo
