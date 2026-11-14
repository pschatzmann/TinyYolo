#pragma once
/// @file Backend.h  Strategy interface for the compute kernels.

#include "TinyYolo/platform/KernelOptimize.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/QTensor.h"
#include "TinyYolo/core/Quantize.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/core/Types.h"

namespace tinyyolo {

/// Parameters of one convolution.
struct ConvParams {
  int kernel;
  int stride;
  int pad;
  int out_channels;
  Activation activation;
};

/// Portable int8 convolution: int8 x int8 products summed in int32, dequantized with
/// in.scale() * w_scales[o], biased, activated in float and requantized with out.scale().
/// Backends that have an integer kernel override Backend::conv2dQ() and may use this as
/// their fallback for shapes they do not support.
inline Status referenceConv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales,
                               const float* bias, const ConvParams& p, QTensor& out) {
  const Shape is = in.shape();
  const Shape os = out.shape();
  if (weights == nullptr || w_scales == nullptr || bias == nullptr || p.out_channels <= 0) {
    TINYYOLO_LOGE("Reference", "conv2dQ: missing weights, scales, bias or channel count");
    return Status::InvalidArgument;
  }
  if (!(out.scale() > 0.0f)) {
    TINYYOLO_LOGE("Reference", "conv2dQ: output scale must be positive");
    return Status::InvalidArgument;
  }
  if (os.c != p.out_channels || os.h != windowOutputSize(is.h, p.kernel, p.stride, p.pad) ||
      os.w != windowOutputSize(is.w, p.kernel, p.stride, p.pad)) {
    TINYYOLO_LOGE("Reference", "conv2dQ: output %dx%dx%d does not match input %dx%dx%d", os.h,
                  os.w, os.c, is.h, is.w, is.c);
    return Status::ShapeMismatch;
  }
  const int K = p.kernel;
  const int S = p.stride;
  const int P = p.pad;
  const int IC = is.c;
  const float inv_out = 1.0f / out.scale();
  for (int oc = 0; oc < os.c; ++oc) {
    const int8_t* w_oc = weights + (size_t)oc * K * K * IC;  // OHWI: [oc][ky][kx][ic]
    const float dequant = in.scale() * w_scales[oc];
    for (int oy = 0; oy < os.h; ++oy) {
      for (int ox = 0; ox < os.w; ++ox) {
        int32_t acc = 0;
        const int y0 = oy * S - P;
        const int x0 = ox * S - P;
        for (int ic = 0; ic < IC; ++ic) {
          for (int ky = 0; ky < K; ++ky) {
            const int iy = y0 + ky;
            if (iy < 0 || iy >= is.h) continue;
            for (int kx = 0; kx < K; ++kx) {
              const int ix = x0 + kx;
              if (ix < 0 || ix >= is.w) continue;
              acc += (int32_t)in.at(iy, ix, ic) * (int32_t)w_oc[(ky * K + kx) * IC + ic];
            }
          }
        }
        const float real = activate(p.activation, bias[oc] + (float)acc * dequant);
        out.at(oy, ox, oc) = quantizeValue(real, inv_out);
      }
    }
  }
  return Status::Ok;
}

/// Portable int8 depthwise convolution (one 3x3 or KxK filter per channel). `weights` is [C][K][K],
/// `w_scales` one scale per channel. p.out_channels == C.
inline Status referenceDepthwiseConv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales,
                                        const float* bias, const ConvParams& p, QTensor& out) {
  const Shape is = in.shape();
  const Shape os = out.shape();
  if (weights == nullptr || w_scales == nullptr || bias == nullptr || p.out_channels != is.c) {
    TINYYOLO_LOGE("Reference", "depthwise conv2dQ: missing weights, scales, bias or channel mismatch");
    return Status::InvalidArgument;
  }
  if (!(out.scale() > 0.0f) || os.c != is.c || os.h != windowOutputSize(is.h, p.kernel, p.stride, p.pad) ||
      os.w != windowOutputSize(is.w, p.kernel, p.stride, p.pad)) {
    TINYYOLO_LOGE("Reference", "depthwise conv2dQ: output shape or scale does not match");
    return Status::ShapeMismatch;
  }
  const int K = p.kernel, S = p.stride, P = p.pad, C = is.c;
  const float inv_out = 1.0f / out.scale();
  for (int c = 0; c < C; ++c) {
    const int8_t* w_c = weights + (size_t)c * K * K;
    const float dequant = in.scale() * w_scales[c];
    for (int oy = 0; oy < os.h; ++oy) {
      for (int ox = 0; ox < os.w; ++ox) {
        int32_t acc = 0;
        for (int ky = 0; ky < K; ++ky) {
          const int iy = oy * S - P + ky;
          if (iy < 0 || iy >= is.h) continue;
          for (int kx = 0; kx < K; ++kx) {
            const int ix = ox * S - P + kx;
            if (ix < 0 || ix >= is.w) continue;
            acc += (int32_t)in.at(iy, ix, c) * (int32_t)w_c[ky * K + kx];
          }
        }
        const float real = activate(p.activation, bias[c] + (float)acc * dequant);
        out.at(oy, ox, c) = quantizeValue(real, inv_out);
      }
    }
  }
  return Status::Ok;
}

/// Portable int8 max pooling. The scale is unchanged, so the result equals the quantized
/// float max pooling exactly (rounding is monotonic).
inline Status referenceMaxPoolQ(const QTensor& in, int kernel, int stride, int pad_total,
                                QTensor& out) {
  const Shape is = in.shape();
  const Shape os = out.shape();
  if (kernel <= 0 || stride <= 0 || pad_total < 0) {
    TINYYOLO_LOGE("Reference", "maxPoolQ: invalid kernel=%d stride=%d pad=%d", kernel, stride,
                  pad_total);
    return Status::InvalidArgument;
  }
  if (os.c != is.c || os.h != poolOutputSize(is.h, kernel, stride, pad_total) ||
      os.w != poolOutputSize(is.w, kernel, stride, pad_total)) {
    TINYYOLO_LOGE("Reference", "maxPoolQ: output %dx%dx%d does not match input %dx%dx%d", os.h,
                  os.w, os.c, is.h, is.w, is.c);
    return Status::ShapeMismatch;
  }
  out.setScale(in.scale());
  const int begin = pad_total / 2;
  for (int oy = 0; oy < os.h; ++oy) {
    for (int ox = 0; ox < os.w; ++ox) {
      for (int c = 0; c < os.c; ++c) {
        bool found = false;
        int8_t best = 0;
        for (int ky = 0; ky < kernel; ++ky) {
          const int iy = oy * stride - begin + ky;
          if (iy < 0 || iy >= is.h) continue;
          for (int kx = 0; kx < kernel; ++kx) {
            const int ix = ox * stride - begin + kx;
            if (ix < 0 || ix >= is.w) continue;
            const int8_t v = in.at(iy, ix, c);
            if (!found || v > best) {
              best = v;
              found = true;
            }
          }
        }
        out.at(oy, ox, c) = best;
      }
    }
  }
  return Status::Ok;
}

/// Compute kernels. Layers describe *what* to compute, a Backend decides *how*.
/// ReferenceBackend is portable C++. Optimised platform backends (step 2) derive from
/// this class and override only the operations they accelerate.
///
/// Conventions: activations are HWC float tensors (or int8 QTensors with a scale); float conv
/// weights are OIHW ([out][in][kh][kw]); int8 conv weights are OHWI ([out][kh][kw][in]), the
/// layout ESP-NN uses; conv bias is [out]. `out` must already have the right shape.
class Backend {
 public:
  virtual ~Backend() {}
  virtual const char* name() const = 0;
  /// Threads the kernels may use (1 = the calling thread only).
  virtual int workers() const { return 1; }

  /// Input channels of a standard convolution as the backend wants them stored (the graph pads
  /// with zeros). Default: a multiple of 4. The P4 assembly kernels read 16 channels at a time.
  virtual int inputChannelPadding(int channels) const { return (channels + 3) / 4 * 4; }

  virtual Status conv2d(const Tensor& in, const float* weights, const float* bias,
                        const ConvParams& params, Tensor& out) = 0;
  /// Convolution with int8 weights: weight (o, i) = weights[o][i] * scales[o]. Otherwise
  /// identical to conv2d().
  virtual Status conv2dInt8(const Tensor& in, const int8_t* weights, const float* scales,
                            const float* bias, const ConvParams& params, Tensor& out) = 0;
  /// Int8 activations and int8 weights. `out` carries the output scale (set when allocated).
  /// Default: portable reference kernel.
  virtual Status conv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales,
                         const float* bias, const ConvParams& params, QTensor& out) {
    return referenceConv2dQ(in, weights, w_scales, bias, params, out);
  }
  /// Depthwise int8 convolution (groups == channels). Default: portable reference kernel.
  virtual Status depthwiseConv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales,
                                  const float* bias, const ConvParams& params, QTensor& out) {
    return referenceDepthwiseConv2dQ(in, weights, w_scales, bias, params, out);
  }
  /// Max pooling on int8 activations. Default: portable reference kernel.
  virtual Status maxPoolQ(const QTensor& in, int kernel, int stride, int pad, QTensor& out) {
    return referenceMaxPoolQ(in, kernel, stride, pad, out);
  }
  virtual Status maxPool(const Tensor& in, int kernel, int stride, int pad, Tensor& out) = 0;
  /// Nearest-neighbour upsampling by an integer factor.
  virtual Status upsample(const Tensor& in, int factor, Tensor& out) = 0;
  /// Channel concatenation: out = [a, b].
  virtual Status concat(const Tensor& a, const Tensor& b, Tensor& out) = 0;
  virtual Status copy(const Tensor& in, Tensor& out) = 0;
  /// Copies channels [first, first + count) of `in` to `out` (same height and width).
  virtual Status slice(const Tensor& in, int first, int count, Tensor& out) = 0;
  /// Element-wise out = act(a + b). All three tensors have the same shape.
  virtual Status add(const Tensor& a, const Tensor& b, Activation act, Tensor& out) = 0;
};

}  // namespace tinyyolo
