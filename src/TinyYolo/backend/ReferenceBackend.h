#pragma once
/// @file ReferenceBackend.h  Portable, unoptimised kernels that run on every platform.

#include "TinyYolo/platform/KernelOptimize.h"
#include <string.h>

#include "TinyYolo/backend/Backend.h"
#include "TinyYolo/core/Logger.h"

namespace tinyyolo {

class ReferenceBackend : public Backend {
 public:
  const char* name() const override { return "reference"; }

  Status conv2d(const Tensor& in, const float* weights, const float* bias, const ConvParams& p,
                Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (weights == nullptr || bias == nullptr || p.out_channels <= 0) {
      TINYYOLO_LOGE("Reference", "conv2d: missing weights, bias or channel count");
      return Status::InvalidArgument;
    }
    if (os.c != p.out_channels || os.h != windowOutputSize(is.h, p.kernel, p.stride, p.pad) ||
        os.w != windowOutputSize(is.w, p.kernel, p.stride, p.pad)) {
      TINYYOLO_LOGE("Reference", "conv2d: output %dx%dx%d does not match input %dx%dx%d", os.h,
                    os.w, os.c, is.h, is.w, is.c);
      return Status::ShapeMismatch;
    }

    const int K = p.kernel;
    const int S = p.stride;
    const int P = p.pad;
    const int IC = is.c;
    for (int oc = 0; oc < os.c; ++oc) {
      const float* w_oc = weights + (size_t)oc * IC * K * K;
      for (int oy = 0; oy < os.h; ++oy) {
        for (int ox = 0; ox < os.w; ++ox) {
          float acc = bias[oc];
          const int y0 = oy * S - P;
          const int x0 = ox * S - P;
          for (int ic = 0; ic < IC; ++ic) {
            const float* w_ic = w_oc + (size_t)ic * K * K;
            for (int ky = 0; ky < K; ++ky) {
              const int iy = y0 + ky;
              if (iy < 0 || iy >= is.h) continue;
              for (int kx = 0; kx < K; ++kx) {
                const int ix = x0 + kx;
                if (ix < 0 || ix >= is.w) continue;
                acc += in.at(iy, ix, ic) * w_ic[ky * K + kx];
              }
            }
          }
          out.at(oy, ox, oc) = activate(p.activation, acc);
        }
      }
    }
    return Status::Ok;
  }

  /// `pad_total` is split as pad_total/2 before the window and the rest after (Darknet).
  Status conv2dInt8(const Tensor& in, const int8_t* weights, const float* scales,
                    const float* bias, const ConvParams& p, Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (weights == nullptr || scales == nullptr || bias == nullptr || p.out_channels <= 0) {
      TINYYOLO_LOGE("Reference", "conv2dInt8: missing weights, scales, bias or channel count");
      return Status::InvalidArgument;
    }
    if (os.c != p.out_channels || os.h != windowOutputSize(is.h, p.kernel, p.stride, p.pad) ||
        os.w != windowOutputSize(is.w, p.kernel, p.stride, p.pad)) {
      TINYYOLO_LOGE("Reference", "conv2dInt8: output %dx%dx%d does not match input %dx%dx%d", os.h,
                    os.w, os.c, is.h, is.w, is.c);
      return Status::ShapeMismatch;
    }
    const int K = p.kernel;
    const int S = p.stride;
    const int P = p.pad;
    const int IC = is.c;
    for (int oc = 0; oc < os.c; ++oc) {
      const int8_t* w_oc = weights + (size_t)oc * K * K * IC;  // OHWI: [oc][ky][kx][ic]
      for (int oy = 0; oy < os.h; ++oy) {
        for (int ox = 0; ox < os.w; ++ox) {
          float acc = 0.0f;
          const int y0 = oy * S - P;
          const int x0 = ox * S - P;
          for (int ic = 0; ic < IC; ++ic) {
            for (int ky = 0; ky < K; ++ky) {
              const int iy = y0 + ky;
              if (iy < 0 || iy >= is.h) continue;
              for (int kx = 0; kx < K; ++kx) {
                const int ix = x0 + kx;
                if (ix < 0 || ix >= is.w) continue;
                acc += in.at(iy, ix, ic) * (float)w_oc[(ky * K + kx) * IC + ic];
              }
            }
          }
          out.at(oy, ox, oc) = activate(p.activation, bias[oc] + acc * scales[oc]);
        }
      }
    }
    return Status::Ok;
  }

  Status maxPool(const Tensor& in, int kernel, int stride, int pad_total, Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (kernel <= 0 || stride <= 0 || pad_total < 0) {
      TINYYOLO_LOGE("Reference", "maxPool: invalid kernel=%d stride=%d pad=%d", kernel, stride,
                    pad_total);
      return Status::InvalidArgument;
    }
    if (os.c != is.c || os.h != poolOutputSize(is.h, kernel, stride, pad_total) ||
        os.w != poolOutputSize(is.w, kernel, stride, pad_total)) {
      TINYYOLO_LOGE("Reference", "maxPool: output %dx%dx%d does not match input %dx%dx%d", os.h,
                    os.w, os.c, is.h, is.w, is.c);
      return Status::ShapeMismatch;
    }
    const int begin = pad_total / 2;
    for (int oy = 0; oy < os.h; ++oy) {
      for (int ox = 0; ox < os.w; ++ox) {
        for (int c = 0; c < os.c; ++c) {
          bool found = false;
          float best = 0.0f;
          for (int ky = 0; ky < kernel; ++ky) {
            const int iy = oy * stride - begin + ky;
            if (iy < 0 || iy >= is.h) continue;
            for (int kx = 0; kx < kernel; ++kx) {
              const int ix = ox * stride - begin + kx;
              if (ix < 0 || ix >= is.w) continue;
              const float v = in.at(iy, ix, c);
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

  Status upsample(const Tensor& in, int factor, Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (factor <= 0 || os.h != is.h * factor || os.w != is.w * factor || os.c != is.c) {
      TINYYOLO_LOGE("Reference", "upsample: factor %d maps %dx%dx%d to %dx%dx%d", factor, is.h,
                    is.w, is.c, os.h, os.w, os.c);
      return Status::ShapeMismatch;
    }
    for (int oy = 0; oy < os.h; ++oy) {
      for (int ox = 0; ox < os.w; ++ox) {
        for (int c = 0; c < os.c; ++c) {
          out.at(oy, ox, c) = in.at(oy / factor, ox / factor, c);
        }
      }
    }
    return Status::Ok;
  }

  Status concat(const Tensor& a, const Tensor& b, Tensor& out) override {
    const Shape as = a.shape();
    const Shape bs = b.shape();
    const Shape os = out.shape();
    if (as.h != bs.h || as.w != bs.w || os.h != as.h || os.w != as.w || os.c != as.c + bs.c) {
      TINYYOLO_LOGE("Reference", "concat: %dx%dx%d + %dx%dx%d into %dx%dx%d", as.h, as.w, as.c,
                    bs.h, bs.w, bs.c, os.h, os.w, os.c);
      return Status::ShapeMismatch;
    }
    for (int y = 0; y < os.h; ++y) {
      for (int x = 0; x < os.w; ++x) {
        for (int c = 0; c < as.c; ++c) out.at(y, x, c) = a.at(y, x, c);
        for (int c = 0; c < bs.c; ++c) out.at(y, x, as.c + c) = b.at(y, x, c);
      }
    }
    return Status::Ok;
  }

  Status slice(const Tensor& in, int first, int count, Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (first < 0 || count <= 0 || first + count > is.c || os.h != is.h || os.w != is.w || os.c != count) {
      TINYYOLO_LOGE("Reference", "slice: channels %d..%d of %d into %dx%dx%d", first, first + count,
                    is.c, os.h, os.w, os.c);
      return Status::ShapeMismatch;
    }
    for (int y = 0; y < os.h; ++y) {
      for (int x = 0; x < os.w; ++x) {
        for (int c = 0; c < count; ++c) out.at(y, x, c) = in.at(y, x, first + c);
      }
    }
    return Status::Ok;
  }

  Status add(const Tensor& a, const Tensor& b, Activation act, Tensor& out) override {
    if (a.shape() != b.shape() || out.shape() != a.shape()) {
      TINYYOLO_LOGE("Reference", "add: shapes %dx%dx%d, %dx%dx%d and %dx%dx%d differ", a.shape().h,
                    a.shape().w, a.shape().c, b.shape().h, b.shape().w, b.shape().c,
                    out.shape().h, out.shape().w, out.shape().c);
      return Status::ShapeMismatch;
    }
    for (size_t i = 0; i < out.size(); ++i) out.data()[i] = activate(act, a.data()[i] + b.data()[i]);
    return Status::Ok;
  }

  Status copy(const Tensor& in, Tensor& out) override {
    if (in.shape() != out.shape()) {
      TINYYOLO_LOGE("Reference", "copy: shape %dx%dx%d differs from %dx%dx%d", in.shape().h,
                    in.shape().w, in.shape().c, out.shape().h, out.shape().w, out.shape().c);
      return Status::ShapeMismatch;
    }
    memcpy(out.data(), in.data(), in.size() * sizeof(float));
    return Status::Ok;
  }
};

}  // namespace tinyyolo
