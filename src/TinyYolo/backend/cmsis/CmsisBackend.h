#pragma once
/// @file CmsisBackend.h  int8 convolutions with CMSIS-NN (Arm DSP kernels) on the RP2040/RP2350.
///
/// The standard convolutions run in CMSIS-NN. Requantization uses one rounding: the per-channel
/// multiplier is ratio * 2^31 with shift 0, as in Esp32NnBackend. The activation (SiLU, ...) is a
/// lookup table on the int8 output, so activated layers round twice (about 2 to 4 % of the values
/// are one step off). Depthwise convolutions and everything else use the portable ReferenceBackend.
#include <math.h>
#include <string.h>
#include "TinyYolo/backend/ReferenceBackend.h"
#include "TinyYolo/backend/cmsis/arm_cmsis_nn_config.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/QTensor.h"
#include "TinyYolo/platform/Memory.h"

#if defined(ARDUINO_ARCH_RP2040) && defined(CMSIS_NN)
#define TINYYOLO_HAS_CMSIS 1
extern "C" {
#include "TinyYolo/backend/cmsis/cmsis_nn_all.h"
}
#endif

namespace tinyyolo {

#if defined(TINYYOLO_HAS_CMSIS)

class CmsisBackend : public ReferenceBackend {
 public:
  const char* name() const override { return "cmsis-nn"; }

  Status conv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                 const ConvParams& p, QTensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const bool valid = weights != nullptr && w_scales != nullptr && bias != nullptr && out.scale() > 0.0f &&
                       in.scale() > 0.0f && p.out_channels == os.c && is.c > 0 &&
                       os.h == windowOutputSize(is.h, p.kernel, p.stride, p.pad) &&
                       os.w == windowOutputSize(is.w, p.kernel, p.stride, p.pad);
    if (!valid) return ReferenceBackend::conv2dQ(in, weights, w_scales, bias, p, out);
    TINYYOLO_CHECK(prepareRequant(in, w_scales, bias, out.scale(), os.c));

    cmsis_nn_conv_params cp;
    cp.input_offset = 0;
    cp.output_offset = 0;
    cp.stride.w = p.stride;
    cp.stride.h = p.stride;
    cp.padding.w = p.pad;
    cp.padding.h = p.pad;
    cp.dilation.w = 1;
    cp.dilation.h = 1;
    cp.activation.min = -127;  // the lookup table below applies the activation; the kernel clamps as before
    cp.activation.max = 127;
    const cmsis_nn_dims id = {1, is.h, is.w, is.c};
    const cmsis_nn_dims fd = {os.c, p.kernel, p.kernel, is.c};  // OHWI, as the CMSIS-NN filter layout
    const cmsis_nn_dims bd = {1, 1, 1, os.c};
    const cmsis_nn_dims od = {1, os.h, os.w, os.c};
    cmsis_nn_per_channel_quant_params q;
    q.multiplier = mult_.data();
    q.shift = shift_.data();

    const int32_t need = arm_convolve_wrapper_s8_get_buffer_size(&cp, &id, &fd, &od);
    cmsis_nn_context ctx;
    ctx.buf = nullptr;
    ctx.size = 0;
    if (need > 0) {
      if (need > (int32_t)scratch_size_) {
        releaseMemory(scratch_, scratch_in_psram_);
        scratch_ = (int8_t*)allocateMemory((size_t)need, false, scratch_in_psram_);
        if (scratch_ == nullptr) return Status::OutOfMemory;
        scratch_size_ = (size_t)need;
      }
      ctx.buf = scratch_;
      ctx.size = need;
    }
    const arm_cmsis_nn_status st = arm_convolve_wrapper_s8(&ctx, &cp, &q, &id, in.data(), &fd, weights, &bd,
                                                           bias32_.data(), &od, out.data());
    if (st != ARM_CMSIS_NN_SUCCESS) return ReferenceBackend::conv2dQ(in, weights, w_scales, bias, p, out);
    applyActivation(out, p.activation);
    return Status::Ok;
  }

  ~CmsisBackend() override { releaseMemory(scratch_, scratch_in_psram_); }

 private:
  /// One rounding: multiplier = ratio * 2^31 with shift 0 (ratio < 1). Other ratios use the
  /// normalized form, which rounds twice.
  Status prepareRequant(const QTensor& in, const float* w_scales, const float* bias, float out_scale, int oc) {
    TINYYOLO_CHECK(mult_.allocate((size_t)oc));
    TINYYOLO_CHECK(shift_.allocate((size_t)oc));
    TINYYOLO_CHECK(bias32_.allocate((size_t)oc));
    for (int o = 0; o < oc; ++o) {
      const double acc_scale = (double)in.scale() * (double)w_scales[o];
      const double ratio = acc_scale / (double)out_scale;
      const long long m = llround(ratio * 2147483648.0);
      if (ratio > 0.0 && m > 0 && m < 2147483648LL) {
        mult_.data()[o] = (int32_t)m;
        shift_.data()[o] = 0;
      } else {
        int exponent = 0;
        const double fraction = frexp(ratio, &exponent);
        long long mm = llround(fraction * 2147483648.0);
        if (mm == 2147483648LL) {
          mm /= 2;
          ++exponent;
        }
        mult_.data()[o] = (int32_t)mm;
        shift_.data()[o] = exponent;
      }
      const double b = llround((double)bias[o] / acc_scale);
      bias32_.data()[o] = b > 2147483647.0 ? 2147483647 : (b < -2147483648.0 ? (int32_t)-2147483647 - 1 : (int32_t)b);
    }
    return Status::Ok;
  }

  /// The activation as a lookup table over the int8 values (the same as Esp32NnBackend).
  static void applyActivation(QTensor& out, Activation activation) {
    if (activation == Activation::Linear) return;
    const float s = out.scale();
    const float inv = 1.0f / s;
    int8_t lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = quantizeValue(activate(activation, (float)(i - 128) * s), inv);
    int8_t* d = out.data();
    for (size_t i = 0; i < out.size(); ++i) d[i] = lut[(uint8_t)d[i] ^ 0x80];
  }

  Buffer<int32_t> mult_;
  Buffer<int32_t> shift_;
  Buffer<int32_t> bias32_;
  int8_t* scratch_ = nullptr;
  bool scratch_in_psram_ = false;
  size_t scratch_size_ = 0;
};

#else

/// Without CMSIS-NN (other cores, or GCC older than 12) the backend is the portable one.
using CmsisBackend = ReferenceBackend;

#endif

}  // namespace tinyyolo
