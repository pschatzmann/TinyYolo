#pragma once
/// @file Esp32NnBackend.h  Int8 convolutions with the ESP-NN kernels on the ESP32-S3.
///
/// conv2dQ() uses ESP-NN when the build has it (TINYYOLO_HAS_ESP_NN):
///  - 3x3 kernels (padding 1): the S3-specific or the generic ESP-NN convolution, in row bands; on the
///    S3, layers the S3-specific kernel does not handle reliably are padded with zero channels;
///  - depthwise 3x3 kernels: esp_nn_depthwise_conv_s8 (generic on the S3);
///  - 1x1 kernels, stride 1, no padding: esp_nn_fully_connected_per_ch_s8, one call per pixel.
/// Everything else, including every host build, uses the portable int8 kernel. Float
/// convolutions and all other operations are inherited from Esp32Backend.
///
/// ESP-NN can only clamp its output to the int8 range, so every activation other than Linear
/// runs as a float pass over the int8 output. Values beyond the output scale's range are
/// already clipped at that point, so the output scale must cover the data (calibration does).
///
/// Requantization: the real multiplier of output channel o is in_scale * w_scale[o] /
/// out_scale. It is stored as mult * 2^(shift - 31) with mult in [2^30, 2^31), the same
/// convention as TFLite, which ESP-NN follows. The bias is converted to the int32
/// accumulator scale.

#include "TinyYolo/platform/KernelOptimize.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "TinyYolo/backend/Esp32Backend.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/platform/Features.h"

#if defined(ESP32) && __has_include("sdkconfig.h")
#include "sdkconfig.h"
#endif
#if defined(CONFIG_IDF_TARGET_ESP32P4)
#include "TinyYolo/backend/esp32p4/Esp32P4Conv.h"  // the P4 assembly kernels (vendored, MIT)
#endif

#if defined(TINYYOLO_HAS_ESP_NN)
#include "esp_heap_caps.h"
extern "C" {
#if defined(TINYYOLO_ESP_NN_P4)
#include "TinyYolo/backend/esp32p4/esp-nn/esp_nn_riscv_pie.h"  // ESP32-P4 (vendored declarations): the same API, RISC-V PIE kernels
#else
#include "TinyYolo/backend/esp32s3/esp-nn/include/esp_nn_esp32s3.h"  // vendored ESP-NN (S3)
// The S3 build uses the S3-specific ESP-NN 3x3 convolution (padded where it is not reliable), the generic
// optimised kernels for the depthwise convolutions (the S3-specific depthwise kernel is wrong), and our
// exact kernel for the linear 1x1 convolutions (see exactConv2dQ).
#include "TinyYolo/backend/esp32s3/esp-nn/include/esp_nn_ansi_headers.h"
// The vendored ESP-NN sources, as headers (see esp_nn_s3_sources.h): the S3 kernels are compiled with the sketch.
#include "TinyYolo/backend/esp32s3/esp-nn/esp_nn_s3_sources.h"
#endif
}
#endif

#if defined(TINYYOLO_HAS_ESP_NN) && defined(TINYYOLO_ESP_NN_S3)
// The S3 has two kernel sets. 3x3 convolutions use the S3-specific kernel, on layers with 4 or a multiple
// of 16 input and output channels; other 3x3 layers are padded with zero channels to multiples of 16
// (see espNnConvPadded). Depthwise convolutions always use the generic kernel.
#define NN_CONV_SCRATCH(s3, ...) ((s3) ? esp_nn_get_conv_scratch_size_esp32s3(__VA_ARGS__) : esp_nn_get_conv_scratch_size_opt(__VA_ARGS__))
#define NN_CONV_SET_SCRATCH(s3, buf) ((s3) ? esp_nn_set_conv_scratch_buf_esp32s3(buf) : esp_nn_set_conv_scratch_buf_opt(buf))
#define NN_CONV(s3, ...) ((s3) ? esp_nn_conv_s8_esp32s3(__VA_ARGS__) : esp_nn_conv_s8_opt(__VA_ARGS__))
#define NN_DW_SCRATCH(...) esp_nn_get_depthwise_conv_scratch_size_opt(__VA_ARGS__)
#define NN_DW_SET_SCRATCH(buf) esp_nn_set_depthwise_conv_scratch_buf_opt(buf)
#define NN_DW(...) esp_nn_depthwise_conv_s8_opt(__VA_ARGS__)
#elif defined(TINYYOLO_HAS_ESP_NN)
#define NN_CONV_SCRATCH(s3, ...) esp_nn_get_conv_scratch_size(__VA_ARGS__)
#define NN_CONV_SET_SCRATCH(s3, buf) esp_nn_set_conv_scratch_buf(buf)
#define NN_CONV(s3, ...) esp_nn_conv_s8(__VA_ARGS__)
#define NN_DW_SCRATCH(...) esp_nn_get_depthwise_conv_scratch_size(__VA_ARGS__)
#define NN_DW_SET_SCRATCH(buf) esp_nn_set_depthwise_conv_scratch_buf(buf)
#define NN_DW(...) esp_nn_depthwise_conv_s8(__VA_ARGS__)
#endif

namespace tinyyolo {

class Esp32NnBackend : public Esp32Backend {
 public:
  /// workers = 0 uses both cores (2), as for Esp32Backend.
  explicit Esp32NnBackend(int workers = 0) : Esp32Backend(workers) {}
  Esp32NnBackend(const Esp32NnBackend&) = delete;
  Esp32NnBackend& operator=(const Esp32NnBackend&) = delete;

  /// On the P4: true (the default) runs the int8 convolutions and depthwise convolutions on the
  /// vendored P4 assembly kernels (from esp-dl) where they fit; false runs ESP-NN for the convolutions instead (the
  /// depthwise convolutions then take the portable kernel). Use false to compare the two paths.
  void setP4Kernels(bool enable) { p4_kernels_ = enable; }
  bool p4KernelsEnabled() const { return p4_kernels_; }

  /// The first convolution (3 input channels) is zero-padded to 16 for the P4 assembly kernels, and to
  /// a multiple of 4 otherwise (16 channels would need 1.6 MB at 320x320 on the S3).
  int inputChannelPadding(int channels) const override {
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    if (p4_kernels_ && channels == 3) return 16;
#endif
    return (channels + 3) / 4 * 4;
  }

  const char* name() const override {
#if defined(TINYYOLO_HAS_ESP_NN)
    return "esp32-nn";
#else
    return Esp32Backend::name();
#endif
  }

  /// True when this build calls the ESP-NN kernels (otherwise conv2dQ is the portable kernel).
  static bool hasEspNn() {
#if defined(TINYYOLO_HAS_ESP_NN)
    return true;
#else
    return false;
#endif
  }

  /// Depthwise int8 convolution. On the P4, 3x3 with stride 1 and padding 1 and one shift for all
  /// channels (the weights share one power-of-two scale) runs the P4 assembly kernel; anything
  /// else takes the portable reference.
  Status depthwiseConv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                          const ConvParams& p, QTensor& out) override {
#if defined(TINYYOLO_HAS_ESP_NN) && defined(CONFIG_IDF_TARGET_ESP32P4)
    const Shape is = in.shape();
    const bool shape_ok = p4_kernels_ && p.kernel == 3 && p.stride == 1 && p.pad == 1 && is.c % 16 == 0 && p.out_channels == is.c &&
                          weights != nullptr && w_scales != nullptr && bias != nullptr && in.scale() > 0.0f &&
                          out.scale() > 0.0f && ((uintptr_t)out.data() & 15) == 0;
    if (shape_ok) {
      const Status s = p4Depthwise(in, weights, w_scales, bias, p, out);
      if (s != Status::Unsupported) return s;
    }
#endif
#if defined(TINYYOLO_HAS_ESP_NN)
    {
      // ESP-NN's depthwise kernel for 3x3 with padding 1 and stride 1 or 2 (any channel count).
      const Shape is = in.shape();
      const Shape os = out.shape();
      const bool valid = p.kernel == 3 && p.pad == 1 && (p.stride == 1 || p.stride == 2) &&
                         p.out_channels == is.c && os.c == is.c && weights != nullptr && w_scales != nullptr &&
                         bias != nullptr && in.scale() > 0.0f && out.scale() > 0.0f &&
                         os.h == windowOutputSize(is.h, 3, p.stride, 1) && os.w == windowOutputSize(is.w, 3, p.stride, 1);
      if (valid) {
        const Status s = espNnDepthwise(in, weights, w_scales, bias, p, out);
        if (s != Status::Unsupported) return s;
      }
    }
#endif
    return referenceDepthwiseConv2dQ(in, weights, w_scales, bias, p, out);
  }

#if defined(TINYYOLO_ESP_NN_S3)
  /// The S3 convolution for layers with an activation: our own int8 kernel, not ESP-NN. The sum is
  /// exact (int32), the activation is applied in float to that sum, and the result is rounded once,
  /// as the reference does. (ESP-NN rounds before the activation, which changes about 2% of the
  /// values by one step, and the graph amplifies that.) Weights are OHWI, as in the other kernels.
  static Status exactConv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales,
                             const float* bias, const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int K = p.kernel, S = p.stride, P = p.pad, IC = is.c, OC = os.c;
    const int IH = is.h, IW = is.w;
    const float inv_out = 1.0f / out.scale();
    Buffer<float> deq;
    TINYYOLO_CHECK(deq.allocate((size_t)OC));
    for (int oc = 0; oc < OC; ++oc) deq.data()[oc] = in.scale() * w_scales[oc];
    const int8_t* x = in.data();
    const size_t row = (size_t)IW * IC;
    const size_t wstride = (size_t)K * K * IC;
    for (int oy = 0; oy < os.h; ++oy) {
      const int y0 = oy * S - P;
      const int ky0 = y0 < 0 ? -y0 : 0;
      const int ky1 = y0 + K > IH ? IH - y0 : K;
      for (int ox = 0; ox < os.w; ++ox) {
        const int x0 = ox * S - P;
        const int kx0 = x0 < 0 ? -x0 : 0;
        const int kx1 = x0 + K > IW ? IW - x0 : K;
        int8_t* o = out.data() + ((size_t)oy * os.w + ox) * OC;
        // Interior pixel: each kernel row is one contiguous run of K*IC values in the input and in
        // the OHWI filter, so the S3 vector dot product (exact int32) does the row.
        const bool fast = ky0 == 0 && ky1 == K && kx0 == 0 && kx1 == K && ((K * IC) % 16) == 0;
        if (fast) {
          const int8_t* xw = x + ((size_t)y0 * IW + x0) * IC;
          const int len16 = K * IC / 16;
          for (int oc = 0; oc < OC; ++oc) {
            const int8_t* wo = weights + (size_t)oc * wstride;
            int32_t acc = 0;
            for (int ky = 0; ky < K; ++ky) {
              acc += esp_nn_dot_s8_unaligned_esp32s3(xw + (size_t)ky * row, wo + (size_t)ky * K * IC, len16);
            }
            const float real = activate(p.activation, bias[oc] + (float)acc * deq.data()[oc]);
            o[oc] = quantizeValue(real, inv_out);
          }
          continue;
        }
        for (int oc = 0; oc < OC; ++oc) {
          const int8_t* wo = weights + (size_t)oc * wstride;
          int32_t acc = 0;
          for (int ky = ky0; ky < ky1; ++ky) {
            const int8_t* xr = x + (size_t)(y0 + ky) * row;
            const int8_t* wr = wo + (size_t)ky * K * IC;
            for (int kx = kx0; kx < kx1; ++kx) {
              const int8_t* xp = xr + (size_t)(x0 + kx) * IC;
              const int8_t* wp = wr + (size_t)kx * IC;
              for (int ic = 0; ic < IC; ++ic) acc += (int32_t)xp[ic] * (int32_t)wp[ic];
            }
          }
          const float real = activate(p.activation, bias[oc] + (float)acc * deq.data()[oc]);
          o[oc] = quantizeValue(real, inv_out);
        }
      }
    }
    return Status::Ok;
  }
#endif  // TINYYOLO_ESP_NN_S3

  Status conv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales,
                 const float* bias, const ConvParams& p, QTensor& out) override {
#if defined(TINYYOLO_HAS_ESP_NN)
    const Shape is = in.shape();
    const Shape os = out.shape();
    const bool valid = weights != nullptr && w_scales != nullptr && bias != nullptr &&
                       out.scale() > 0.0f && in.scale() > 0.0f && p.out_channels == os.c &&
                       os.h == windowOutputSize(is.h, p.kernel, p.stride, p.pad) &&
                       os.w == windowOutputSize(is.w, p.kernel, p.stride, p.pad);
#if defined(CONFIG_IDF_TARGET_ESP32P4)
    if (p4_kernels_ && valid && ((p.kernel == 3 && p.pad <= 1) || (p.kernel == 1 && p.pad == 0)) &&
        is.c % 16 == 0 &&
        os.c % 16 == 0) {
      const Status s = p4Conv(in, weights, w_scales, bias, p, out);
      if (s != Status::Unsupported) return s;  // otherwise ESP-NN below
    }
#endif
#if defined(TINYYOLO_ESP_NN_S3)
    // The linear 1x1 layers (the detection heads and the neck projections) use our own exact kernel,
    // see exactConv2dQ. The ESP-NN pointwise path differs from it in the graph, and those layers decide the result.
    if (valid && p.kernel == 1 && p.activation == Activation::Linear) {
      return exactConv2dQ(in, weights, w_scales, bias, p, out);
    }
#endif
    if (valid && p.kernel == 3 && is.c % 4 == 0 && os.c % 4 == 0) {
      return espNnConv(in, weights, w_scales, bias, p, out);
    }
    if (valid && p.kernel == 1 && p.stride == 1 && p.pad == 0) {
      return espNnPointwise(in, weights, w_scales, bias, p, out);
    }
#endif
    return Esp32Backend::conv2dQ(in, weights, w_scales, bias, p, out);
  }

#if defined(TINYYOLO_HAS_ESP_NN)
  ~Esp32NnBackend() override { releaseScratch(); }
#endif

 private:
  bool p4_kernels_ = true;

#if defined(TINYYOLO_HAS_ESP_NN)
  /// Per-channel requantization (mult_, shift_) and bias (bias32_) for `OC` channels.
  Status prepareRequant(const QTensor& in, const float* w_scales, const float* bias, float out_scale,
                        int OC) {
    TINYYOLO_CHECK(reserveChannels(OC));
    for (int o = 0; o < OC; ++o) {
      const double acc_scale = (double)in.scale() * (double)w_scales[o];
      int exponent = 0;
      const double fraction = frexp(acc_scale / (double)out_scale, &exponent);  // in [0.5, 1)
      long long mult = llround(fraction * 2147483648.0);
      if (mult == 2147483648LL) {
        mult /= 2;
        ++exponent;
      }
      mult_[o] = (int32_t)mult;
      shift_[o] = exponent;
      const double b = llround((double)bias[o] / acc_scale);
      bias32_[o] = b > 2147483647.0 ? 2147483647
                                    : (b < -2147483648.0 ? (int32_t)-2147483647 - 1 : (int32_t)b);
    }
    return Status::Ok;
  }

  /// Requantization for the ESP-NN kernels, with one rounding. prepareRequant() normalizes the
  /// multiplier to [0.5, 1) and so needs a right shift, and the kernel rounds twice (once in the
  /// multiply, once in the shift). Here the multiplier is ratio * 2^31 with shift 0, so the kernel
  /// rounds once, as the reference does. Ratios of 1 or more keep the normalized form.
  Status prepareRequantSingle(const QTensor& in, const float* w_scales, const float* bias, float out_scale,
                              int OC) {
    TINYYOLO_CHECK(prepareRequant(in, w_scales, bias, out_scale, OC));
    for (int o = 0; o < OC; ++o) {
      const double ratio = ((double)in.scale() * (double)w_scales[o]) / (double)out_scale;
      const long long m = llround(ratio * 2147483648.0);
      if (ratio > 0.0 && m > 0 && m < 2147483648LL) {
        mult_[o] = (int32_t)m;
        shift_[o] = 0;
      }
    }
    return Status::Ok;
  }

#if defined(CONFIG_IDF_TARGET_ESP32P4)
  /// Pointer into `buf` rounded up to a multiple of 16 bytes (the buffer has 16 bytes to spare).
  template <typename T>
  static T* alignTo16(Buffer<int8_t>& buf) {
    uint8_t* p = (uint8_t*)buf.data();
    return (T*)(p + ((16 - ((uintptr_t)p & 15)) & 15));
  }

#if defined(CONFIG_IDF_TARGET_ESP32P4)
  /// Packed weights of one convolution ((N/16)HWC16, see Esp32P4Conv.h), built once per layer and kept:
  /// the weights of a layer do not change between frames.
  struct PackedEntry {
    const int8_t* src = nullptr;
    int OC = 0, IC = 0, K = 0;
    Buffer<int8_t> data;
  };
  static constexpr int kPackedCacheSize = 128;  ///< 84 convolutions + 6 depthwise layers in YOLO11n
  PackedEntry packed_cache_[kPackedCacheSize];
  int packed_cache_count_ = 0;

  Status packedWeights(const int8_t* weights, int OC, int IC, int K, int8_t*& packed) {
    for (int i = 0; i < packed_cache_count_; ++i) {
      PackedEntry& e = packed_cache_[i];
      if (e.src == weights && e.OC == OC && e.IC == IC && e.K == K) {
        packed = alignTo16<int8_t>(e.data);
        return Status::Ok;
      }
    }
    if (packed_cache_count_ >= kPackedCacheSize) return Status::OutOfMemory;
    PackedEntry& e = packed_cache_[packed_cache_count_];
    TINYYOLO_CHECK(e.data.allocate((size_t)OC * K * K * IC + 16, MemoryPlacement::PsramFirst));
    e.src = weights;
    e.OC = OC;
    e.IC = IC;
    e.K = K;
    packed = alignTo16<int8_t>(e.data);
    // Weights: OHWI -> (N/16)HWC16, the packing of the kernels.
    for (int oc = 0; oc < OC; ++oc) {
      for (int tap = 0; tap < K * K; ++tap) {
        for (int ic = 0; ic < IC; ++ic) {
          packed[(((oc / 16) * K * K + tap) * IC + ic) * 16 + oc % 16] = weights[((size_t)oc * K * K + tap) * IC + ic];
        }
      }
    }
    ++packed_cache_count_;
    return Status::Ok;
  }

  /// Packed depthwise weights ([C/16][9][16], see depthwise3x3Pack), built once per layer and kept.
  /// Entries of the depthwise layers have IC = 0 and K = 3 in the cache.
  Status packedDepthwise(const int8_t* weights, int C, int8_t*& packed) {
    for (int i = 0; i < packed_cache_count_; ++i) {
      PackedEntry& e = packed_cache_[i];
      if (e.src == weights && e.OC == C && e.IC == 0 && e.K == 3) {
        packed = alignTo16<int8_t>(e.data);
        return Status::Ok;
      }
    }
    if (packed_cache_count_ >= kPackedCacheSize) return Status::OutOfMemory;
    PackedEntry& e = packed_cache_[packed_cache_count_];
    TINYYOLO_CHECK(e.data.allocate((size_t)C * 9 + 16, MemoryPlacement::PsramFirst));
    e.src = weights;
    e.OC = C;
    e.IC = 0;
    e.K = 3;
    packed = alignTo16<int8_t>(e.data);
    tinyyolo::esp32p4::depthwise3x3Pack(weights, C, packed);
    ++packed_cache_count_;
    return Status::Ok;
  }

  /// The P4 depthwise path: needs one power-of-two multiplier for every channel (one shift).
  Status p4Depthwise(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                     const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const int C = is.c;
    TINYYOLO_CHECK(prepareRequant(in, w_scales, bias, out.scale(), C));
    const int shift = 1 - shift_[0];
    for (int o = 0; o < C; ++o) {
      if (mult_[o] != (1 << 30) || 1 - shift_[o] != shift) return Status::Unsupported;
    }
    if (shift < 0 || shift > 62) return Status::Unsupported;
    int8_t* packed = nullptr;
    TINYYOLO_CHECK(packedDepthwise(weights, C, packed));
    Buffer<int8_t> bias_mem;
    TINYYOLO_CHECK(bias_mem.allocate((size_t)C * sizeof(int32_t) + 16, MemoryPlacement::PsramFirst));
    int32_t* bias32 = alignTo16<int32_t>(bias_mem);
    // The kernel truncates: half a divisor added to the bias gives rounding to nearest (as in p4Conv).
    for (int o = 0; o < C; ++o) bias32[o] = bias32_[o] + (shift > 0 ? (int32_t)(1u << (shift - 1)) : 0);
    TINYYOLO_CHECK(tinyyolo::esp32p4::depthwise3x3(in.data(), is.h, is.w, C, packed, shift, bias32, out.data(),
                                                   workers()));
    applyActivation(out, p.activation);
    return Status::Ok;
  }
#endif

  /// The P4 assembly path (see Esp32P4Conv.h): a 3x3 convolution whose requantization is a right
  /// shift per output channel, which holds when every scale is a power of two. Returns
  /// Unsupported when it does not apply, so that the caller takes the ESP-NN path.
  Status p4Conv(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int OC = os.c, IC = is.c, K = p.kernel;
    if (((uintptr_t)out.data() & 15) != 0) return Status::Unsupported;
    // The windows inside the picture are read in place, so the input must be 16-byte aligned for any pad.
    if (((uintptr_t)in.data() & 15) != 0) return Status::Unsupported;
    TINYYOLO_CHECK(prepareRequant(in, w_scales, bias, out.scale(), OC));
    for (int o = 0; o < OC; ++o) {
      // frexp gives mult = 2^30 (a fraction of exactly 0.5) when the multiplier is a power of two:
      // the multiplier is then 2^(shift - 1), i.e. a right shift of 1 - shift.
      if (mult_[o] != (1 << 30)) return Status::Unsupported;
      if (1 - shift_[o] < 0 || 1 - shift_[o] > 62) return Status::Unsupported;
    }
    Buffer<int8_t> shifts_mem, bias_mem;
    TINYYOLO_CHECK(shifts_mem.allocate((size_t)OC + 16, MemoryPlacement::PsramFirst));
    TINYYOLO_CHECK(bias_mem.allocate((size_t)OC * sizeof(int32_t) + 16, MemoryPlacement::PsramFirst));
    int8_t* shifts = alignTo16<int8_t>(shifts_mem);
    int32_t* bias32 = alignTo16<int32_t>(bias_mem);
    int8_t* packed = nullptr;
    TINYYOLO_CHECK(packedWeights(weights, OC, IC, K, packed));
    // The kernels truncate (floor) the shifted accumulator. Adding half of the divisor to the bias
    // turns that into rounding to nearest, as ESP-NN does.
    for (int o = 0; o < OC; ++o) {
      const int shift = 1 - shift_[o];
      shifts[o] = (int8_t)shift;
      bias32[o] = bias32_[o] + (shift > 0 ? (int32_t)(1u << (shift - 1)) : 0);
    }
    if (K == 3) {
      TINYYOLO_CHECK(tinyyolo::esp32p4::conv3x3(in.data(), is.h, is.w, IC, p.stride, p.pad, packed, OC, 0, bias32,
                                                out.data(), shifts, workers()));
    } else {
      TINYYOLO_CHECK(tinyyolo::esp32p4::conv1x1(in.data(), is.h, is.w, IC, p.stride, packed, OC, 0, bias32,
                                                out.data(), shifts, workers()));
    }
    applyActivation(out, p.activation);
    return Status::Ok;
  }
#endif

  /// Applies every activation except Linear in a float pass over the int8 output. (The kernels
  /// clamp to [-127, 127] only; a ReLU clamp at zero inside the kernel gave wrong values on the
  /// S3, so ReLU takes this path too.)
  static void applyActivation(QTensor& out, Activation activation) {
    if (activation == Activation::Linear) return;
    const float s = out.scale();
    const float inv = 1.0f / s;
    // The output takes only 256 values, so the activation is a lookup table: one evaluation per
    // value instead of one per element (SiLU needs an exp per evaluation).
    int8_t lut[256];
    for (int i = 0; i < 256; ++i) {
      lut[i] = quantizeValue(activate(activation, (float)(i - 128) * s), inv);
    }
    int8_t* d = out.data();
    for (size_t i = 0; i < out.size(); ++i) {
      d[i] = lut[(uint8_t)d[i] ^ 0x80];  // d[i] + 128, as an index
    }
  }

  /// The S3-specific 3x3 kernel is reliable for 4 and for multiples of 16 input and output channels.
  /// Other 3x3 layers are padded with zero channels to multiples of 16 (zero weights, unit scales), run
  /// with the S3-specific kernel, and the padding output channels are dropped.
  Status espNnConvPadded(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                         const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int IC = is.c, OC = os.c, K = p.kernel;
    const int ICp = IC == 4 ? 4 : (IC + 15) / 16 * 16;
    const int OCp = (OC + 15) / 16 * 16;
    const size_t pixels = (size_t)is.h * is.w;
    QTensor inp, outp;
    TINYYOLO_CHECK(inp.allocate(Shape(is.h, is.w, ICp), in.scale(), MemoryPlacement::PsramFirst));
    memset(inp.data(), 0, pixels * ICp);
    for (size_t i = 0; i < pixels; ++i) memcpy(inp.data() + i * ICp, in.data() + i * IC, IC);
    Buffer<int8_t> wp;
    TINYYOLO_CHECK(wp.allocate((size_t)OCp * K * K * ICp + 16, MemoryPlacement::PsramFirst));
    memset(wp.data(), 0, (size_t)OCp * K * K * ICp + 16);
    for (int oc = 0; oc < OC; ++oc) {
      for (int t = 0; t < K * K; ++t) {
        for (int ic = 0; ic < IC; ++ic) {
          wp.data()[((size_t)oc * K * K + t) * ICp + ic] = weights[((size_t)oc * K * K + t) * IC + ic];
        }
      }
    }
    Buffer<float> sp, bp;
    TINYYOLO_CHECK(sp.allocate((size_t)OCp));
    TINYYOLO_CHECK(bp.allocate((size_t)OCp));
    for (int oc = 0; oc < OCp; ++oc) {
      sp.data()[oc] = oc < OC ? w_scales[oc] : 1.0f;
      bp.data()[oc] = oc < OC ? bias[oc] : 0.0f;
    }
    TINYYOLO_CHECK(outp.allocate(Shape(os.h, os.w, OCp), out.scale(), MemoryPlacement::PsramFirst));
    const ConvParams pp = {K, p.stride, p.pad, OCp, p.activation};
    TINYYOLO_CHECK(espNnConv(inp, wp.data(), sp.data(), bp.data(), pp, outp));
    for (size_t i = 0; i < (size_t)os.h * os.w; ++i) memcpy(out.data() + i * OC, outp.data() + i * OCp, OC);
    return Status::Ok;
  }

  Status espNnConv(const QTensor& in, const int8_t* weights, const float* w_scales,
                   const float* bias, const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int OC = os.c;
    if (p.kernel == 3 && (((is.c % 16 != 0 && is.c != 4) ) || OC % 16 != 0)) {
#if defined(TINYYOLO_ESP_NN_S3)
      return espNnConvPadded(in, weights, w_scales, bias, p, out);
#endif
    }
    TINYYOLO_CHECK(prepareRequantSingle(in, w_scales, bias, out.scale(), OC));

    const act_params_t act = {-127, 127};
    const int K = p.kernel, S = p.stride, P = p.pad;
    const data_dims_t fd = {K, K, is.c, OC};  // filter: OHWI
    const quant_data_t q = {shift_.data(), mult_.data()};

    // ESP-NN keeps an im2col-style scratch buffer that grows with the rows it convolves, which
    // is too much for the whole picture (2.4 MB for a 96x96 layer). The output is computed in
    // bands of rows: the largest band whose scratch fits kScratchBudget. Each band gets its
    // input rows copied in, with zero rows outside the picture, so the kernel needs no height
    // padding (padding {P, 0}).
    // The S3-specific kernel is wrong for 8 input channels with stride 2 (see the note above).
    // The S3-specific kernel is correct for 4 and for multiples of 16 input channels; it is wrong for 8
    // input channels (stride 1 and 2), so those layers use the generic kernel.
    const bool s3k = is.c == 4 || is.c % 16 == 0;
    int band = os.h;
    int scratch = 0;
    for (;;) {
      const data_dims_t id_b = {is.w, (band - 1) * S + K, is.c, 1};
      const data_dims_t od_b = {os.w, band, OC, 1};
      const conv_params_t cp_b = {0, 0, {S, S}, {P, 0}, {1, 1}, act};
      scratch = NN_CONV_SCRATCH(s3k, &id_b, &fd, &od_b, &cp_b);
      if (scratch <= (int)kScratchBudget || band == 1) break;
      band = (band + 1) / 2;
    }
    TINYYOLO_CHECK(reserveScratch(scratch > 0 ? (size_t)scratch : 0));
    TINYYOLO_CHECK(reserveBand((size_t)((band - 1) * S + K) * is.w * is.c));
    NN_CONV_SET_SCRATCH(s3k, scratch_);

    const size_t row = (size_t)is.w * is.c;  // one input row, HWC
    for (int oy0 = 0; oy0 < os.h; oy0 += band) {
      const int rows = os.h - oy0 < band ? os.h - oy0 : band;
      const int rows_in = (rows - 1) * S + K;
      const int iy0 = oy0 * S - P;
      for (int r = 0; r < rows_in; ++r) {
        const int iy = iy0 + r;
        int8_t* dst = band_.data() + (size_t)r * row;
        if (iy >= 0 && iy < is.h) {
          memcpy(dst, in.data() + (size_t)iy * row, row);
        } else {
          memset(dst, 0, row);
        }
      }
      const data_dims_t id_b = {is.w, rows_in, is.c, 1};
      const data_dims_t od_b = {os.w, rows, OC, 1};
      const conv_params_t cp_b = {0, 0, {S, S}, {P, 0}, {1, 1}, act};
      NN_CONV(s3k, &id_b, band_.data(), &fd, weights, bias32_.data(), &od_b,
              out.data() + (size_t)oy0 * os.w * OC, &cp_b, &q);
    }
    applyActivation(out, p.activation);
    return Status::Ok;
  }

  /// Depthwise 3x3 with ESP-NN. Our depthwise weights are [C][3*3]; ESP-NN reads [3][3][C], so the
  /// filter is repacked into a temporary buffer. The requantization is the convolution's.
  Status espNnDepthwise(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                        const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int C = is.c;
    const int S = p.stride;
    TINYYOLO_CHECK(prepareRequantSingle(in, w_scales, bias, out.scale(), C));

    Buffer<int8_t> filter;
    TINYYOLO_CHECK(filter.allocate((size_t)9 * C + 16, MemoryPlacement::PsramFirst));
    for (int c = 0; c < C; ++c) {
      for (int t = 0; t < 9; ++t) filter.data()[(size_t)t * C + c] = weights[(size_t)c * 9 + t];
    }

    const data_dims_t id = {is.w, is.h, C, 1};
    const data_dims_t fd = {3, 3, C, 1};
    const data_dims_t od = {os.w, os.h, C, 1};
    const act_params_t act = {-127, 127};
    const dw_conv_params_t cp = {0, 0, 1, {S, S}, {1, 1}, {1, 1}, act};
    const quant_data_t q = {shift_.data(), mult_.data()};
    const int scratch = NN_DW_SCRATCH(&id, &fd, &od, &cp);
    TINYYOLO_CHECK(reserveScratch(scratch > 0 ? (size_t)scratch : 0));
    NN_DW_SET_SCRATCH(scratch_);
    NN_DW(&id, in.data(), &fd, filter.data(), bias32_.data(), &od, out.data(), &cp, &q);
    applyActivation(out, p.activation);
    return Status::Ok;
  }

  /// 1x1 convolution: each pixel's channel vector is one fully-connected product. The filter
  /// of a 1x1 OHWI layer is [out][in], which is the fully-connected layout.
  Status espNnPointwise(const QTensor& in, const int8_t* weights, const float* w_scales,
                        const float* bias, const ConvParams& p, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int OC = os.c;
    const int IC = is.c;
    TINYYOLO_CHECK(prepareRequantSingle(in, w_scales, bias, out.scale(), OC));
    const size_t pixels = (size_t)is.h * is.w;
    for (size_t px = 0; px < pixels; ++px) {
      esp_nn_fully_connected_per_ch_s8(
          in.data() + px * IC, 0, (uint16_t)IC, weights, 0, bias32_.data(),
          out.data() + px * OC, (uint16_t)OC, 0, shift_.data(), mult_.data(), -127, 127);
    }
    applyActivation(out, p.activation);
    return Status::Ok;
  }

  Status reserveBand(size_t values) {
    if (values <= band_capacity_) return Status::Ok;
    TINYYOLO_CHECK(band_.allocate(values));
    band_capacity_ = values;
    return Status::Ok;
  }

  Status reserveChannels(int channels) {
    if (channel_capacity_ >= (size_t)channels) return Status::Ok;
    TINYYOLO_CHECK(mult_.allocate((size_t)channels));
    TINYYOLO_CHECK(shift_.allocate((size_t)channels));
    TINYYOLO_CHECK(bias32_.allocate((size_t)channels));
    channel_capacity_ = (size_t)channels;
    return Status::Ok;
  }

  Status reserveScratch(size_t bytes) {
    if (bytes <= scratch_size_) return Status::Ok;
    releaseScratch();
    if (bytes == 0) return Status::Ok;
    scratch_ = allocateMemory(bytes, false, scratch_in_psram_);  // same placement as Buffer
    if (scratch_ == nullptr) {
      TINYYOLO_LOGE("Esp32NN", "out of memory for the ESP-NN scratch buffer (%lu bytes)",
                    (unsigned long)bytes);
      return Status::OutOfMemory;
    }
    scratch_size_ = bytes;
    return Status::Ok;
  }

  void releaseScratch() {
    releaseMemory(scratch_, scratch_in_psram_);
    scratch_ = nullptr;
    scratch_in_psram_ = false;
    scratch_size_ = 0;
  }

  static constexpr size_t kScratchBudget = 512 * 1024;  ///< ESP-NN scratch per call, at most

  Buffer<int32_t> mult_;
  Buffer<int32_t> shift_;
  Buffer<int32_t> bias32_;
  size_t channel_capacity_ = 0;
  Buffer<int8_t> band_;  ///< input rows of one output band, zero-padded at the picture edges
  size_t band_capacity_ = 0;
  void* scratch_ = nullptr;
  bool scratch_in_psram_ = false;
  size_t scratch_size_ = 0;
#endif
};

}  // namespace tinyyolo
