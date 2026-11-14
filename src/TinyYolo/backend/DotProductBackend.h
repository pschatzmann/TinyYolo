#pragma once
/// @file DotProductBackend.h  Convolution as patch gathering plus dot products, multi-threaded.
///
/// Each output pixel gathers its receptive field into one contiguous patch, ordered like
/// the OIHW weights (input channel, then kernel row, then kernel column). Every output
/// channel is then a single dot product of two contiguous vectors, which is the operation
/// SIMD hardware is built for. Output rows are split across workers, each with its own
/// patch buffer.
///
/// Platforms only override dot(): the ESP32 backend uses esp-dsp, see Esp32Backend.h.

#include "TinyYolo/platform/KernelOptimize.h"
#include <stdint.h>

#include "TinyYolo/backend/ReferenceBackend.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/platform/Parallel.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace tinyyolo {

class DotProductBackend : public ReferenceBackend {
 public:
  /// workers = 0 uses hardwareWorkers().
  explicit DotProductBackend(int workers = 0) { setWorkers(workers); }

  const char* name() const override { return "dot-product"; }

  /// Number of threads used for convolutions (1 = run on the calling thread).
  void setWorkers(int workers) {
    if (workers <= 0) workers = hardwareWorkers();
    workers_ = workers > kMaxWorkers ? kMaxWorkers : workers;
  }
  int workers() const override { return workers_; }

  Status conv2d(const Tensor& in, const float* weights, const float* bias, const ConvParams& p,
                Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (weights == nullptr || bias == nullptr || p.out_channels <= 0) {
      TINYYOLO_LOGE("DotProduct", "conv2d: missing weights, bias or channel count");
      return Status::InvalidArgument;
    }
    if (os.c != p.out_channels || os.h != windowOutputSize(is.h, p.kernel, p.stride, p.pad) ||
        os.w != windowOutputSize(is.w, p.kernel, p.stride, p.pad)) {
      TINYYOLO_LOGE("DotProduct", "conv2d: output %dx%dx%d does not match input %dx%dx%d", os.h,
                    os.w, os.c, is.h, is.w, is.c);
      return Status::ShapeMismatch;
    }

    const int patch_len = is.c * p.kernel * p.kernel;
    const int workers = os.h < workers_ ? os.h : workers_;
    if (scratch_.size() < (size_t)workers * (size_t)patch_len) {
      TINYYOLO_CHECK(scratch_.allocate((size_t)workers * (size_t)patch_len));
    }

    ConvJob job = {this, &in, weights, bias, p, &out, scratch_.data(), patch_len};
    parallelRange(os.h, workers, &DotProductBackend::runRows, &job);
    return Status::Ok;
  }

  /// Int8 convolution: output channels are processed in tiles of kInt8Tile. Each tile is
  /// dequantized to float once per worker, and the float dot product (dot(), SIMD on the ESP32)
  /// does the work. This is much faster than a dot product on int8 values.
  Status conv2dInt8(const Tensor& in, const int8_t* weights, const float* scales,
                    const float* bias, const ConvParams& p, Tensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (weights == nullptr || scales == nullptr || bias == nullptr || p.out_channels <= 0) {
      TINYYOLO_LOGE("DotProduct", "conv2dInt8: missing weights, scales, bias or channel count");
      return Status::InvalidArgument;
    }
    if (os.c != p.out_channels || os.h != windowOutputSize(is.h, p.kernel, p.stride, p.pad) ||
        os.w != windowOutputSize(is.w, p.kernel, p.stride, p.pad)) {
      TINYYOLO_LOGE("DotProduct", "conv2dInt8: output %dx%dx%d does not match input %dx%dx%d",
                    os.h, os.w, os.c, is.h, is.w, is.c);
      return Status::ShapeMismatch;
    }
    const int patch_len = is.c * p.kernel * p.kernel;
    const int workers = os.h < workers_ ? os.h : workers_;
    // Per worker: one tile of float weights and one patch.
    const size_t per_worker = (size_t)(kInt8Tile + 1) * (size_t)patch_len;
    if (scratch_.size() < (size_t)workers * per_worker) {
      TINYYOLO_CHECK(scratch_.allocate((size_t)workers * per_worker));
    }
    Int8Job job = {this, &in, weights, scales, bias, p, &out, scratch_.data(), patch_len};
    parallelRange(os.h, workers, &DotProductBackend::runRowsInt8, &job);
    return Status::Ok;
  }

  /// Int8 convolution with exact int32 sums and one rounding, the same result as the reference kernel.
  /// On x86 with AVX2 (-mavx2) the rows of the kernel are dot products of 16 int8 values at a time; the
  /// scalar code is used otherwise. An interior output pixel's kernel row is one contiguous run of K*IC
  /// values in the input (HWC) and in the filter (OHWI).
  Status conv2dQ(const QTensor& in, const int8_t* weights, const float* w_scales, const float* bias,
                 const ConvParams& p, QTensor& out) override {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const bool valid = weights != nullptr && w_scales != nullptr && bias != nullptr && out.scale() > 0.0f &&
                       in.scale() > 0.0f && p.out_channels == os.c &&
                       os.h == windowOutputSize(is.h, p.kernel, p.stride, p.pad) &&
                       os.w == windowOutputSize(is.w, p.kernel, p.stride, p.pad);
    if (!valid) return ReferenceBackend::conv2dQ(in, weights, w_scales, bias, p, out);
    const int K = p.kernel, S = p.stride, P = p.pad, IC = is.c, OC = os.c;
    const int IH = is.h, IW = is.w;
    const size_t row = (size_t)IW * IC;
    const size_t wstride = (size_t)K * K * IC;
    const float inv_out = 1.0f / out.scale();
    const int8_t* x = in.data();
    int8_t* o_all = out.data();
    // Dequantization per output channel, as in the reference (bias + acc * in_scale * w_scale).
    float* deq = nullptr;
    Buffer<float> deq_buf;
    if (deq_buf.allocate((size_t)OC) != Status::Ok) return Status::OutOfMemory;
    deq = deq_buf.data();
    for (int oc = 0; oc < OC; ++oc) deq[oc] = in.scale() * w_scales[oc];
    for (int oy = 0; oy < os.h; ++oy) {
      const int y0 = oy * S - P;
      const int ky0 = y0 < 0 ? -y0 : 0;
      const int ky1 = y0 + K > IH ? IH - y0 : K;
      for (int ox = 0; ox < os.w; ++ox) {
        const int x0 = ox * S - P;
        const int kx0 = x0 < 0 ? -x0 : 0;
        const int kx1 = x0 + K > IW ? IW - x0 : K;
        const bool full_x = kx0 == 0 && kx1 == K;
        int8_t* o = o_all + ((size_t)oy * os.w + ox) * OC;
        for (int oc = 0; oc < OC; ++oc) {
          const int8_t* wo = weights + (size_t)oc * wstride;
          int32_t acc = 0;
          for (int ky = ky0; ky < ky1; ++ky) {
            const int8_t* xr = x + (size_t)(y0 + ky) * row;
            const int8_t* wr = wo + (size_t)ky * K * IC;
            if (full_x) {
              acc += dotI8(xr + (size_t)x0 * IC, wr, K * IC);
            } else {
              for (int kx = kx0; kx < kx1; ++kx) {
                acc += dotI8(xr + (size_t)(x0 + kx) * IC, wr + (size_t)kx * IC, IC);
              }
            }
          }
          const float real = activate(p.activation, bias[oc] + (float)acc * deq[oc]);
          o[oc] = quantizeValue(real, inv_out);
        }
      }
    }
    return Status::Ok;
  }

  /// Exact int32 dot product of two int8 vectors of length n (AVX2 when available).
  static int32_t dotI8(const int8_t* a, const int8_t* b, int n) {
    int i = 0;
    int32_t acc = 0;
#if defined(__AVX2__)
    __m256i sum = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
      const __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)));
      const __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i)));
      sum = _mm256_add_epi32(sum, _mm256_madd_epi16(va, vb));
    }
    __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    acc = _mm_cvtsi128_si32(s4);
#endif
    for (; i < n; ++i) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
  }

 protected:
  /// Dot product of two contiguous vectors of length n. Platform backends override this.
  virtual float dot(const float* a, const float* b, int n) const {
    // Eight independent partial sums let the compiler vectorize the loop without
    // -ffast-math (a single running sum is a dependency chain it must not reorder).
    float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      for (int k = 0; k < 8; ++k) lane[k] += a[i + k] * b[i + k];
    }
    float acc = 0.0f;
    for (int k = 0; k < 8; ++k) acc += lane[k];
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
  }
  /// Dot product of a float vector with an int8 vector (weights-only quantization). The
  /// caller applies the per-channel scale. Platform backends override this.
  virtual float dotInt8(const float* a, const int8_t* w, int n) const {
    float lane[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      for (int k = 0; k < 8; ++k) lane[k] += a[i + k] * (float)w[i + k];
    }
    float acc = 0.0f;
    for (int k = 0; k < 8; ++k) acc += lane[k];
    for (; i < n; ++i) acc += a[i] * (float)w[i];
    return acc;
  }

 private:
  struct ConvJob {
    DotProductBackend* self;
    const Tensor* in;
    const float* weights;
    const float* bias;
    ConvParams params;
    Tensor* out;
    float* scratch;
    int patch_len;
  };

  static constexpr int kInt8Tile = 4;  ///< output channels dequantized at a time

  struct Int8Job {
    DotProductBackend* self;
    const Tensor* in;
    const int8_t* weights;
    const float* scales;
    const float* bias;
    ConvParams params;
    Tensor* out;
    float* scratch;
    int patch_len;
  };

  static void runRowsInt8(void* ctx, int y0, int y1, int worker) {
    const Int8Job& j = *static_cast<const Int8Job*>(ctx);
    const Tensor& in = *j.in;
    Tensor& out = *j.out;
    const ConvParams& p = j.params;
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int K = p.kernel;
    const int plen = j.patch_len;
    float* tile = j.scratch + (size_t)worker * (size_t)(kInt8Tile + 1) * (size_t)plen;
    float* patch = tile + (size_t)kInt8Tile * (size_t)plen;

    for (int t0 = 0; t0 < os.c; t0 += kInt8Tile) {
      const int n = os.c - t0 < kInt8Tile ? os.c - t0 : kInt8Tile;
      for (int r = 0; r < n; ++r) {
        const int8_t* w = j.weights + (size_t)(t0 + r) * (size_t)plen;
        const float s = j.scales[t0 + r];
        float* dst = tile + (size_t)r * (size_t)plen;
        for (int i = 0; i < plen; ++i) dst[i] = (float)w[i] * s;
      }
      for (int oy = y0; oy < y1; ++oy) {
        const int iy0 = oy * p.stride - p.pad;
        for (int ox = 0; ox < os.w; ++ox) {
          const int ix0 = ox * p.stride - p.pad;
          // Patch order (ky, kx, ic) matches the OHWI weight rows of the tile.
          int k = 0;
          for (int ky = 0; ky < K; ++ky) {
            const int iy = iy0 + ky;
            const bool row_ok = iy >= 0 && iy < is.h;
            for (int kx = 0; kx < K; ++kx) {
              const int ix = ix0 + kx;
              const bool ok = row_ok && ix >= 0 && ix < is.w;
              for (int ic = 0; ic < is.c; ++ic) {
                patch[k++] = ok ? in.at(iy, ix, ic) : 0.0f;
              }
            }
          }
          for (int r = 0; r < n; ++r) {
            const float acc = j.bias[t0 + r] + j.self->dot(patch, tile + (size_t)r * (size_t)plen, plen);
            out.at(oy, ox, t0 + r) = activate(p.activation, acc);
          }
        }
      }
    }
  }

  static void runRows(void* ctx, int y0, int y1, int worker) {
    const ConvJob& j = *static_cast<const ConvJob*>(ctx);
    const Tensor& in = *j.in;
    Tensor& out = *j.out;
    const ConvParams& p = j.params;
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int K = p.kernel;
    float* patch = j.scratch + (size_t)worker * (size_t)j.patch_len;

    for (int oy = y0; oy < y1; ++oy) {
      const int iy0 = oy * p.stride - p.pad;
      for (int ox = 0; ox < os.w; ++ox) {
        const int ix0 = ox * p.stride - p.pad;
        // Gather the receptive field in OIHW order: channel, kernel row, kernel column.
        int k = 0;
        for (int ic = 0; ic < is.c; ++ic) {
          for (int ky = 0; ky < K; ++ky) {
            const int iy = iy0 + ky;
            const bool row_ok = iy >= 0 && iy < is.h;
            for (int kx = 0; kx < K; ++kx) {
              const int ix = ix0 + kx;
              patch[k++] = (row_ok && ix >= 0 && ix < is.w) ? in.at(iy, ix, ic) : 0.0f;
            }
          }
        }
        for (int oc = 0; oc < os.c; ++oc) {
          const float* w = j.weights + (size_t)oc * (size_t)j.patch_len;
          const float dotp = j.self->dot(patch, w, j.patch_len);
          out.at(oy, ox, oc) = activate(p.activation, j.bias[oc] + dotp);
        }
      }
    }
  }

  int workers_ = 1;
  Buffer<float> scratch_;
};

}  // namespace tinyyolo
