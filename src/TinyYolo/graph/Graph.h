#pragma once
/// @file Graph.h  Runs a network given as a list of primitive operations on tensors.
///
/// The Darknet description cannot express every network. YOLO11 needs C3k2 and SPPF blocks
/// (these are built from the operations below), depthwise convolutions, attention (C2PSA),
/// and its anchor-free head. A Graph is such a network: an operation list, the parameters in
/// the order the operations use them, and the tensor ids that connect them. Tensors are float
/// HWC, as elsewhere in the library; convolution weights are OIHW (grouped: [out][in/groups]).

#include <math.h>
#include <string.h>

#include "TinyYolo/backend/Backend.h"
#include "TinyYolo/graph/AttentionKernel.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/core/Types.h"
#include "TinyYolo/platform/Clock.h"
#include "TinyYolo/platform/KernelOptimize.h"

namespace tinyyolo {

enum class GraphOpKind : uint8_t { Conv, Add, Chunk, Concat, MaxPool, Upsample, Attention };

/// One operation. Unused fields are 0 or -1. Field order matches the generated tables.
struct GraphOp {
  GraphOpKind kind;
  uint8_t n_in;        ///< number of inputs (in[0 .. n_in-1])
  int in[4];           ///< input tensor ids
  int out;             ///< output tensor id
  int w, b;            ///< Conv: weight and bias parameters. Attention: qkv weight and bias
  int k, s, p;         ///< Conv: kernel, stride, padding. MaxPool: kernel, stride, total padding
  int g;               ///< Conv: groups
  bool silu;           ///< Conv: SiLU activation (otherwise linear)
  int index;           ///< Chunk: which half (0 or 1) of the channels
  int w2, b2;          ///< Attention: positional-encoding (depthwise) weight and bias
  int w3, b3;          ///< Attention: projection weight and bias
  int heads, key_dim, head_dim;  ///< Attention
  int q;                         ///< Conv: index into the quantization table, or -1 (float)
  int q2 = -1, q3 = -1;          ///< Attention: quantization entries of the qkv and projection convolutions
};

/// Int8 convolution: where its weights (OHWI int8) and scales are, and its activation scales.
struct GraphQuant {
  int w_offset;       ///< offset (in int8 values) of the weights
  int s_offset;       ///< offset (in floats) of the per-output-channel weight scales
  float in_scale;     ///< scale of the input activations
  float out_scale;    ///< scale of the output activations
};

/// A parameter tensor: its offset (in floats) in the weight array, and its dimensions.
struct GraphParam {
  int offset;
  int rank;
  int dims[4];
};

class Graph {
 public:
  Graph() {}
  ~Graph() {
    delete[] tensors_;
    delete[] q8_;
  }
  Graph(const Graph&) = delete;
  Graph& operator=(const Graph&) = delete;

  /// Sets the operations. The arrays and `weights` must outlive the graph (in flash on boards).
  /// The tensors listed in `keep` stay in memory after run(); the others are freed after the
  /// last operation that reads them, which keeps the peak memory low.
  Status begin(const GraphOp* ops, int op_count, const GraphParam* params, int param_count,
               const float* weights, int tensor_count, int input_id, const int* keep = nullptr,
               int keep_count = 0, const int8_t* qweights = nullptr, const float* qscales = nullptr,
               const GraphQuant* quant = nullptr) {
    if (ops == nullptr || params == nullptr || weights == nullptr || op_count <= 0 ||
        tensor_count <= 0 || input_id < 0 || input_id >= tensor_count) {
      TINYYOLO_LOGE("Graph", "begin: missing operations, parameters or weights");
      return Status::InvalidArgument;
    }
    delete[] tensors_;
    tensors_ = new Tensor[(size_t)tensor_count];
    if (tensors_ == nullptr) return Status::OutOfMemory;
    TINYYOLO_CHECK(last_use_.allocate((size_t)tensor_count));
    TINYYOLO_CHECK(kept_.allocate((size_t)tensor_count));
    for (int t = 0; t < tensor_count; ++t) last_use_[(size_t)t] = -1;
    for (int i = 0; i < op_count; ++i) {
      const GraphOp& op = ops[i];
      for (int j = 0; j < (int)op.n_in; ++j) {
        if (op.in[j] >= 0 && op.in[j] < tensor_count) last_use_[(size_t)op.in[j]] = i;
      }
    }
    for (int k = 0; k < keep_count; ++k) {
      if (keep[k] >= 0 && keep[k] < tensor_count) kept_[(size_t)keep[k]] = 1;
    }
    ops_ = ops;
    op_count_ = op_count;
    if (profiling_) enableProfiling(true);
    delete[] q8_;
    q8_ = new QTensor[(size_t)tensor_count];
    if (q8_ == nullptr) return Status::OutOfMemory;
    TINYYOLO_CHECK(int8_only_.allocate((size_t)tensor_count));
    TINYYOLO_CHECK(producer_.allocate((size_t)tensor_count));
    TINYYOLO_CHECK(tscale_.allocate((size_t)tensor_count));
    TINYYOLO_CHECK(q8_ready_.allocate((size_t)tensor_count));
    for (int t = 0; t < tensor_count; ++t) q8_ready_[(size_t)t] = 0;
    qweights_ = qweights;
    qscales_ = qscales;
    quant_ = quant;
    params_ = params;
    param_count_ = param_count;
    weights_ = weights;
    tensor_count_ = tensor_count;
    input_id_ = input_id;
    computeInt8Only();
    return Status::Ok;
  }

  /// Runs every operation on `input` (HWC). The kept tensors stay available afterwards (see
  /// tensor()).
  Status run(Backend& backend, const Tensor& input) {
    if (tensors_ == nullptr) return Status::NotInitialized;
    // Free what the previous run left behind, so that this run starts from the same memory.
    for (int t = 0; t < tensor_count_; ++t) {
      tensors_[t].release();
      q8_[t].release();
      q8_ready_[(size_t)t] = 0;
    }
    if (profiling_) resetProfile();  // the times describe this run only
    TINYYOLO_CHECK(tensors_[input_id_].allocate(input.shape()));
    memcpy(tensors_[input_id_].data(), input.data(), input.size() * sizeof(float));
    for (int i = 0; i < op_count_; ++i) {
      const GraphOp& op = ops_[i];
      Status status = Status::Ok;
      const uint32_t t0 = tick();
      switch (op.kind) {
        case GraphOpKind::Conv: status = conv(backend, op); break;
        case GraphOpKind::Add:
          if (int8_only_[(size_t)op.out]) {
            status = addInt8(op);
            break;
          }
          status = withOut(op.out, tensors_[op.in[0]].shape(),
                           [&](Tensor& out) { return backend.add(tensors_[op.in[0]], tensors_[op.in[1]], Activation::Linear, out); });
          break;
        case GraphOpKind::Chunk: status = int8_only_[(size_t)op.out] ? chunkInt8(op) : chunk(backend, op); break;
        case GraphOpKind::Concat: status = int8_only_[(size_t)op.out] ? concatInt8(op) : concat(backend, op); break;
        case GraphOpKind::MaxPool: status = int8_only_[(size_t)op.out] ? maxPoolInt8(backend, op) : maxPool(backend, op); break;
        case GraphOpKind::Upsample:
          if (int8_only_[(size_t)op.out]) {
            status = upsampleInt8(op);
            break;
          }
          status = withOut(op.out, scaled(tensors_[op.in[0]].shape(), 2), [&](Tensor& out) {
            return backend.upsample(tensors_[op.in[0]], 2, out);
          });
          break;
        case GraphOpKind::Attention: status = attention(backend, op); break;
      }
      if (profiling_) {
        const uint32_t us = nowMicros() - t0;
        op_us_[(size_t)i] = us;
        // Int8 convolutions add their own phases (quantize, int8 conv, dequantize).
        if (!isQuantized(op)) addPhase(phaseOf(op), us);
      }
      if (status != Status::Ok) {
        TINYYOLO_LOGE("Graph", "operation %d failed: %s", i, statusName(status));
        return status;
      }
      // Free the inputs that this operation reads for the last time (unless they are kept).
      for (int j = 0; j < (int)op.n_in; ++j) {
        const int t = op.in[j];
        if (t >= 0 && t < tensor_count_ && last_use_[(size_t)t] == i && !kept_[(size_t)t]) {
          tensors_[t].release();
          q8_[t].release();
          q8_ready_[(size_t)t] = 0;
        }
      }
    }
    return Status::Ok;
  }

  /// Tensor `id` after run().
  const Tensor& tensor(int id) const { return tensors_[id]; }

  /// Where the time of run() goes. Phases of int8 convolutions are measured inside them.
  enum class Phase : int { Quantize, Int8Conv, Dequantize, FloatConv, Grouped, Attention, Other, Count };

  /// Turns the timers on (they cost two clock reads per operation). The per-operation and
  /// per-phase times are reset by each call and refreshed by run().
  Status enableProfiling(bool on) {
    profiling_ = on;
    resetProfile();
    if (on && op_count_ > 0) {
      op_us_.release();
      TINYYOLO_CHECK(op_us_.allocate((size_t)op_count_));
    }
    return Status::Ok;
  }
  void resetProfile() {
    for (uint32_t& us : phase_us_) us = 0;
    for (size_t i = 0; i < op_us_.size(); ++i) op_us_[i] = 0;
  }
  /// Microseconds of operation `i` in the last run() (0 when profiling is off).
  uint32_t opMicros(int i) const { return i >= 0 && (size_t)i < op_us_.size() ? op_us_[(size_t)i] : 0; }
  uint32_t phaseMicros(Phase p) const { return phase_us_[(int)p]; }
  static const char* phaseName(Phase p) {
    static const char* const names[] = {"quantize", "int8 conv", "dequantize", "float conv",
                                        "grouped conv", "attention", "other"};
    return names[(int)p];
  }

 private:
  uint32_t tick() const { return profiling_ ? nowMicros() : 0; }
  void addPhase(Phase p, uint32_t us) { phase_us_[(int)p] += us; }
  void addSince(Phase p, uint32_t t0) {
    if (profiling_) phase_us_[(int)p] += nowMicros() - t0;
  }
  bool isQuantized(const GraphOp& op) const {
    return op.kind == GraphOpKind::Conv && op.q >= 0 && qweights_ != nullptr && quant_ != nullptr && op.g == 1;
  }

  /// True for an int8 depthwise convolution (one filter per channel: groups == output channels).
  bool isQuantizedDepthwise(const GraphOp& op) const {
    return op.kind == GraphOpKind::Conv && op.q >= 0 && qweights_ != nullptr && quant_ != nullptr && op.g > 1 &&
           params_[op.w].dims[0] == op.g;
  }

  /// Any int8 convolution (regular or depthwise).
  bool isQuantizedAny(const GraphOp& op) const { return isQuantized(op) || isQuantizedDepthwise(op); }
  Phase phaseOf(const GraphOp& op) const {
    switch (op.kind) {
      case GraphOpKind::Conv: return op.g == 1 ? Phase::FloatConv : Phase::Grouped;
      case GraphOpKind::Attention: return Phase::Attention;
      default: return Phase::Other;
    }
  }

  static Shape scaled(const Shape& s, int factor) { return Shape(s.h * factor, s.w * factor, s.c); }

  const float* weight(int index) const { return weights_ + params_[index].offset; }
  size_t count(int index) const {
    size_t n = 1;
    for (int d = 0; d < params_[index].rank; ++d) n *= (size_t)params_[index].dims[d];
    return n;
  }

  /// Allocates tensor `id` with `shape` and calls `fn` to fill it.
  template <typename Fn>
  Status withOut(int id, const Shape& shape, Fn fn) {
    TINYYOLO_CHECK(tensors_[id].allocate(shape));
    return fn(tensors_[id]);
  }

  Status conv(Backend& backend, const GraphOp& op) {
    const int oc = params_[op.w].dims[0];
    const int icg = params_[op.w].dims[1];  // input channels per group
    const Activation act = op.silu ? Activation::SiLU : Activation::Linear;
    if (isQuantizedAny(op)) {
      const bool in_int8 = q8_ready_[(size_t)op.in[0]] != 0;
      const Shape is = in_int8 ? q8_[op.in[0]].shape() : tensors_[op.in[0]].shape();
      const Shape os(windowOutputSize(is.h, op.k, op.s, op.p), windowOutputSize(is.w, op.k, op.s, op.p), oc);
      return quantizedConv(backend, op, in_int8, os, act);
    }
    const Tensor& in = tensors_[op.in[0]];
    const Shape os(windowOutputSize(in.shape().h, op.k, op.s, op.p),
                   windowOutputSize(in.shape().w, op.k, op.s, op.p), oc);
    return withOut(op.out, os, [&](Tensor& out) {
      if (op.g == 1) {
        const ConvParams p = {op.k, op.s, op.p, oc, act};
        return backend.conv2d(in, weight(op.w), weight(op.b), p, out);
      }
      if (icg * op.g != in.shape().c) return Status::ShapeMismatch;
      groupedConv(in, weight(op.w), weight(op.b), oc, op.k, op.s, op.p, op.g, out);
      if (op.silu) {
        for (size_t i = 0; i < out.size(); ++i) out.data()[i] = activate(act, out.data()[i]);
      }
      return Status::Ok;
    });
  }

  /// 1x1 int8 convolution of the float tensor `in` with the quantization entry `q_index` (weights
  /// per output channel). `out` (already allocated, float) receives the dequantized result.
  Status int8Pointwise(Backend& backend, const Tensor& in, int q_index, const float* bias, Tensor& out) {
    const GraphQuant& q = quant_[q_index];
    QTensor qin, qout;
    TINYYOLO_CHECK(qin.allocate(in.shape(), q.in_scale, MemoryPlacement::PsramFirst));
    TINYYOLO_CHECK(quantizePadded(in, q.in_scale, qin));
    TINYYOLO_CHECK(qout.allocate(out.shape(), q.out_scale, MemoryPlacement::PsramFirst));
    const ConvParams p = {1, 1, 0, out.shape().c, Activation::Linear};
    TINYYOLO_CHECK(backend.conv2dQ(qin, qweights_ + q.w_offset, qscales_ + q.s_offset, bias, p, qout));
    return dequantizeTensor(qout, out);
  }

  /// Quantizes `in` (HWC) into `out`, which may have more channels than `in`: the extra channels
  /// stay zero (QTensor is zero-initialised).
  /// Quantizes `in` into `out`, which is `pad` pixels larger on every side with zeros around it.
  /// The extra channels of `out` (beyond in's channels) are zero too.
  static Status quantizeBordered(const Tensor& in, float scale, int pad, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (!(scale > 0.0f) || os.h != is.h + 2 * pad || os.w != is.w + 2 * pad || os.c < is.c) {
      return Status::ShapeMismatch;
    }
    out.setScale(scale);
    int8_t* dst = out.data();
    memset(dst, 0, out.size());
    const float inv = 1.0f / scale;
    const float* src = in.data();
    for (int y = 0; y < is.h; ++y) {
      for (int x = 0; x < is.w; ++x) {
        int8_t* d = dst + (((size_t)(y + pad) * os.w + (size_t)(x + pad)) * os.c);
        const float* s = src + (((size_t)y * is.w + (size_t)x) * is.c);
        for (int c = 0; c < is.c; ++c) d[c] = quantizeValue(s[c], inv);
      }
    }
    return Status::Ok;
  }

  static Status quantizePadded(const Tensor& in, float scale, QTensor& out) {
    const Shape is = in.shape();
    if (!(scale > 0.0f) || out.shape().h != is.h || out.shape().w != is.w || out.shape().c < is.c) {
      return Status::ShapeMismatch;
    }
    out.setScale(scale);
    const float inv = 1.0f / scale;
    const size_t pixels = (size_t)is.h * is.w;
    const size_t icp = (size_t)out.shape().c;
    const float* src = in.data();
    int8_t* dst = out.data();
    for (size_t p = 0; p < pixels; ++p) {
      for (int c = 0; c < is.c; ++c) dst[p * icp + c] = quantizeValue(src[p * is.c + c], inv);
      for (size_t c = (size_t)is.c; c < icp; ++c) dst[p * icp + c] = 0;  // padding channels
    }
    return Status::Ok;
  }

  /// A tensor stays int8 between two convolutions when one int8 convolution produces it and only
  /// int8 convolutions read it, all with the same scale (the calibration gives one scale per tensor),
  /// and the reader's input channels are a multiple of 4. Such a tensor is never converted to float.
  void computeInt8Only() {
    // The producer of each tensor (an int8 convolution or a concatenation) and the scale of the
    // tensor when it is int8 (a concatenation keeps the scale of its inputs).
    for (int t = 0; t < tensor_count_; ++t) {
      producer_[(size_t)t] = -1;
      tscale_[(size_t)t] = 0.0f;
      int8_only_[(size_t)t] = 0;
    }
    for (int i = 0; i < op_count_; ++i) {
      const GraphOp& op = ops_[i];
      const bool pass_through = isPassThrough(op.kind);
      if (isQuantizedAny(op) || pass_through) producer_[(size_t)op.out] = i;
      if (isQuantizedAny(op)) {
        tscale_[(size_t)op.out] = quant_[op.q].out_scale;
      } else if (pass_through && op.n_in > 0 && op.in[0] >= 0 && op.in[0] < tensor_count_) {
        tscale_[(size_t)op.out] = tscale_[(size_t)op.in[0]];
      }
    }
    for (int t = 0; t < tensor_count_; ++t) {
      if (producer_[(size_t)t] >= 0 && !kept_[(size_t)t]) int8_only_[(size_t)t] = 1;
    }
    // Remove the tensors whose readers cannot take int8, until nothing changes.
    for (int pass = 0; pass < tensor_count_ + 1; ++pass) {
      bool changed = false;
      for (int t = 0; t < tensor_count_; ++t) {
        if (!int8_only_[(size_t)t]) continue;
        bool ok = true, any = false;
        const GraphOp& prod = ops_[producer_[(size_t)t]];
        if (isPassThrough(prod.kind)) {
          for (int k = 0; k < (int)prod.n_in; ++k) {
            if (prod.in[k] < 0 || !int8_only_[(size_t)prod.in[k]]) ok = false;
          }
        }
        for (int j = 0; j < op_count_ && ok; ++j) {
          const GraphOp& c = ops_[j];
          for (int k = 0; k < (int)c.n_in && ok; ++k) {
            if (c.in[k] != t) continue;
            any = true;
            if (isQuantizedAny(c)) {
              const bool channels_ok = isQuantized(c) ? params_[c.w].dims[1] % 4 == 0 : true;
              if (k != 0 || !channels_ok || quant_[c.q].in_scale != tscale_[(size_t)t]) ok = false;
            } else if (isPassThrough(c.kind)) {
              if (!int8_only_[(size_t)c.out]) ok = false;
              // Every input of the operation must be int8 at the same scale (a sum or a concatenation).
              for (int m = 0; m < (int)c.n_in; ++m) {
                if (c.in[m] < 0 || !int8_only_[(size_t)c.in[m]] || tscale_[(size_t)c.in[m]] != tscale_[(size_t)t]) {
                  ok = false;
                }
              }
            } else {
              ok = false;
            }
          }
        }
        if (!any) ok = false;
        if (!ok) {
          int8_only_[(size_t)t] = 0;
          changed = true;
        }
      }
      if (!changed) break;
    }
  }

  /// Int8 convolution. The input is the int8 activation of the previous convolution when it is kept
  /// in int8 (`in_int8`), otherwise the float tensor is quantized here (padded to a multiple of 4
  /// channels). The output stays int8 when only int8 convolutions read it, otherwise it is
  /// dequantized to float.
  Status quantizedConv(Backend& backend, const GraphOp& op, bool in_int8, const Shape& os, Activation act) {
    const GraphQuant& q = quant_[op.q];
    const int oc = os.c;
    const bool depthwise = isQuantizedDepthwise(op);
    const bool out_int8 = int8_only_[(size_t)op.out] != 0;
    const Shape is = in_int8 ? q8_[op.in[0]].shape() : tensors_[op.in[0]].shape();
    // The first conv (3 input channels) is padded to 16: the P4 kernels read 16 channels per pixel.
    const int ic = is.c, icp = depthwise ? ic : backend.inputChannelPadding(ic);
    // A float input with padding and extra channels gets its zero border here, so the backend
    // needs no padded copy of its own (that copy doubled the peak memory of the first conv).
    const int border = (!in_int8 && !depthwise && icp != ic) ? op.p : 0;
    QTensor qin_local;
    const QTensor* qin = nullptr;
    if (in_int8) {
      qin = &q8_[op.in[0]];
    } else {
      // The activations go to PSRAM first: they are large, and the P4 kernels need aligned memory.
      const Shape bs(is.h + 2 * border, is.w + 2 * border, icp);
      TINYYOLO_CHECK(qin_local.allocate(bs, q.in_scale, MemoryPlacement::PsramFirst));
      uint32_t t = tick();
      TINYYOLO_CHECK(quantizeBordered(tensors_[op.in[0]], q.in_scale, border, qin_local));
      addSince(Phase::Quantize, t);
      qin = &qin_local;
    }
    const int8_t* w8 = qweights_ + q.w_offset;
    Buffer<int8_t> padded;
    if (!depthwise && icp != ic) {
      // Weights are OHWI: one row of ic values per filter tap and output channel.
      TINYYOLO_CHECK(padded.allocate((size_t)oc * op.k * op.k * icp));
      const size_t rows = (size_t)oc * op.k * op.k;
      for (size_t r = 0; r < rows; ++r) {
        for (int c = 0; c < icp; ++c) padded[r * icp + c] = c < ic ? w8[r * ic + c] : 0;
      }
      w8 = padded.data();
    }
    QTensor qout_local;
    QTensor& qout = out_int8 ? q8_[op.out] : qout_local;
    if (out_int8) q8_[op.out].release();
    TINYYOLO_CHECK(qout.allocate(os, q.out_scale, MemoryPlacement::PsramFirst));
    const ConvParams p = {op.k, op.s, border > 0 ? 0 : op.p, oc, act};
    uint32_t t = tick();
    const Status conv_status =
        depthwise ? backend.depthwiseConv2dQ(*qin, w8, qscales_ + q.s_offset, weight(op.b), p, qout)
                  : backend.conv2dQ(*qin, w8, qscales_ + q.s_offset, weight(op.b), p, qout);
    TINYYOLO_CHECK(conv_status);
    addSince(Phase::Int8Conv, t);
    if (out_int8) {
      q8_ready_[(size_t)op.out] = 1;
      return Status::Ok;
    }
    return withOut(op.out, os, [&](Tensor& out) {
      t = tick();
      const Status status = dequantizeTensor(qout, out);
      addSince(Phase::Dequantize, t);
      return status;
    });
  }

  /// Convolution with groups (depthwise when groups == input channels).
  static void groupedConv(const Tensor& in, const float* w, const float* bias, int oc, int k, int s,
                          int p, int groups, Tensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    const int icg = is.c / groups;
    const int ocg = oc / groups;
    // The same sums in the same order as before, with the input indexed directly (no per-element
    // bounds check through Tensor::at).
    const float* src = in.data();
    float* dst = out.data();
    const size_t row = (size_t)is.w * is.c;
    for (int o = 0; o < oc; ++o) {
      const int gi = o / ocg;
      for (int y = 0; y < os.h; ++y) {
        for (int x = 0; x < os.w; ++x) {
          float acc = bias[o];
          for (int c = 0; c < icg; ++c) {
            const float* wk = w + ((size_t)o * icg + c) * k * k;
            const int ch = gi * icg + c;
            for (int ky = 0; ky < k; ++ky) {
              const int iy = y * s - p + ky;
              if (iy < 0 || iy >= is.h) continue;
              const float* in_row = src + (size_t)iy * row;
              for (int kx = 0; kx < k; ++kx) {
                const int ix = x * s - p + kx;
                if (ix < 0 || ix >= is.w) continue;
                acc += in_row[(size_t)ix * is.c + ch] * wk[ky * k + kx];
              }
            }
          }
          dst[((size_t)y * os.w + x) * os.c + o] = acc;
        }
      }
    }
  }

  Status chunk(Backend& backend, const GraphOp& op) {
    const Tensor& in = tensors_[op.in[0]];
    const int half = in.shape().c / 2;
    return withOut(op.out, Shape(in.shape().h, in.shape().w, half), [&](Tensor& out) {
      return backend.slice(in, op.index * half, half, out);
    });
  }

  /// Channel concatenation of two inputs. (Longer lists are split into pairs by the generator.)
  /// Operations whose int8 form keeps the scale of their input(s).
  static bool isPassThrough(GraphOpKind kind) {
    return kind == GraphOpKind::Concat || kind == GraphOpKind::Chunk || kind == GraphOpKind::MaxPool ||
           kind == GraphOpKind::Upsample || kind == GraphOpKind::Add;
  }

  /// Saturating sum of two int8 tensors with the same scale.
  Status addInt8(const GraphOp& op) {
    if (!q8_ready_[(size_t)op.in[0]] || !q8_ready_[(size_t)op.in[1]]) return Status::InvalidModel;
    const QTensor& a = q8_[op.in[0]];
    const QTensor& b = q8_[op.in[1]];
    if (a.shape() != b.shape() || a.scale() != b.scale()) {
      TINYYOLO_LOGE("Graph", "int8 add needs equal shapes and scales");
      return Status::InvalidModel;
    }
    q8_[op.out].release();
    TINYYOLO_CHECK(q8_[op.out].allocate(a.shape(), a.scale(), MemoryPlacement::PsramFirst));
    int8_t* d = q8_[op.out].data();
    for (size_t i = 0; i < a.size(); ++i) {
      int v = (int)a.data()[i] + (int)b.data()[i];
      d[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
    }
    q8_ready_[(size_t)op.out] = 1;
    return Status::Ok;
  }

  /// Max pooling of an int8 tensor (the same window and padding as the float operation).
  Status maxPoolInt8(Backend& backend, const GraphOp& op) {
    if (!q8_ready_[(size_t)op.in[0]]) return Status::InvalidModel;
    const QTensor& in = q8_[op.in[0]];
    const Shape is = in.shape();
    const Shape os(poolOutputSize(is.h, op.k, op.s, op.p), poolOutputSize(is.w, op.k, op.s, op.p), is.c);
    q8_[op.out].release();
    TINYYOLO_CHECK(q8_[op.out].allocate(os, in.scale(), MemoryPlacement::PsramFirst));
    TINYYOLO_CHECK(backend.maxPoolQ(in, op.k, op.s, op.p, q8_[op.out]));
    q8_ready_[(size_t)op.out] = 1;
    return Status::Ok;
  }

  /// Nearest-neighbour upsampling by 2 of an int8 tensor.
  Status upsampleInt8(const GraphOp& op) {
    if (!q8_ready_[(size_t)op.in[0]]) return Status::InvalidModel;
    const QTensor& in = q8_[op.in[0]];
    const Shape is = in.shape();
    const Shape os(is.h * 2, is.w * 2, is.c);
    q8_[op.out].release();
    TINYYOLO_CHECK(q8_[op.out].allocate(os, in.scale(), MemoryPlacement::PsramFirst));
    const int8_t* src = in.data();
    int8_t* dst = q8_[op.out].data();
    for (int y = 0; y < os.h; ++y) {
      for (int x = 0; x < os.w; ++x) {
        memcpy(dst + ((size_t)y * os.w + x) * is.c, src + ((size_t)(y / 2) * is.w + x / 2) * is.c, (size_t)is.c);
      }
    }
    q8_ready_[(size_t)op.out] = 1;
    return Status::Ok;
  }

  /// Channel half of an int8 tensor (the scale is unchanged).
  Status chunkInt8(const GraphOp& op) {
    if (!q8_ready_[(size_t)op.in[0]]) return Status::InvalidModel;
    const QTensor& in = q8_[op.in[0]];
    const Shape is = in.shape();
    const int half = is.c / 2;
    const Shape os(is.h, is.w, half);
    q8_[op.out].release();
    TINYYOLO_CHECK(q8_[op.out].allocate(os, in.scale(), MemoryPlacement::PsramFirst));
    const int8_t* src = in.data() + (size_t)op.index * half;
    int8_t* dst = q8_[op.out].data();
    const size_t pixels = (size_t)is.h * is.w;
    for (size_t p = 0; p < pixels; ++p) memcpy(dst + p * half, src + p * is.c, (size_t)half);
    q8_ready_[(size_t)op.out] = 1;
    return Status::Ok;
  }

  /// Concatenation of two int8 tensors that share one scale: the channels are copied per pixel.
  Status concatInt8(const GraphOp& op) {
    if (!q8_ready_[(size_t)op.in[0]] || !q8_ready_[(size_t)op.in[1]]) return Status::InvalidModel;
    const QTensor& a = q8_[op.in[0]];
    const QTensor& b = q8_[op.in[1]];
    const Shape as = a.shape(), bs = b.shape();
    if (as.h != bs.h || as.w != bs.w || a.scale() != b.scale()) {
      TINYYOLO_LOGE("Graph", "int8 concatenation needs equal sizes and scales");
      return Status::InvalidModel;
    }
    const Shape os(as.h, as.w, as.c + bs.c);
    q8_[op.out].release();
    TINYYOLO_CHECK(q8_[op.out].allocate(os, a.scale(), MemoryPlacement::PsramFirst));
    int8_t* d = q8_[op.out].data();
    const int8_t* pa = a.data();
    const int8_t* pb = b.data();
    const size_t pixels = (size_t)os.h * os.w;
    for (size_t p = 0; p < pixels; ++p) {
      memcpy(d + p * os.c, pa + p * as.c, (size_t)as.c);
      memcpy(d + p * os.c + as.c, pb + p * bs.c, (size_t)bs.c);
    }
    q8_ready_[(size_t)op.out] = 1;
    return Status::Ok;
  }

  Status concat(Backend& backend, const GraphOp& op) {
    const Tensor& a = tensors_[op.in[0]];
    const Tensor& b = tensors_[op.in[1]];
    const Shape s(a.shape().h, a.shape().w, a.shape().c + b.shape().c);
    return withOut(op.out, s, [&](Tensor& out) { return backend.concat(a, b, out); });
  }

  Status maxPool(Backend& backend, const GraphOp& op) {
    const Tensor& in = tensors_[op.in[0]];
    const Shape os(poolOutputSize(in.shape().h, op.k, op.s, op.p), poolOutputSize(in.shape().w, op.k, op.s, op.p),
                   in.shape().c);
    return withOut(op.out, os, [&](Tensor& out) { return backend.maxPool(in, op.k, op.s, op.p, out); });
  }

  /// Multi-head attention over the positions of the input (see Ultralytics' Attention):
  ///   qkv = conv1x1(x), per head the channels are [q (key_dim), k (key_dim), v (head_dim)];
  ///   a = softmax((q * scale)^T k) over positions; y = v a^T;
  ///   y += depthwise3x3(v); out = conv1x1(y).
  Status attention(Backend& backend, const GraphOp& op) {
    const Tensor& x = tensors_[op.in[0]];
    const Shape xs = x.shape();
    const int N = xs.h * xs.w;
    const int heads = op.heads, kd = op.key_dim, hd = op.head_dim;
    const int C = heads * hd;
    const int per_head = 2 * kd + hd;

    // 1. qkv
    Tensor qkv;
    TINYYOLO_CHECK(qkv.allocate(Shape(xs.h, xs.w, heads * per_head)));
    if (op.q2 >= 0 && qweights_ != nullptr && quant_ != nullptr) {
      TINYYOLO_CHECK(int8Pointwise(backend, x, op.q2, weight(op.b), qkv));
    } else {
      TINYYOLO_CHECK(backend.conv2d(x, weight(op.w), weight(op.b), ConvParams{1, 1, 0, heads * per_head, Activation::Linear}, qkv));
    }

    // 2. attention per head, and v as an image (for the positional term)
    Tensor y;
    TINYYOLO_CHECK(y.allocate(Shape(xs.h, xs.w, C)));
    Tensor v_img;
    TINYYOLO_CHECK(v_img.allocate(Shape(xs.h, xs.w, C)));
    // The queries, keys and values are copied into contiguous per-head arrays (position-major),
    // so the loops below run over memory in order and need no index arithmetic per element.
    const float scale = 1.0f / sqrtf((float)kd);
    const size_t rows = (size_t)heads * N;  // one row per (head, position)
    Buffer<float> qs, ks, vs, scores;
    TINYYOLO_CHECK(qs.allocate(rows * kd));
    TINYYOLO_CHECK(ks.allocate(rows * kd));
    TINYYOLO_CHECK(vs.allocate(rows * hd));
    // The work items are (head, block of 4 queries); each worker has its own scores buffer.
    const int blocks_per_head = (N + 3) / 4;
    const int items = heads * blocks_per_head;
    const int workers = attentionWorkers(backend.workers(), items);
    TINYYOLO_CHECK(scores.allocate((size_t)workers * 4 * N));
    float* qp = qs.data();
    float* kp = ks.data();
    float* vp = vs.data();
    for (int n = 0; n < N; ++n) {
      const int yy = n / xs.w, xx = n % xs.w;
      for (int h = 0; h < heads; ++h) {
        const int base = h * per_head;
        const size_t row = (size_t)h * N + n;
        for (int d = 0; d < kd; ++d) {
          qp[row * kd + d] = qkv.at(yy, xx, base + d);
          kp[row * kd + d] = qkv.at(yy, xx, base + kd + d);
        }
        for (int d = 0; d < hd; ++d) vp[row * hd + d] = qkv.at(yy, xx, base + 2 * kd + d);
      }
    }
    AttentionJob job = {qp, kp, vp, scores.data(), y.data(), N, heads, kd, hd, C, scale, blocks_per_head};
    parallelRange(items, workers, &attentionBlocks, &job);

    for (int yy = 0; yy < xs.h; ++yy) {
      for (int xx = 0; xx < xs.w; ++xx) {
        for (int h = 0; h < heads; ++h) {
          for (int d = 0; d < hd; ++d) {
            v_img.at(yy, xx, h * hd + d) = qkv.at(yy, xx, h * per_head + 2 * kd + d);
          }
        }
      }
    }

    // 3. positional term: depthwise 3x3 on v, added to y
    Tensor pe;
    TINYYOLO_CHECK(pe.allocate(Shape(xs.h, xs.w, C)));
    groupedConv(v_img, weight(op.w2), weight(op.b2), C, 3, 1, 1, C, pe);
    for (size_t i = 0; i < y.size(); ++i) y.data()[i] += pe.data()[i];

    // 4. projection
    return withOut(op.out, xs, [&](Tensor& out) {
      if (op.q3 >= 0 && qweights_ != nullptr && quant_ != nullptr) {
        return int8Pointwise(backend, y, op.q3, weight(op.b3), out);
      }
      return backend.conv2d(y, weight(op.w3), weight(op.b3), ConvParams{1, 1, 0, xs.c, Activation::Linear}, out);
    });
  }

  const GraphOp* ops_ = nullptr;
  int op_count_ = 0;
  const GraphParam* params_ = nullptr;
  int param_count_ = 0;
  const float* weights_ = nullptr;
  int tensor_count_ = 0;
  int input_id_ = 0;
  Tensor* tensors_ = nullptr;
  Buffer<int> last_use_;   ///< index of the last operation that reads each tensor (-1: none)
  Buffer<uint8_t> kept_;   ///< 1 for tensors that must stay after run()
  const int8_t* qweights_ = nullptr;
  const float* qscales_ = nullptr;
  const GraphQuant* quant_ = nullptr;
  QTensor* q8_ = nullptr;          ///< int8 activations (see computeInt8Only)
  Buffer<uint8_t> int8_only_;      ///< 1 for tensors that stay int8 between operations (see computeInt8Only)
  Buffer<int> producer_;           ///< operation that produces each tensor (for int8 tensors), or -1
  Buffer<float> tscale_;           ///< scale of each tensor when it is int8
  Buffer<uint8_t> q8_ready_;       ///< 1 while q8_ holds the tensor of the current run
  bool profiling_ = false;
  uint32_t phase_us_[(int)Phase::Count] = {};
  Buffer<uint32_t> op_us_;  ///< microseconds of each operation in the last run()
};

}  // namespace tinyyolo
