#pragma once
/// @file Layer.h  Layer classes. Each layer owns its parameters and its output tensor.

#include "TinyYolo/backend/Backend.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/core/WeightSource.h"
#include "TinyYolo/core/Quantize.h"
#include "TinyYolo/core/Types.h"
#include "TinyYolo/model/LayerSpec.h"

namespace tinyyolo {

class ConvLayer;

class Layer {
 public:
  explicit Layer(const LayerSpec& spec) : spec_(spec) {}
  virtual ~Layer() {}
  Layer(const Layer&) = delete;
  Layer& operator=(const Layer&) = delete;

  virtual const char* name() const = 0;
  const LayerSpec& spec() const { return spec_; }
  const Tensor& output() const { return output_; }
  /// Output shape, known after configure(). Storage is provided by the model (see attachOutput).
  const Shape& outputShape() const { return out_shape_; }
  /// Makes the output a view of `storage` (outputShape().elements() floats, owned by the model).
  void attachOutput(float* storage) { output_.attachView(storage, out_shape_); }

  /// Infers the output shape from the input shape(s) and allocates storage.
  /// `second` is an invalid Shape when the layer has a single input.
  virtual Status configure(const Shape& input, const Shape& second) = 0;
  /// Number of trainable values (weights + biases).
  virtual size_t paramCount() const { return 0; }
  /// Bytes used for parameters (weights, scales and biases) after configure().
  virtual size_t weightBytes() const { return 0; }
  virtual Status loadWeights(WeightSource&) { return Status::Ok; }
  virtual Status forward(Backend& backend, const Tensor& input, const Tensor* second) = 0;
  /// The layer as a convolution, or nullptr for other layer types (no RTTI needed).
  virtual const ConvLayer* asConv() const { return nullptr; }
  /// Non-const access for calibration (nullptr for other layer types).
  virtual ConvLayer* asMutableConv() { return nullptr; }

 protected:
  LayerSpec spec_;
  Shape out_shape_;
  Tensor output_;
};

class ConvLayer : public Layer {
 public:
  ConvLayer(const LayerSpec& spec, WeightPrecision precision) : Layer(spec), precision_(precision) {}
  const char* name() const override { return "Conv"; }
  const ConvLayer* asConv() const override { return this; }
  ConvLayer* asMutableConv() override { return this; }

  Status configure(const Shape& input, const Shape&) override {
    const int pad = spec_.pad < 0 ? spec_.kernel / 2 : spec_.pad;
    if (spec_.filters <= 0 || spec_.kernel <= 0 || spec_.stride <= 0 || pad < 0 || !input.isValid()) {
      TINYYOLO_LOGE("Conv", "invalid parameters: filters=%d kernel=%d stride=%d pad=%d",
                    spec_.filters, spec_.kernel, spec_.stride, pad);
      return Status::InvalidModel;
    }
    const int oh = windowOutputSize(input.h, spec_.kernel, spec_.stride, pad);
    const int ow = windowOutputSize(input.w, spec_.kernel, spec_.stride, pad);
    if (oh <= 0 || ow <= 0) {
      TINYYOLO_LOGE("Conv", "input %dx%d is too small for kernel %d with padding %d", input.h,
                    input.w, spec_.kernel, pad);
      return Status::ShapeMismatch;
    }
    pad_ = pad;
    in_shape_ = input;
    out_channels_ = (size_t)spec_.filters;
    weight_count_ = out_channels_ * (size_t)input.c * spec_.kernel * spec_.kernel;
    out_shape_ = Shape(oh, ow, spec_.filters);
    // Parameters are allocated by loadWeights(), so a view into read-only memory costs no RAM.
    return Status::Ok;
  }

  size_t paramCount() const override { return weight_count_ + out_channels_; }

  size_t weightBytes() const override {
    const size_t weights = precision_ == WeightPrecision::Int8
                               ? weight_count_ * sizeof(int8_t) + out_channels_ * sizeof(float)
                               : weight_count_ * sizeof(float);
    return weights + out_channels_ * sizeof(float);  // bias is always float32
  }

  /// Takes the parameters from `source`: a view when it can give one, else a copy. With int8
  /// precision, float32 weights are quantized while they are copied.
  Status loadWeights(WeightSource& source) override {
    if (precision_ == WeightPrecision::Int8) {
      const int8_t* w8 = nullptr;
      const float* sc = nullptr;
      const float* b = nullptr;
      if (source.viewConvInt8(out_channels_, weight_count_, w8, sc, b)) {
        w8_ = w8;
        scales_ = sc;
        bias_ = b;
        loaded_ = true;
        return Status::Ok;
      }
      // Int8 weights are large and read in full by every output band: PSRAM first, so that
      // internal RAM stays free for the kernels' scratch buffers.
      TINYYOLO_CHECK(own_w8_.allocate(weight_count_, MemoryPlacement::PsramFirst));
      TINYYOLO_CHECK(own_scales_.allocate(out_channels_));
      TINYYOLO_CHECK(own_bias_.allocate(out_channels_));
      if (source.readConvInt8(out_channels_, weight_count_, own_w8_.data(), own_scales_.data(),
                              own_bias_.data())) {
        w8_ = own_w8_.data();
        scales_ = own_scales_.data();
        bias_ = own_bias_.data();
        loaded_ = true;
        return Status::Ok;
      }
      Buffer<float> float_weights;
      TINYYOLO_CHECK(float_weights.allocate(weight_count_));
      ConvWeights target = {float_weights.data(), weight_count_, own_bias_.data(), out_channels_,
                            spec_.batch_norm};
      TINYYOLO_CHECK(source.readConv(target));
      // The source gives OIHW. Int8 weights are stored OHWI, so each output channel is
      // gathered into that order and quantized on its own (one scale per output channel).
      const size_t plen = weight_count_ / out_channels_;
      const size_t K = (size_t)spec_.kernel;
      const size_t IC = plen / (K * K);
      Buffer<float> row;
      TINYYOLO_CHECK(row.allocate(plen));
      for (size_t o = 0; o < out_channels_; ++o) {
        const float* src = float_weights.data() + o * plen;  // [ic][ky][kx]
        for (size_t ic = 0; ic < IC; ++ic) {
          for (size_t ky = 0; ky < K; ++ky) {
            for (size_t kx = 0; kx < K; ++kx) {
              row[(ky * K + kx) * IC + ic] = src[(ic * K + ky) * K + kx];  // [ky][kx][ic]
            }
          }
        }
        quantizeRows(row.data(), 1, plen, own_w8_.data() + o * plen, own_scales_.data() + o);
      }
      w8_ = own_w8_.data();
      scales_ = own_scales_.data();
      bias_ = own_bias_.data();
    } else {
      TINYYOLO_CHECK(own_wf_.allocate(weight_count_));
      TINYYOLO_CHECK(own_bias_.allocate(out_channels_));
      ConvWeights target = {own_wf_.data(), weight_count_, own_bias_.data(), out_channels_,
                            spec_.batch_norm};
      TINYYOLO_CHECK(source.readConv(target));
      wf_ = own_wf_.data();
      bias_ = own_bias_.data();
    }
    loaded_ = true;
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor*) override {
    if (!loaded_) {
      TINYYOLO_LOGE("Conv", "forward called before the weights were loaded");
      return Status::NotInitialized;
    }
    const ConvParams params = {spec_.kernel, spec_.stride, pad_, spec_.filters, spec_.activation};
    if (quantized_) {
      // Int8 activations: quantize the input, convolve on integers, return float output.
      // The int8 buffers belong to the model and are shared by all such layers.
      QTensor qin, qout;
      qin.attachView(in_storage_, int8InputShape(), in_scale_);
      qout.attachView(out_storage_, out_shape_, out_scale_);
      TINYYOLO_CHECK(quantizeWithPadding(input, in_scale_, qin));
      TINYYOLO_CHECK(backend.conv2dQ(qin, w8_q_, scales_, bias_, params, qout));
      return dequantizeTensor(qout, output_);
    }
    if (precision_ == WeightPrecision::Int8) {
      return backend.conv2dInt8(input, w8_, scales_, bias_, params, output_);
    }
    return backend.conv2d(input, wf_, bias_, params, output_);
  }

  /// Enables int8 activations (int8 weights only). The input is quantized with in_scale and
  /// the output with out_scale; the output is handed on as float. `in_storage` must hold
  /// inputShape().elements() values and `out_storage` outputShape().elements() values; the
  /// model shares them between layers. Call after loadWeights().
  Status enableInt8Activations(float in_scale, float out_scale, int8_t* in_storage,
                               int8_t* out_storage) {
    if (precision_ != WeightPrecision::Int8 || !loaded_ || !(in_scale > 0.0f) ||
        !(out_scale > 0.0f) || in_storage == nullptr || out_storage == nullptr) {
      TINYYOLO_LOGE("Conv", "int8 activations need int8 weights, loaded, scales and storage");
      return Status::InvalidArgument;
    }
    // ESP-NN needs input channels in multiples of 4. Inputs with other channel counts (the
    // 3-channel first layer) get zero channels in the int8 path, and their weights get zero
    // columns to match. The float-activation paths keep the unpadded weights.
    const int channels = in_shape_.c;
    const int padded = paddedChannels();
    w8_q_ = w8_;
    if (padded != channels) {
      const size_t kk = (size_t)spec_.kernel * spec_.kernel;
      TINYYOLO_CHECK(own_w8_padded_.allocate(out_channels_ * kk * (size_t)padded,  // zeroed
                                             MemoryPlacement::PsramFirst));
      for (size_t o = 0; o < out_channels_; ++o) {
        for (size_t p = 0; p < kk; ++p) {
          const int8_t* src = w8_ + (o * kk + p) * (size_t)channels;  // OHWI: [o][p][c]
          int8_t* dst = own_w8_padded_.data() + (o * kk + p) * (size_t)padded;
          for (int c = 0; c < channels; ++c) dst[c] = src[c];
        }
      }
      w8_q_ = own_w8_padded_.data();
    }
    in_scale_ = in_scale;
    out_scale_ = out_scale;
    in_storage_ = in_storage;
    out_storage_ = out_storage;
    quantized_ = true;
    return Status::Ok;
  }
  void disableInt8Activations() { quantized_ = false; }
  bool int8Activations() const { return quantized_; }
  Shape inputShape() const { return in_shape_; }
  /// The shape of the int8 input: the input with its channels padded to a multiple of 4.
  Shape int8InputShape() const { return Shape(in_shape_.h, in_shape_.w, paddedChannels()); }

  // Parameter access, for exporting the model (valid after loadWeights()).
  WeightPrecision precision() const { return precision_; }
  bool batchNorm() const { return spec_.batch_norm; }
  size_t outChannels() const { return out_channels_; }
  size_t weightCount() const { return weight_count_; }
  const int8_t* int8Weights() const { return w8_; }
  const float* scales() const { return scales_; }
  const float* floatWeights() const { return wf_; }
  const float* bias() const { return bias_; }

 private:
  WeightPrecision precision_;
  bool loaded_ = false;
  // What the layer reads from: either its own copies (owned) or a view (read-only memory).
  const float* wf_ = nullptr;
  const int8_t* w8_ = nullptr;
  const float* scales_ = nullptr;
  const float* bias_ = nullptr;
  Buffer<float> own_wf_;
  Buffer<int8_t> own_w8_;
  Buffer<int8_t> own_w8_padded_;     ///< int8 weights with zero input channels (int8 path only)
  const int8_t* w8_q_ = nullptr;     ///< weights for the int8 activation path (padded or w8_)

  /// Input channels rounded up to a multiple of 4 (ESP-NN requirement).
  int paddedChannels() const { return (in_shape_.c + 3) / 4 * 4; }

  /// Quantizes `in` into `out`, whose channels may be more than the input's: the extra
  /// channels are zero. Heights and widths must match.
  static Status quantizeWithPadding(const Tensor& in, float scale, QTensor& out) {
    const Shape is = in.shape();
    const Shape os = out.shape();
    if (os.h != is.h || os.w != is.w || os.c < is.c || !(scale > 0.0f)) {
      TINYYOLO_LOGE("Conv", "quantize: shape or scale mismatch");
      return Status::ShapeMismatch;
    }
    out.setScale(scale);
    const float inv = 1.0f / scale;
    for (int y = 0; y < is.h; ++y) {
      for (int x = 0; x < is.w; ++x) {
        const float* src = in.data() + ((size_t)y * is.w + x) * is.c;
        int8_t* dst = out.data() + ((size_t)y * os.w + x) * os.c;
        for (int c = 0; c < is.c; ++c) dst[c] = quantizeValue(src[c], inv);
        for (int c = is.c; c < os.c; ++c) dst[c] = 0;
      }
    }
    return Status::Ok;
  }
  Buffer<float> own_scales_;
  Buffer<float> own_bias_;
  int pad_ = 0;
  size_t out_channels_ = 0;
  size_t weight_count_ = 0;
  Shape in_shape_;
  bool quantized_ = false;  ///< int8 activations enabled (see enableInt8Activations)
  float in_scale_ = 1.0f;
  float out_scale_ = 1.0f;
  int8_t* in_storage_ = nullptr;   ///< shared with the other int8 layers (owned by the model)
  int8_t* out_storage_ = nullptr;
};

class MaxPoolLayer : public Layer {
 public:
  explicit MaxPoolLayer(const LayerSpec& spec) : Layer(spec) {}
  const char* name() const override { return "MaxPool"; }

  Status configure(const Shape& input, const Shape&) override {
    pad_total_ = spec_.pad < 0 ? 0 : spec_.pad;
    const int oh = poolOutputSize(input.h, spec_.kernel, spec_.stride, pad_total_);
    const int ow = poolOutputSize(input.w, spec_.kernel, spec_.stride, pad_total_);
    if (oh <= 0 || ow <= 0) {
      TINYYOLO_LOGE("MaxPool", "invalid kernel=%d stride=%d for input %dx%d", spec_.kernel,
                    spec_.stride, input.h, input.w);
      return Status::InvalidModel;
    }
    out_shape_ = Shape(oh, ow, input.c);
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor*) override {
    return backend.maxPool(input, spec_.kernel, spec_.stride, pad_total_, output_);
  }

 private:
  int pad_total_ = 0;
};

class UpsampleLayer : public Layer {
 public:
  explicit UpsampleLayer(const LayerSpec& spec) : Layer(spec) {}
  const char* name() const override { return "Upsample"; }

  Status configure(const Shape& input, const Shape&) override {
    if (spec_.stride <= 0) {
      TINYYOLO_LOGE("Upsample", "invalid factor %d", spec_.stride);
      return Status::InvalidModel;
    }
    out_shape_ = Shape(input.h * spec_.stride, input.w * spec_.stride, input.c);
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor*) override {
    return backend.upsample(input, spec_.stride, output_);
  }
};

class RouteLayer : public Layer {
 public:
  explicit RouteLayer(const LayerSpec& spec) : Layer(spec) {}
  const char* name() const override { return "Route"; }

  Status configure(const Shape& input, const Shape& second) override {
    if (!second.isValid()) {
      out_shape_ = input;
      return Status::Ok;
    }
    if (second.h != input.h || second.w != input.w) {
      TINYYOLO_LOGE("Route", "cannot concatenate %dx%d with %dx%d", input.h, input.w, second.h,
                    second.w);
      return Status::ShapeMismatch;
    }
    out_shape_ = Shape(input.h, input.w, input.c + second.c);
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor* second) override {
    if (second != nullptr) return backend.concat(input, *second, output_);
    return backend.copy(input, output_);
  }
};

/// Marks a detection head. Its output is the raw grid of predictions:
/// per anchor, (tx, ty, tw, th, objectness, class logits...).
class YoloLayer : public Layer {
 public:
  explicit YoloLayer(const LayerSpec& spec) : Layer(spec) {}
  const char* name() const override { return "Yolo"; }

  Status configure(const Shape& input, const Shape&) override {
    if (spec_.classes <= 0 || spec_.num_anchors <= 0 || spec_.anchors == nullptr) {
      TINYYOLO_LOGE("Yolo", "invalid head: classes=%d anchors=%d", spec_.classes, spec_.num_anchors);
      return Status::InvalidModel;
    }
    const int expected = spec_.num_anchors * (5 + spec_.classes);
    if (input.c != expected) {
      TINYYOLO_LOGE("Yolo", "head expects %d channels (%d anchors x %d) but receives %d", expected,
                    spec_.num_anchors, 5 + spec_.classes, input.c);
      return Status::InvalidModel;
    }
    out_shape_ = input;
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor*) override {
    return backend.copy(input, output_);
  }
};

/// One channel group of the input (grouped Darknet route).
class SplitLayer : public Layer {
 public:
  explicit SplitLayer(const LayerSpec& spec) : Layer(spec) {}
  const char* name() const override { return "Split"; }

  Status configure(const Shape& input, const Shape&) override {
    const int groups = spec_.kernel;
    const int group_id = spec_.stride;
    if (groups <= 0 || group_id < 0 || group_id >= groups || input.c % groups != 0) {
      TINYYOLO_LOGE("Split", "cannot take group %d of %d from %d channels", group_id, groups, input.c);
      return Status::InvalidModel;
    }
    count_ = input.c / groups;
    first_ = group_id * count_;
    out_shape_ = Shape(input.h, input.w, count_);
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor*) override {
    return backend.slice(input, first_, count_, output_);
  }

 private:
  int first_ = 0;
  int count_ = 0;
};

/// Element-wise sum of the primary input and the second input, followed by the activation.
class AddLayer : public Layer {
 public:
  explicit AddLayer(const LayerSpec& spec) : Layer(spec) {}
  const char* name() const override { return "Add"; }

  Status configure(const Shape& input, const Shape& second) override {
    if (!second.isValid() || second != input) {
      TINYYOLO_LOGE("Add", "shortcut inputs differ: %dx%dx%d and %dx%dx%d", input.h, input.w,
                    input.c, second.h, second.w, second.c);
      return Status::ShapeMismatch;
    }
    out_shape_ = input;
    return Status::Ok;
  }

  Status forward(Backend& backend, const Tensor& input, const Tensor* second) override {
    if (second == nullptr) {
      TINYYOLO_LOGE("Add", "forward called without a second input");
      return Status::InvalidArgument;
    }
    return backend.add(input, *second, spec_.activation, output_);
  }
};

}  // namespace tinyyolo
