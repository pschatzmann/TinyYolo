#pragma once
/// @file Model.h  A compiled network: a graph of layers with pre-allocated activations.

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/model/Layer.h"
#include "TinyYolo/platform/Clock.h"
#include "TinyYolo/model/LayerSpec.h"

namespace tinyyolo {

inline Layer* createLayer(const LayerSpec& spec, WeightPrecision precision) {
  switch (spec.type) {
    case LayerType::Conv: return new ConvLayer(spec, precision);
    case LayerType::MaxPool: return new MaxPoolLayer(spec);
    case LayerType::Upsample: return new UpsampleLayer(spec);
    case LayerType::Route: return new RouteLayer(spec);
    case LayerType::Yolo: return new YoloLayer(spec);
    case LayerType::Add: return new AddLayer(spec);
    case LayerType::Split: return new SplitLayer(spec);
  }
  return nullptr;
}

class Model {
 public:
  Model() {}
  ~Model() { destroy(); }
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  /// Validates the spec, creates the layers, infers all shapes and allocates memory.
  /// `precision` selects how convolution weights are stored (see WeightPrecision).
  /// `reuse_activations` = false gives every output its own memory (useful for testing).
  Status build(const NetworkSpec& spec, Backend& backend,
               WeightPrecision precision = WeightPrecision::Float32, bool reuse_activations = true) {
    destroy();
    TINYYOLO_CHECK(validateSpec(spec));
    spec_ = spec;
    backend_ = &backend;
    precision_ = precision;
    TINYYOLO_CHECK(input_.allocate(Shape(spec.input_h, spec.input_w, spec.input_c)));

    layers_ = new Layer*[spec.layer_count];
    if (layers_ == nullptr) {
      TINYYOLO_LOGE("Model", "out of memory for %d layer pointers", spec.layer_count);
      return Status::OutOfMemory;
    }
    for (int i = 0; i < spec.layer_count; ++i) layers_[i] = nullptr;
    TINYYOLO_CHECK(sources_.allocate((size_t)spec.layer_count * 2));
    TINYYOLO_CHECK(heads_.allocate((size_t)spec.layer_count));
    TINYYOLO_CHECK(feeds_head_.allocate((size_t)spec.layer_count));

    // Resolve every layer's inputs first, so we know which layers feed a YOLO head.
    for (int i = 0; i < spec.layer_count; ++i) {
      TINYYOLO_CHECK(resolve(i, spec.layers[i].from[0], sources_[2 * i]));
      TINYYOLO_CHECK(resolve(i, spec.layers[i].from[1], sources_[2 * i + 1]));
    }
    for (int i = 0; i < spec.layer_count; ++i) {
      if (spec.layers[i].type == LayerType::Yolo && sources_[2 * i] >= 0) {
        feeds_head_[(size_t)sources_[2 * i]] = 1;
      }
    }

    for (int i = 0; i < spec.layer_count; ++i) {
      const LayerSpec& ls = spec.layers[i];
      const int first = sources_[2 * i];
      const int second = sources_[2 * i + 1];
      const WeightPrecision layer_precision = effectivePrecision(ls, i);

      Layer* layer = createLayer(ls, layer_precision);
      if (layer == nullptr) {
        TINYYOLO_LOGE("Model", "layer %d: out of memory", i);
        return Status::OutOfMemory;
      }
      layers_[i] = layer;
      layer_count_ = i + 1;

      const Shape in = shapeOf(first);
      const Shape other = second == kNone ? Shape() : shapeOf(second);
      const Status status = layer->configure(in, other);
      if (status != Status::Ok) {
        TINYYOLO_LOGE("Model", "layer %d (%s) cannot be configured: %s", i, layer->name(),
                      statusName(status));
        return status;
      }
      if (ls.type == LayerType::Yolo) heads_[head_count_++] = i;
      if (ls.type == LayerType::Conv) {
        TINYYOLO_LOGD("Model", "layer %d (Conv): %s weights%s", i,
                      layer_precision == WeightPrecision::Int8 ? "int8" : "float32",
                      feeds_head_[(size_t)i] ? " (feeds a YOLO head)" : "");
      }
    }

    if (head_count_ == 0) {
      TINYYOLO_LOGW("Model", "network has no YOLO output layer; detections will be empty");
    }
    TINYYOLO_CHECK(planActivations(reuse_activations));
    built_ = true;
    return Status::Ok;
  }

  /// Reads every layer's parameters, in network order, from `source`.
  Status loadWeights(WeightSource& source) {
    if (!built_) {
      TINYYOLO_LOGE("Model", "loadWeights called before a successful build()");
      return Status::NotInitialized;
    }
    for (int i = 0; i < layer_count_; ++i) {
      const Status status = layers_[i]->loadWeights(source);
      if (status != Status::Ok) {
        TINYYOLO_LOGE("Model", "layer %d (%s): reading weights failed: %s", i, layers_[i]->name(),
                      statusName(status));
        return status;
      }
    }
    TINYYOLO_LOGI("Model", "loaded %lu parameters into %d layers", (unsigned long)paramCount(),
                  layer_count_);
    return Status::Ok;
  }

  /// Runs all layers. The input tensor must be filled beforehand.
  Status run() {
    if (!built_) {
      TINYYOLO_LOGE("Model", "run called before a successful build()");
      return Status::NotInitialized;
    }
    for (int i = 0; i < layer_count_; ++i) {
      const Tensor& first = outputOf(sources_[2 * i]);
      const int second_ref = sources_[2 * i + 1];
      const Tensor* second = second_ref == kNone ? nullptr : &outputOf(second_ref);
      const uint32_t start = layer_us_.empty() ? 0 : nowMicros();
      const Status status = layers_[i]->forward(*backend_, first, second);
      if (!layer_us_.empty()) layer_us_[(size_t)i] = nowMicros() - start;
      if (status != Status::Ok) {
        TINYYOLO_LOGE("Model", "layer %d (%s) failed during inference: %s", i, layers_[i]->name(),
                      statusName(status));
        return status;
      }
    }
    return Status::Ok;
  }

  /// Times every layer in run() (for finding the slow ones). Off by default.
  Status enableProfiling(bool on) {
    layer_us_.release();
    if (on) TINYYOLO_CHECK(layer_us_.allocate((size_t)layer_count_));
    return Status::Ok;
  }
  /// Time of layer `index` in the last run(), in microseconds (0 when profiling is off).
  uint32_t layerMicros(int index) const { return layer_us_.empty() ? 0 : layer_us_[(size_t)index]; }

  /// Chooses int8 activation scales for every convolution with int8 weights, from one
  /// representative input (for example a camera frame). Call after loadWeights(). The scales
  /// are the largest values seen in this run, so an input with much larger values will clip.
  /// Convolutions with float32 weights and the layers between them stay float.
  Status calibrate(const Tensor& sample) {
    if (!built_) {
      TINYYOLO_LOGE("Model", "calibrate called before a successful build()");
      return Status::NotInitialized;
    }
    if (sample.shape() != input_.shape()) {
      TINYYOLO_LOGE("Model", "calibrate: sample %dx%dx%d, network input %dx%dx%d", sample.shape().h,
                    sample.shape().w, sample.shape().c, input_.shape().h, input_.shape().w,
                    input_.shape().c);
      return Status::ShapeMismatch;
    }
    for (int i = 0; i < layer_count_; ++i) {
      if (ConvLayer* conv = layers_[i]->asMutableConv()) conv->disableInt8Activations();
    }
    quantized_layers_ = 0;

    // One float pass records the range of every layer output.
    for (size_t i = 0; i < input_.size(); ++i) input_.data()[i] = sample.data()[i];
    TINYYOLO_CHECK(run());
    TINYYOLO_CHECK(act_max_.allocate((size_t)layer_count_));
    const float input_max = maxAbsValue(input_);
    for (int i = 0; i < layer_count_; ++i) act_max_[i] = maxAbsValue(layers_[i]->output());

    // The int8 input and output of a convolution only live during its own forward pass, so
    // all int8 convolutions share one input and one output buffer, sized for the largest.
    size_t max_in = 0, max_out = 0;
    for (int i = 0; i < layer_count_; ++i) {
      ConvLayer* conv = layers_[i]->asMutableConv();
      if (conv == nullptr || conv->precision() != WeightPrecision::Int8) continue;
      const size_t in_elements = conv->int8InputShape().elements();
      const size_t out_elements = layers_[i]->outputShape().elements();
      if (in_elements > max_in) max_in = in_elements;
      if (out_elements > max_out) max_out = out_elements;
    }
    if (max_in > 0) {
      TINYYOLO_CHECK(int8_in_.allocate(max_in));
      TINYYOLO_CHECK(int8_out_.allocate(max_out));
    }
    for (int i = 0; i < layer_count_; ++i) {
      ConvLayer* conv = layers_[i]->asMutableConv();
      if (conv == nullptr || conv->precision() != WeightPrecision::Int8) continue;
      const int src = sources_[2 * i];
      const float in_max = src < 0 ? input_max : act_max_[src];
      TINYYOLO_CHECK(conv->enableInt8Activations(scaleFor(in_max), scaleFor(act_max_[i]),
                                                 int8_in_.data(), int8_out_.data()));
      ++quantized_layers_;
    }
    TINYYOLO_LOGI("Model", "calibrated: %d convolutions use int8 activations", quantized_layers_);
    return Status::Ok;
  }

  /// Number of convolutions that run with int8 activations (0 before calibrate()).
  int int8ActivationLayers() const { return quantized_layers_; }

  /// Bytes used for all parameters after build().
  size_t weightBytes() const {
    size_t total = 0;
    for (int i = 0; i < layer_count_; ++i) total += layers_[i]->weightBytes();
    return total;
  }

  WeightPrecision precision() const { return precision_; }

  /// Bytes used for layer outputs after build(), with memory shared between layers.
  size_t activationBytes() const { return activation_elements_ * sizeof(float); }
  /// Bytes the layer outputs would need without sharing memory.
  size_t activationBytesWithoutReuse() const { return naive_elements_ * sizeof(float); }

  size_t paramCount() const {
    size_t total = 0;
    for (int i = 0; i < layer_count_; ++i) total += layers_[i]->paramCount();
    return total;
  }

  bool isBuilt() const { return built_; }
  Tensor& input() { return input_; }
  const Shape& inputShape() const { return input_.shape(); }
  const NetworkSpec& spec() const { return spec_; }
  int layerCount() const { return layer_count_; }
  const Layer& layer(int index) const { return *layers_[index]; }
  int headCount() const { return head_count_; }
  const Layer& head(int index) const { return *layers_[heads_[(size_t)index]]; }

 private:
  /// Places every layer output in one shared arena. An output lives from the layer that
  /// writes it to the last layer that reads it; YOLO head outputs are read after the run and
  /// live to the end. Outputs whose lifetimes do not overlap may share memory.
  Status planActivations(bool reuse) {
    const int n = layer_count_;
    Buffer<int> last_use;
    Buffer<size_t> offsets;
    Buffer<size_t> sizes;
    TINYYOLO_CHECK(last_use.allocate((size_t)n));
    TINYYOLO_CHECK(offsets.allocate((size_t)n));
    TINYYOLO_CHECK(sizes.allocate((size_t)n));
    for (int i = 0; i < n; ++i) {
      last_use[i] = i;
      sizes[i] = layers_[i]->outputShape().elements();
      naive_elements_ += sizes[i];
    }
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < 2; ++k) {
        const int src = sources_[2 * j + k];
        if (src >= 0 && last_use[src] < j) last_use[src] = j;
      }
    }
    for (int i = 0; i < n; ++i) {
      if (spec_.layers[i].type == LayerType::Yolo) last_use[i] = n;
    }

    size_t arena_elements = 0;
    for (int i = 0; i < n; ++i) {
      // Candidate offsets: 0, and the end of every earlier output still alive at layer i.
      size_t best = (size_t)-1;
      for (int c = -1; c < i; ++c) {
        // Without reuse, the candidate is the end of everything placed so far.
        size_t candidate = c < 0 ? 0 : offsets[c] + sizes[c];
        if (!reuse) {
          candidate = 0;
          for (int j = 0; j < i; ++j) {
            if (offsets[j] + sizes[j] > candidate) candidate = offsets[j] + sizes[j];
          }
        }
        if (c >= 0 && last_use[c] < i) continue;
        if (candidate >= best) continue;
        bool free_here = reuse || c < 0;  // without reuse, every output gets fresh memory
        for (int j = 0; j < i && free_here; ++j) {
          if (last_use[j] < i) continue;  // j is dead when i is written
          const bool overlap = candidate < offsets[j] + sizes[j] && offsets[j] < candidate + sizes[i];
          free_here = !overlap;
        }
        if (free_here) best = candidate;
      }
      offsets[i] = best;
      if (best + sizes[i] > arena_elements) arena_elements = best + sizes[i];
    }

    TINYYOLO_CHECK(arena_.allocate(arena_elements));
    activation_elements_ = arena_elements;
    for (int i = 0; i < n; ++i) layers_[i]->attachOutput(arena_.data() + offsets[i]);
    TINYYOLO_LOGD("Model", "activations: %lu bytes planned, %lu bytes without reuse",
                  (unsigned long)(activation_elements_ * sizeof(float)),
                  (unsigned long)(naive_elements_ * sizeof(float)));
    return Status::Ok;
  }

  /// Resolves a layer reference to an absolute index (-1 = network input, kNone = unused).
  Status resolve(int index, int reference, int& resolved) const {
    if (reference == kNone) {
      resolved = kNone;
      return Status::Ok;
    }
    resolved = reference == kPrevious ? index - 1 : reference;
    if (resolved < -1 || resolved >= index) {
      TINYYOLO_LOGE("Model", "layer %d reads layer %d, which is not an earlier layer", index,
                    reference);
      return Status::InvalidModel;
    }
    return Status::Ok;
  }

  /// Output shape of a resolved reference. Valid during build(), before outputs are planned.
  Shape shapeOf(int ref) const { return ref < 0 ? input_.shape() : layers_[ref]->outputShape(); }

  const Tensor& outputOf(int ref) const { return ref < 0 ? input_ : layers_[ref]->output(); }

  static Status validateSpec(const NetworkSpec& spec) {
    if (spec.layers == nullptr || spec.layer_count <= 0) {
      TINYYOLO_LOGE("Model", "network has no layers");
      return Status::InvalidModel;
    }
    if (spec.input_h <= 0 || spec.input_w <= 0) {
      TINYYOLO_LOGE("Model", "invalid input size %dx%d", spec.input_w, spec.input_h);
      return Status::InvalidModel;
    }
    if (spec.input_c != 1 && spec.input_c != 3) {
      TINYYOLO_LOGE("Model", "unsupported input channel count %d (use 1 or 3)", spec.input_c);
      return Status::InvalidModel;
    }
    return Status::Ok;
  }

  /// The precision of one layer. Order: the layer's own override, then (for int8 models) a
  /// float32 rule for layers that feed a YOLO head, then the model's precision.
  WeightPrecision effectivePrecision(const LayerSpec& ls, int index) const {
    if (ls.precision_override != kInheritPrecision) {
      return static_cast<WeightPrecision>(ls.precision_override);
    }
    if (precision_ == WeightPrecision::Int8 && feeds_head_[(size_t)index]) {
      return WeightPrecision::Float32;
    }
    return precision_;
  }

  /// Scale that maps values up to `max_abs` onto the int8 range.
  static float scaleFor(float max_abs) { return (max_abs > 0.0f ? max_abs : 1.0f) / 127.0f; }

  void destroy() {
    act_max_.release();
    layer_us_.release();
    int8_in_.release();
    int8_out_.release();
    quantized_layers_ = 0;
    for (int i = 0; i < layer_count_; ++i) delete layers_[i];
    delete[] layers_;
    layers_ = nullptr;
    layer_count_ = 0;
    head_count_ = 0;
    sources_.release();
    heads_.release();
    feeds_head_.release();
    arena_.release();
    activation_elements_ = 0;
    naive_elements_ = 0;
    input_ = Tensor();
    built_ = false;
  }

  NetworkSpec spec_{0, 0, 0, nullptr, 0, nullptr, 0};
  Backend* backend_ = nullptr;
  WeightPrecision precision_ = WeightPrecision::Float32;
  Layer** layers_ = nullptr;
  int layer_count_ = 0;
  Buffer<int> sources_;  ///< two resolved inputs per layer
  Buffer<int> heads_;    ///< indices of YOLO layers
  Buffer<uint8_t> feeds_head_;  ///< 1 for layers whose output feeds a YOLO head
  Buffer<float> arena_;         ///< storage shared by all layer outputs
  size_t activation_elements_ = 0;
  size_t naive_elements_ = 0;
  int head_count_ = 0;
  Tensor input_;
  bool built_ = false;
  Buffer<float> act_max_;  ///< largest output value per layer, from calibrate()
  int quantized_layers_ = 0;
  Buffer<uint32_t> layer_us_;  ///< per-layer time of the last run (profiling only)
  Buffer<int8_t> int8_in_;   ///< shared int8 input of the int8 convolutions
  Buffer<int8_t> int8_out_;  ///< shared int8 output of the int8 convolutions
};

}  // namespace tinyyolo
