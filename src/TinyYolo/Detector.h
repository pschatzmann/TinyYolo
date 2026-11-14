#pragma once
/// @file Detector.h  High-level facade: image in, detections out.

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Quantize.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/WeightSource.h"
#include "TinyYolo/image/ImageView.h"
#include "TinyYolo/image/Letterbox.h"
#include "TinyYolo/image/Preprocessor.h"
#include "TinyYolo/model/Model.h"
#include "TinyYolo/platform/Platform.h"
#include "TinyYolo/postprocess/Detection.h"
#include "TinyYolo/postprocess/Postprocessor.h"

namespace tinyyolo {

struct DetectorConfig {
  float score_threshold;  ///< minimum objectness * class score, in (0, 1]
  float iou_threshold;    ///< NMS overlap threshold, in [0, 1]
  int max_candidates;     ///< pre-NMS box capacity
  WeightPrecision weight_precision;  ///< Float32 (default) or Int8 weights, see WeightPrecision

  DetectorConfig()
      : score_threshold(0.25f),
        iou_threshold(0.45f),
        max_candidates(1024),
        weight_precision(WeightPrecision::Float32) {}
};

class Detector {
 public:
  Detector() {}
  Detector(const Detector&) = delete;
  Detector& operator=(const Detector&) = delete;

  /// Builds the network, loads the weights and allocates buffers. May be called again.
  Status begin(const NetworkSpec& spec, WeightSource& weights,
               const DetectorConfig& config = DetectorConfig(), Backend* backend = nullptr) {
    ready_ = false;
    config_ = config;
    if (config_.score_threshold <= 0.0f || config_.score_threshold > 1.0f) {
      TINYYOLO_LOGW("Detector", "score_threshold %.3f outside (0, 1]; clamped", config_.score_threshold);
      config_.score_threshold = clampf(config_.score_threshold, 0.001f, 1.0f);
    }
    if (config_.iou_threshold < 0.0f || config_.iou_threshold > 1.0f) {
      TINYYOLO_LOGW("Detector", "iou_threshold %.3f outside [0, 1]; clamped", config_.iou_threshold);
      config_.iou_threshold = clampf(config_.iou_threshold, 0.0f, 1.0f);
    }
    if (config_.max_candidates <= 0) {
      TINYYOLO_LOGE("Detector", "max_candidates must be positive, got %d", config_.max_candidates);
      return Status::InvalidArgument;
    }
    backend_ = backend != nullptr ? backend : &default_backend_;

    TINYYOLO_CHECK(model_.build(spec, *backend_, config_.weight_precision));
    if (spec.input_w % 32 != 0 || spec.input_h % 32 != 0) {
      TINYYOLO_LOGW("Detector", "input %dx%d is not a multiple of 32; grid cells may misalign",
                    spec.input_w, spec.input_h);
    }
    TINYYOLO_CHECK(model_.loadWeights(weights));
    if (model_.headCount() > 0 && spec.label_count > 0 &&
        spec.label_count != model_.head(0).spec().classes) {
      TINYYOLO_LOGW("Detector", "%d labels given but the head predicts %d classes",
                    spec.label_count, model_.head(0).spec().classes);
    }
    TINYYOLO_CHECK(postprocessor_.reserve(config_.max_candidates));
    ready_ = true;
    TINYYOLO_LOGI("Detector", "ready: platform=%s backend=%s input=%dx%d layers=%d heads=%d",
                  platformName(), backend_->name(), spec.input_w, spec.input_h,
                  model_.layerCount(), model_.headCount());
    return Status::Ok;
  }

  /// Runs detection on `image`. `out` must have capacity (see Detections::reserve).
  Status detect(const ImageView& image, Detections& out) {
    if (!ready_) {
      TINYYOLO_LOGE("Detector", "detect called before a successful begin()");
      return Status::NotInitialized;
    }
    if (out.capacity() == 0) {
      TINYYOLO_LOGE("Detector", "output container has no capacity; call reserve() first");
      return Status::InvalidArgument;
    }
    LetterboxInfo letterbox;
    TINYYOLO_CHECK(preprocessor_.letterbox(image, model_.input(), letterbox));
    TINYYOLO_CHECK(model_.run());
    return postprocessor_.process(model_, letterbox, config_.score_threshold,
                                  config_.iou_threshold, out);
  }

  /// Times every layer of the next detections (see Model::enableProfiling).
  Status enableProfiling(bool on) { return model_.enableProfiling(on); }
  /// Time of layer `index` in the last detection, in microseconds.
  uint32_t layerMicros(int index) const { return model_.layerMicros(index); }

  /// Chooses int8 activation scales from a representative image (see Model::calibrate).
  /// Only needed with DetectorConfig::weight_precision = Int8. Call after begin() and load,
  /// before the first detect(); use an image like the ones that will be detected.
  Status calibrate(const ImageView& image) {
    if (!ready_) {
      TINYYOLO_LOGE("Detector", "calibrate called before a successful begin()");
      return Status::NotInitialized;
    }
    LetterboxInfo letterbox;
    TINYYOLO_CHECK(preprocessor_.letterbox(image, model_.input(), letterbox));
    return model_.calibrate(model_.input());
  }

  /// Class name for `class_id`, or "?" (with a warning) when no label exists.
  const char* label(int class_id) const {
    const NetworkSpec& spec = model_.spec();
    if (spec.labels == nullptr || class_id < 0 || class_id >= spec.label_count) {
      TINYYOLO_LOGW("Detector", "no label for class %d", class_id);
      return "?";
    }
    return spec.labels[class_id];
  }

  /// The built-in backend. Tune it (e.g. setWorkers()) before begin().
  DefaultBackend& defaultBackend() { return default_backend_; }

  bool isReady() const { return ready_; }
  const Model& model() const { return model_; }
  const DetectorConfig& config() const { return config_; }

 private:
  Model model_;
  Preprocessor preprocessor_;
  Postprocessor postprocessor_;
  DefaultBackend default_backend_;
  Backend* backend_ = nullptr;
  DetectorConfig config_;
  bool ready_ = false;
};

}  // namespace tinyyolo
