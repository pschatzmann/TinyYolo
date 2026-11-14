#pragma once
/// @file TinyYoloModel.h  A Darknet model ready for detection: description, weights,
/// detector and result list in one object.

#include "TinyYolo/Detector.h"
#include "TinyYolo/app/ModelSource.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/model/DarknetWeights.h"
#include "TinyYolo/model/NetworkDescription.h"

namespace tinyyolo {

class TinyYoloModel {
 public:
  TinyYoloModel() {}
  TinyYoloModel(const TinyYoloModel&) = delete;
  TinyYoloModel& operator=(const TinyYoloModel&) = delete;

  /// Loads the model from `source` and prepares detection. `max_detections` is the capacity
  /// of the result list. Returns Ok when the model is ready; any failure is logged.
  Status begin(ModelSource& source, const DetectorConfig& config = DetectorConfig(),
               int max_detections = 20) {
    ready_ = false;
    if (!source.open()) {
      TINYYOLO_LOGE("TinyYoloModel", "model files are not available");
      return Status::InvalidArgument;
    }
    const Status status = load(source, config, max_detections);
    source.close();
    if (status != Status::Ok) {
      TINYYOLO_LOGE("TinyYoloModel", "setup failed: %s", statusName(status));
      return status;
    }
    ready_ = true;
    return Status::Ok;
  }

  /// Runs detection on one frame. The results are read with detections().
  Status detect(const ImageView& image) {
    if (!ready_) {
      detections_.clear();
      TINYYOLO_LOGE("TinyYoloModel", "detect called before a successful begin()");
      return Status::NotInitialized;
    }
    return detector_.detect(image, detections_);
  }

  const Detections& detections() const { return detections_; }
  const char* label(int class_id) const { return detector_.label(class_id); }
  int inputWidth() const { return network_.spec().input_w; }
  int inputHeight() const { return network_.spec().input_h; }
  bool isReady() const { return ready_; }
  const Detector& detector() const { return detector_; }
  /// Mutable access, for example to calibrate the int8 activations (Detector::calibrate()).
  Detector& detector() { return detector_; }

 private:
  Status load(ModelSource& source, const DetectorConfig& config, int max_detections) {
    TINYYOLO_CHECK(network_.parseCfg(source.cfg()));
    TINYYOLO_CHECK(network_.parseNames(source.names()));
    DarknetWeightSource weights(source.weights());
    TINYYOLO_CHECK(weights.open());
    TINYYOLO_CHECK(detector_.begin(network_.spec(), weights, config));
    TINYYOLO_CHECK(detections_.reserve(max_detections));
    return weights.finish();
  }

  NetworkDescription network_;
  Detector detector_;
  Detections detections_;
  bool ready_ = false;
};

}  // namespace tinyyolo
