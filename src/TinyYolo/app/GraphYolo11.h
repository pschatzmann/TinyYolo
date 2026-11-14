#pragma once
/// @file GraphYolo11.h  YOLO11 on the TinyYolo graph path: one call per frame.
///
/// The model is described by the generated table (tools/yolo/yolo_gen_graph.py writes `kModel` into
/// the .inc file). The weights come from the caller: the float parameters, the int8 weights and
/// their scales (tools/yolo/yolo_export.py writes them; the examples read them from a partition).
/// `detect()` letterboxes the image to the model input, runs the graph and decodes the three
/// detect heads. The boxes are returned in source-image pixels.
///
/// Speed at 320x320 (int8, the shared-scale table, examples/Yolo11Graph): about 1.5 s per frame on the
/// ESP32-P4 and 6.6 s on the ESP32-S3 (see the README). The int8 convolutions use the backend's kernels
/// (see Esp32NnBackend.h).

#include "TinyYolo/backend/Backend.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/graph/AnchorFree.h"
#include "TinyYolo/graph/Graph.h"
#include "TinyYolo/image/ImageView.h"
#include "TinyYolo/image/Letterbox.h"
#include "TinyYolo/image/Preprocessor.h"
#include "TinyYolo/platform/Clock.h"
#include "TinyYolo/postprocess/Detection.h"

namespace tinyyolo {

/// The description of one exported YOLO11-style graph (see the generated table).
struct GraphModel {
  const GraphOp* ops = nullptr;
  int op_count = 0;
  const GraphParam* params = nullptr;
  int param_count = 0;
  int tensor_count = 0;
  int input_id = 0;
  const GraphQuant* quant = nullptr;            ///< the activation scales (kQuant or kQuantPerTensor)
  const int (*detect_heads)[2] = nullptr;       ///< per detect level: box tensor id, class tensor id
  const int* detect_strides = nullptr;          ///< per detect level: network-input pixels per cell
  int detect_levels = 3;                        ///< at most 3
  int detect_classes = 80;
  int detect_reg_max = 16;                      ///< box bins per side: 16 (DFL) or 1 (direct)
  int input_size = 640;                         ///< square network input
};

class GraphYolo11 {
 public:
  GraphYolo11() {}

  /// Sets up the graph. `weights` (float parameters), `qweights` (int8 weights) and `qscales`
  /// (weight scales) must stay valid while the detector is used; the examples copy the int8 data
  /// into PSRAM first. `backend` is used for every convolution and must outlive the detector.
  Status begin(const GraphModel& model, Backend& backend, const float* weights, const int8_t* qweights,
               const float* qscales, float score_threshold = 0.25f, float iou_threshold = 0.7f) {
    if (model.detect_levels < 1 || model.detect_levels > 3 || model.detect_heads == nullptr ||
        model.detect_strides == nullptr || model.input_size <= 0) {
      TINYYOLO_LOGE("GraphYolo11", "invalid model description");
      return Status::InvalidArgument;
    }
    model_ = model;
    backend_ = &backend;
    score_threshold_ = score_threshold;
    iou_threshold_ = iou_threshold;
    // The detect heads stay alive after their last use; everything else is freed early.
    int keep_count = 0;
    for (int l = 0; l < model.detect_levels; ++l) {
      keep_[keep_count++] = model.detect_heads[l][0];
      keep_[keep_count++] = model.detect_heads[l][1];
    }
    TINYYOLO_CHECK(graph_.begin(model.ops, model.op_count, model.params, model.param_count, weights,
                                model.tensor_count, model.input_id, keep_, keep_count, qweights, qscales,
                                model.quant));
    TINYYOLO_CHECK(input_.allocate(Shape(model.input_size, model.input_size, 3)));
    TINYYOLO_CHECK(candidates_.reserve(2000));
    return Status::Ok;
  }

  /// Runs the network on `image` and writes the detections, in source-image pixels, to `out`.
  /// `out` must have room for the detections (its capacity is the limit).
  Status detect(const ImageView& image, Detections& out) {
    if (backend_ == nullptr) return Status::NotInitialized;
    Preprocessor preprocessor;
    LetterboxInfo letterbox;
    TINYYOLO_CHECK(preprocessor.letterbox(image, input_, letterbox));

    const uint32_t t0 = nowMicros();
    TINYYOLO_CHECK(graph_.run(*backend_, input_));
    const uint32_t t1 = nowMicros();

    AnchorFreeLevel levels[3];
    for (int l = 0; l < model_.detect_levels; ++l) {
      levels[l].box = &graph_.tensor(model_.detect_heads[l][0]);
      levels[l].cls = &graph_.tensor(model_.detect_heads[l][1]);
      levels[l].stride = model_.detect_strides[l];
      levels[l].reg_max = model_.detect_reg_max;
    }
    TINYYOLO_CHECK(decodeAnchorFree(levels, model_.detect_levels, model_.detect_classes, score_threshold_,
                                    iou_threshold_, candidates_, out));
    const uint32_t t2 = nowMicros();
    graph_micros_ = t1 - t0;
    head_micros_ = t2 - t1;

    // The decoder works in network-input pixels: map the corners back to the source image.
    for (int i = 0; i < out.size(); ++i) {
      Detection& d = out[i];
      float x1 = d.x, y1 = d.y, x2 = d.x + d.width, y2 = d.y + d.height;
      letterbox.toSource(x1, y1);
      letterbox.toSource(x2, y2);
      d.x = x1;
      d.y = y1;
      d.width = x2 - x1;
      d.height = y2 - y1;
    }
    return Status::Ok;
  }

  /// The graph (for profiling: Graph::enableProfiling, phaseMicros, opMicros).
  Graph& graph() { return graph_; }
  /// Time of the last detect(): the graph, and the decoding of the heads (microseconds).
  uint32_t lastGraphMicros() const { return graph_micros_; }
  uint32_t lastHeadMicros() const { return head_micros_; }

 private:
  Graph graph_;
  GraphModel model_;
  Backend* backend_ = nullptr;
  Tensor input_;
  Detections candidates_;
  int keep_[6] = {0, 0, 0, 0, 0, 0};
  float score_threshold_ = 0.25f;
  float iou_threshold_ = 0.7f;
  uint32_t graph_micros_ = 0;
  uint32_t head_micros_ = 0;
};

}  // namespace tinyyolo
