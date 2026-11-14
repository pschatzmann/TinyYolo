#pragma once
/// @file Postprocessor.h  YOLO decoding (anchors, sigmoid, letterbox mapping) and greedy NMS.

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Types.h"
#include "TinyYolo/image/Letterbox.h"
#include "TinyYolo/model/Model.h"
#include "TinyYolo/postprocess/Detection.h"

namespace tinyyolo {

class Postprocessor {
 public:
  /// Allocates storage for up to `max_candidates` pre-NMS boxes.
  Status reserve(int max_candidates) {
    if (max_candidates <= 0) {
      TINYYOLO_LOGE("Postprocessor", "max_candidates must be positive, got %d", max_candidates);
      return Status::InvalidArgument;
    }
    TINYYOLO_CHECK(candidates_.reserve(max_candidates));
    TINYYOLO_CHECK(suppressed_.allocate((size_t)max_candidates));
    return Status::Ok;
  }

  /// Decodes all YOLO heads of `model`, applies NMS and writes the result to `out`.
  Status process(const Model& model, const LetterboxInfo& letterbox, float score_threshold,
                 float iou_threshold, Detections& out) {
    if (candidates_.capacity() == 0) {
      TINYYOLO_LOGE("Postprocessor", "process called before reserve()");
      return Status::NotInitialized;
    }
    candidates_.clear();
    out.clear();
    overflow_ = 0;
    const Shape& net = model.inputShape();
    for (int h = 0; h < model.headCount(); ++h) {
      decodeHead(model.head(h), net, letterbox, score_threshold);
    }
    if (overflow_ > 0) {
      TINYYOLO_LOGW("Postprocessor", "%d candidate boxes dropped; raise max_candidates above %d",
                    overflow_, candidates_.capacity());
    }
    nms(iou_threshold, out);
    return Status::Ok;
  }

  const Detections& candidates() const { return candidates_; }

 private:
  void decodeHead(const Layer& head, const Shape& net, const LetterboxInfo& lb,
                  float score_threshold) {
    const LayerSpec& spec = head.spec();
    const Tensor& output = head.output();
    const Shape grid = output.shape();
    const int anchors = spec.num_anchors;
    const int classes = spec.classes;
    const int per_anchor = 5 + classes;
    const float cell_w = (float)net.w / grid.w;
    const float cell_h = (float)net.h / grid.h;
    const float* data = output.data();

    for (int gy = 0; gy < grid.h; ++gy) {
      for (int gx = 0; gx < grid.w; ++gx) {
        const float* cell = data + ((size_t)gy * grid.w + gx) * grid.c;
        for (int a = 0; a < anchors; ++a) {
          const float* p = cell + a * per_anchor;
          const float objectness = sigmoid(p[4]);
          // The class score can never exceed objectness, so this rejects most cells early.
          if (objectness < score_threshold) continue;

          bool box_ready = false;
          float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
          for (int c = 0; c < classes; ++c) {
            const float score = objectness * sigmoid(p[5 + c]);
            if (score < score_threshold) continue;
            if (!box_ready) {
              box_ready = decodeBox(p, gx, gy, cell_w, cell_h, spec, a, lb, x1, y1, x2, y2);
              if (!box_ready) break;  // empty after clipping to the image
            }
            const Detection d = {x1, y1, x2 - x1, y2 - y1, score, c};
            if (!candidates_.push(d)) ++overflow_;
          }
        }
      }
    }
  }

  /// Box decoding for the head's style (see YoloDecode). Returns false if the clipped box is empty.
  static bool decodeBox(const float* p, int gx, int gy, float cell_w, float cell_h,
                        const LayerSpec& spec, int a, const LetterboxInfo& lb, float& x1,
                        float& y1, float& x2, float& y2) {
    const float anchor_w = spec.anchors[2 * a];
    const float anchor_h = spec.anchors[2 * a + 1];
    float cx, cy, bw, bh;
    if (spec.decode == YoloDecode::Darknet) {
      const float s = spec.scale_xy;
      cx = (sigmoid(p[0]) * s - 0.5f * (s - 1.0f) + gx) * cell_w;
      cy = (sigmoid(p[1]) * s - 0.5f * (s - 1.0f) + gy) * cell_h;
      bw = (float)exp(p[2]) * anchor_w;
      bh = (float)exp(p[3]) * anchor_h;
    } else {
      cx = (2.0f * sigmoid(p[0]) - 0.5f + gx) * cell_w;
      cy = (2.0f * sigmoid(p[1]) - 0.5f + gy) * cell_h;
      const float sw = 2.0f * sigmoid(p[2]);
      const float sh = 2.0f * sigmoid(p[3]);
      bw = sw * sw * anchor_w;
      bh = sh * sh * anchor_h;
    }

    x1 = cx - bw / 2.0f;
    y1 = cy - bh / 2.0f;
    x2 = cx + bw / 2.0f;
    y2 = cy + bh / 2.0f;
    lb.toSource(x1, y1);
    lb.toSource(x2, y2);
    x1 = clampf(x1, 0.0f, (float)lb.source_w);
    x2 = clampf(x2, 0.0f, (float)lb.source_w);
    y1 = clampf(y1, 0.0f, (float)lb.source_h);
    y2 = clampf(y2, 0.0f, (float)lb.source_h);
    return x2 > x1 && y2 > y1;
  }

  /// Greedy NMS, per class: repeatedly keep the best remaining box and suppress overlaps.
  void nms(float iou_threshold, Detections& out) {
    const int n = candidates_.size();
    for (int i = 0; i < n; ++i) suppressed_[(size_t)i] = 0;
    for (;;) {
      int best = -1;
      for (int i = 0; i < n; ++i) {
        if (suppressed_[(size_t)i]) continue;
        if (best < 0 || candidates_[i].score > candidates_[best].score) best = i;
      }
      if (best < 0) break;
      if (!out.push(candidates_[best])) {
        TINYYOLO_LOGW("Postprocessor", "detection limit of %d reached; remaining boxes dropped",
                      out.capacity());
        break;
      }
      suppressed_[(size_t)best] = 1;
      const Detection kept = candidates_[best];
      for (int j = 0; j < n; ++j) {
        if (suppressed_[(size_t)j]) continue;
        if (candidates_[j].class_id == kept.class_id &&
            intersectionOverUnion(candidates_[j], kept) > iou_threshold) {
          suppressed_[(size_t)j] = 1;
        }
      }
    }
  }

  Detections candidates_;
  Buffer<uint8_t> suppressed_;
  int overflow_ = 0;
};

}  // namespace tinyyolo
