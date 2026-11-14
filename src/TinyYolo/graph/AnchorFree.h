#pragma once
/// @file AnchorFree.h  The YOLO11 head: distribution (DFL) box decoding, sigmoid class scores,
/// and greedy NMS per class.
///
/// Each scale gives a box tensor [H][W][4 * 16] (for each side of the box, 16 bins of a
/// distribution) and a class tensor [H][W][classes] (logits). The cell (x, y) is an anchor at
/// ((x + 0.5) * stride, (y + 0.5) * stride). A side's distance is the expectation of its
/// distribution (softmax over the 16 bins), in cells.

#include <math.h>
#include <string.h>

#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/postprocess/Detection.h"

namespace tinyyolo {

constexpr int kDflBins = 16;  ///< bins per side of the box distribution (reg_max)

/// One scale of the head.
struct AnchorFreeLevel {
  const Tensor* box;  ///< [H][W][4 * reg_max]
  const Tensor* cls;  ///< [H][W][classes], logits
  int stride;         ///< network-input pixels per cell
  /// Bins per side of the box: 16 for the DFL head (YOLO11), 1 for a direct distance (YOLO26).
  int reg_max = kDflBins;
};

inline float sigmoidValue(float x) { return 1.0f / (1.0f + expf(-x)); }

/// The box of cell (x, y) of a level, as x1, y1, x2, y2 in network-input pixels.
inline void decodeAnchorBox(const AnchorFreeLevel& lv, int y, int x, float xyxy[4]) {
  const int bins = lv.reg_max;
  float dist[4];
  for (int side = 0; side < 4; ++side) {
    if (bins == 1) {
      dist[side] = lv.box->at(y, x, side);  // direct distance in cells (no distribution)
      continue;
    }
    float logits[kDflBins];
    float max_logit = -1e30f;
    for (int b = 0; b < bins; ++b) {
      logits[b] = lv.box->at(y, x, side * bins + b);
      if (logits[b] > max_logit) max_logit = logits[b];
    }
    float sum = 0.0f, expect = 0.0f;
    for (int b = 0; b < bins; ++b) {
      const float e = expf(logits[b] - max_logit);
      sum += e;
      expect += e * (float)b;
    }
    dist[side] = expect / sum;  // distance in cells
  }
  const float cx = ((float)x + 0.5f) * (float)lv.stride;
  const float cy = ((float)y + 0.5f) * (float)lv.stride;
  xyxy[0] = cx - dist[0] * (float)lv.stride;
  xyxy[1] = cy - dist[1] * (float)lv.stride;
  xyxy[2] = cx + dist[2] * (float)lv.stride;
  xyxy[3] = cy + dist[3] * (float)lv.stride;
}

/// Decodes all levels into candidates (one per class whose score is at least
/// `score_threshold`), then keeps the best boxes per class with greedy NMS at `iou_threshold`.
/// `candidates` must have room for the candidates (its capacity is the limit); `out` receives
/// the detections, in network-input pixels.
inline Status decodeAnchorFree(const AnchorFreeLevel* levels, int level_count, int classes,
                               float score_threshold, float iou_threshold, Detections& candidates,
                               Detections& out) {
  candidates.clear();
  out.clear();
  int dropped = 0;
  // Logit of the score threshold, lowered by a margin (see the loop below).
  const float t = score_threshold > 1e-6f ? (score_threshold < 1.0f - 1e-6f ? score_threshold : 1.0f - 1e-6f) : 1e-6f;
  const float logit_margin = logf(t / (1.0f - t)) - 0.01f;
  for (int l = 0; l < level_count; ++l) {
    const AnchorFreeLevel& lv = levels[l];
    const Shape s = lv.box->shape();
    for (int y = 0; y < s.h; ++y) {
      for (int x = 0; x < s.w; ++x) {
        // The box (the DFL decode, the expensive part) is needed only when a class passes the
        // threshold. A logit below the threshold's logit (with a small margin) cannot pass, so the
        // test is conservative and the result is the same as decoding every anchor.
        bool any = false;
        for (int c = 0; c < classes && !any; ++c) any = lv.cls->at(y, x, c) >= logit_margin;
        if (!any) continue;
        float xyxy[4];
        decodeAnchorBox(lv, y, x, xyxy);
        for (int c = 0; c < classes; ++c) {
          const float score = sigmoidValue(lv.cls->at(y, x, c));
          if (score < score_threshold) continue;
          Detection d;
          d.x = xyxy[0];
          d.y = xyxy[1];
          d.width = xyxy[2] - xyxy[0];
          d.height = xyxy[3] - xyxy[1];
          d.score = score;
          d.class_id = c;
          if (!candidates.push(d)) ++dropped;
        }
      }
    }
  }
  if (dropped > 0) {
    TINYYOLO_LOGW("AnchorFree", "%d candidate boxes dropped; raise max_candidates", dropped);
  }

  // Greedy NMS per class: keep the best remaining box, drop the overlapping boxes of its class.
  const int n = candidates.size();
  Buffer<uint8_t> suppressed;
  TINYYOLO_CHECK(suppressed.allocate((size_t)(n > 0 ? n : 1)));
  for (;;) {
    int best = -1;
    for (int i = 0; i < n; ++i) {
      if (suppressed[(size_t)i]) continue;
      if (best < 0 || candidates[i].score > candidates[best].score) best = i;
    }
    if (best < 0) break;
    const Detection kept = candidates[best];
    if (!out.push(kept)) {
      TINYYOLO_LOGW("AnchorFree", "more detections than the output holds; the rest are dropped");
      break;
    }
    suppressed[(size_t)best] = 1;
    for (int j = 0; j < n; ++j) {
      if (suppressed[(size_t)j]) continue;
      if (candidates[j].class_id == kept.class_id && intersectionOverUnion(candidates[j], kept) > iou_threshold) {
        suppressed[(size_t)j] = 1;
      }
    }
  }
  return Status::Ok;
}

}  // namespace tinyyolo
