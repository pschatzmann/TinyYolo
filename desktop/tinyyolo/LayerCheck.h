// The layer check of tinyyolo (--check): runs a YOLO model (its single-file .tym, docs/ModelFormat.md)
// on the exported test input, and compares every layer output and the head with the PyTorch reference
// of the export, then prints the detections on the test photo.
//
// The reference data are in the export folder next to the model: <base>.tym is the model file,
// <dir>/export holds input_640.bin and ref_layerNN_0.bin (written by tools/yolo/yolo_export.py with the
// test photo). The activation scales (shared or per tensor) are the ones the .tym file was written with.

#pragma once

#include <stdio.h>

#include <cmath>
#include <string>
#include <vector>

#include "TinyYolo.h"
#include "TinyYolo/app/ModelFile.h"
#include "TinyYolo/backend/DotProductBackend.h"
#include "TinyYolo/graph/AnchorFree.h"
#include "TinyYolo/graph/Graph.h"

namespace layercheck {

using namespace tinyyolo;

template <typename T>
bool readRaw(const std::string& path, std::vector<T>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  fseek(f, 0, SEEK_END);
  const long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize((size_t)bytes / sizeof(T));
  const size_t got = fread(out.data(), sizeof(T), out.size(), f);
  fclose(f);
  return got == out.size();
}

inline bool readFloats(const std::string& path, std::vector<float>& out) { return readRaw(path, out); }

/// Runs the check for the model `base` (a path without the extension, e.g. models/yolo11n/yolo11n).
/// Returns 0 when every layer and the head match their references.
inline int run(const std::string& base, bool use_int8) {
  const std::string folder = base.find('/') == std::string::npos ? "." : base.substr(0, base.find_last_of('/'));
  const std::string export_dir = folder + "/export";

  std::vector<uint8_t> model_bytes;
  std::vector<float> input;
  if (!readRaw(base + ".tym", model_bytes) || !readFloats(export_dir + "/input_640.bin", input)) {
    fprintf(stderr, "tinyyolo: cannot read %s.tym or %s/input_640.bin\n", base.c_str(), export_dir.c_str());
    return 1;
  }
  tym::ModelFile file;
  if (file.begin(model_bytes.data(), model_bytes.size()) != Status::Ok) {
    fprintf(stderr, "tinyyolo: %s.tym is not a valid model file\n", base.c_str());
    return 1;
  }
  const GraphModel& model = file.model();
  const std::vector<std::pair<int, int>>& outputs = file.layerOutputs();
  if (outputs.empty()) {
    fprintf(stderr, "tinyyolo: %s.tym has no layer outputs (write it with the export, tools/yolo/yolo_export.py)\n", base.c_str());
    return 1;
  }
  if (use_int8 && (file.int8Weights() == nullptr || file.scaleCount() == 0)) {
    fprintf(stderr, "tinyyolo: %s.tym has no int8 weights\n", base.c_str());
    return 1;
  }

  // The input is NCHW (1, 3, 640, 640); the graph works on HWC.
  Tensor image;
  if (image.allocate(Shape(640, 640, 3)) != Status::Ok) return 1;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < 640; ++y) {
      for (int x = 0; x < 640; ++x) image.at(y, x, c) = input[((size_t)c * 640 + y) * 640 + x];
    }
  }

  Graph graph;
  // Keep the outputs that are compared below (the layers and the head).
  std::vector<int> keep;
  for (const auto& output : outputs) keep.push_back(output.second);
  for (int l = 0; l < 6; ++l) keep.push_back(model.detect_heads[l / 2][l % 2]);
  if (graph.begin(model.ops, model.op_count, model.params, model.param_count, file.floatWeights(),
                  model.tensor_count, model.input_id, keep.data(), (int)keep.size(),
                  use_int8 ? file.int8Weights() : nullptr, use_int8 ? file.int8Scales() : nullptr,
                  use_int8 ? model.quant : nullptr) != Status::Ok) {
    return 1;
  }
  DotProductBackend backend;
  if (graph.run(backend, image) != Status::Ok) return 1;

  int failures = 0;
  for (size_t i = 0; i < outputs.size(); ++i) {
    const int layer = outputs[i].first;
    const Tensor& t = graph.tensor(outputs[i].second);
    const Shape s = t.shape();
    char name[64];
    snprintf(name, sizeof(name), "/ref_layer%02d_0.bin", layer);
    std::vector<float> ref;
    if (!readFloats(export_dir + name, ref) || ref.size() != s.elements()) {
      printf("layer %2d: reference missing or of another size (%zu values, ours %zu)\n", layer, ref.size(),
             s.elements());
      ++failures;
      continue;
    }
    double worst = 0.0, range = 0.0;
    for (int c = 0; c < s.c; ++c) {
      for (int y = 0; y < s.h; ++y) {
        for (int x = 0; x < s.w; ++x) {
          const float r = ref[((size_t)c * s.h + y) * s.w + x];  // reference is NCHW
          const double d = std::fabs((double)t.at(y, x, c) - (double)r);
          if (d > worst) worst = d;
          if (std::fabs((double)r) > range) range = std::fabs((double)r);
        }
      }
    }
    printf("layer %2d: %3dx%3dx%4d  max abs error %.3e  (range %.3e)\n", layer, s.h, s.w, s.c, worst, range);
  }

  // The head: the dense output (1, 84, 8400) = (cx, cy, w, h, 80 class scores) per anchor.
  AnchorFreeLevel levels[3];
  for (int l = 0; l < 3; ++l) {
    levels[l].box = &graph.tensor(model.detect_heads[l][0]);
    levels[l].cls = &graph.tensor(model.detect_heads[l][1]);
    levels[l].stride = model.detect_strides[l];
    levels[l].reg_max = model.detect_reg_max;
  }
  const int classes = model.detect_classes;
  int anchors = 0;
  for (int l = 0; l < 3; ++l) anchors += levels[l].box->shape().h * levels[l].box->shape().w;
  std::vector<float> dense((size_t)(4 + classes) * anchors);
  int a0 = 0;
  for (int l = 0; l < 3; ++l) {
    const Shape s = levels[l].box->shape();
    for (int y = 0; y < s.h; ++y) {
      for (int x = 0; x < s.w; ++x, ++a0) {
        float xyxy[4];
        decodeAnchorBox(levels[l], y, x, xyxy);
        dense[(size_t)0 * anchors + a0] = (xyxy[0] + xyxy[2]) * 0.5f;
        dense[(size_t)1 * anchors + a0] = (xyxy[1] + xyxy[3]) * 0.5f;
        dense[(size_t)2 * anchors + a0] = xyxy[2] - xyxy[0];
        dense[(size_t)3 * anchors + a0] = xyxy[3] - xyxy[1];
        for (int c = 0; c < classes; ++c) {
          dense[(size_t)(4 + c) * anchors + a0] = sigmoidValue(levels[l].cls->at(y, x, c));
        }
      }
    }
  }
  std::vector<float> head_ref;
  if (readFloats(export_dir + "/ref_layer23_0.bin", head_ref) && head_ref.size() == dense.size()) {
    double worst = 0.0;
    for (size_t i = 0; i < dense.size(); ++i) {
      const double d = std::fabs((double)dense[i] - (double)head_ref[i]);
      if (d > worst) worst = d;
    }
    printf("head (1, %d, %d): max abs error %.3e\n", 4 + classes, anchors, worst);
  } else {
    printf("head reference missing\n");
    ++failures;
  }

  // Detections, with the same thresholds as the reference predictor (conf 0.25, iou 0.7),
  // mapped back to the photo: the letterbox was scaled by `scale` and padded by 106 pixels.
  LetterboxInfo lb;
  lb.scale = 640.0f / 375.0f < 640.0f / 250.0f ? 640.0f / 375.0f : 640.0f / 250.0f;
  lb.pad_x = 0.0f;
  lb.pad_y = 106.0f;  // (640 - 427) / 2, the resized height is 427 pixels
  lb.source_w = 375;
  lb.source_h = 250;
  Detections candidates, detections;
  candidates.reserve(20000);
  detections.reserve(100);
  decodeAnchorFree(levels, 3, classes, 0.25f, 0.7f, candidates, detections);
  printf("detections: %d\n", detections.size());
  for (int i = 0; i < detections.size(); ++i) {
    const Detection& d = detections[i];
    float x1 = d.x, y1 = d.y, x2 = d.x + d.width, y2 = d.y + d.height;
    lb.toSource(x1, y1);
    lb.toSource(x2, y2);
    printf("  class %2d score %.3f box [%.1f, %.1f, %.1f, %.1f]\n", d.class_id, d.score, x1, y1, x2, y2);
  }
  return failures == 0 ? 0 : 1;
}

}  // namespace layercheck
