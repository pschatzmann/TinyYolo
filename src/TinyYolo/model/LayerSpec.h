#pragma once
/// @file LayerSpec.h  Immutable network description. Designed to live in flash (const arrays).

#include "TinyYolo/core/Quantize.h"
#include "TinyYolo/core/Types.h"

namespace tinyyolo {

/// Input reference: the previous layer's output (the network input for layer 0).
constexpr int kPrevious = -1;
/// Input reference: no second input.
constexpr int kNone = -2;
/// precision_override: use the model's weight precision and its head rule.
constexpr int kInheritPrecision = -1;

enum class LayerType : uint8_t { Conv, MaxPool, Upsample, Route, Yolo, Add, Split };

/// How a YOLO head turns raw predictions into box coordinates.
enum class YoloDecode : uint8_t {
  V5,      ///< YOLOv5 style: bx = (2*sigmoid(tx) - 0.5 + gx) * stride, bw = (2*sigmoid(tw))^2 * anchor
  Darknet  ///< Darknet/YOLOv3/v4 style: bx = (sigmoid(tx)*scale_xy - 0.5*(scale_xy-1) + gx) * stride,
           ///< bw = exp(tw) * anchor
};

struct LayerSpec {
  LayerType type;
  Activation activation;
  int filters;           ///< Conv: output channels
  int kernel;            ///< Conv / MaxPool: kernel size
  int stride;            ///< Conv / MaxPool: stride; Upsample: factor
  /// Conv: padding on each side, -1 = kernel/2.
  /// MaxPool: total padding (pad_total); the window starts at -pad_total/2 (Darknet convention).
  int pad;
  int from[2];           ///< Input layers: kPrevious, kNone or an absolute layer index
  int classes;           ///< Yolo: number of classes
  int num_anchors;       ///< Yolo: number of anchors
  const float* anchors;  ///< Yolo: (width, height) pairs in network-input pixels
  bool batch_norm;       ///< Conv: weights are stored with batch-norm parameters (Darknet)
  YoloDecode decode;     ///< Yolo: box decoding style
  float scale_xy;        ///< Yolo (Darknet decode): centre scaling, 1.0 for YOLOv3
  /// Conv: kInheritPrecision (follow the model), or WeightPrecision cast to int. Set with
  /// withPrecision(), which overrides the model's choice for this layer only.
  int precision_override;

  static constexpr LayerSpec conv(int filters, int kernel, int stride,
                                  Activation act = Activation::Leaky, int pad = -1,
                                  bool batch_norm = false) {
    return LayerSpec{LayerType::Conv, act, filters, kernel, stride, pad,
                     {kPrevious, kNone}, 0, 0, nullptr, batch_norm, YoloDecode::V5, 1.0f,
                     kInheritPrecision};
  }
  /// `pad_total` is the total padding; 0 gives a plain pooling window.
  static constexpr LayerSpec maxPool(int kernel, int stride, int pad_total = 0) {
    return LayerSpec{LayerType::MaxPool, Activation::Linear, 0, kernel, stride, pad_total,
                     {kPrevious, kNone}, 0, 0, nullptr, false, YoloDecode::V5, 1.0f,
                     kInheritPrecision};
  }
  static constexpr LayerSpec upsample(int factor = 2) {
    return LayerSpec{LayerType::Upsample, Activation::Linear, 0, 0, factor, 0,
                     {kPrevious, kNone}, 0, 0, nullptr, false, YoloDecode::V5, 1.0f,
                     kInheritPrecision};
  }
  /// Channel concatenation of two layers (or a copy when `second` is kNone).
  static constexpr LayerSpec route(int first, int second = kNone) {
    return LayerSpec{LayerType::Route, Activation::Linear, 0, 0, 0, 0,
                     {first, second}, 0, 0, nullptr, false, YoloDecode::V5, 1.0f,
                     kInheritPrecision};
  }
  static constexpr LayerSpec yolo(int classes, const float* anchors, int num_anchors,
                                  YoloDecode decode = YoloDecode::V5, float scale_xy = 1.0f) {
    return LayerSpec{LayerType::Yolo, Activation::Linear, 0, 0, 0, 0,
                     {kPrevious, kNone}, classes, num_anchors, anchors, false, decode, scale_xy,
                     kInheritPrecision};
  }

  /// One of `groups` equal channel groups of a layer (Darknet [route] with groups/group_id).
  /// `kernel` holds groups and `stride` holds group_id.
  static constexpr LayerSpec split(int from, int groups, int group_id) {
    return LayerSpec{LayerType::Split, Activation::Linear, 0, groups, group_id, 0,
                     {from, kNone}, 0, 0, nullptr, false, YoloDecode::V5, 1.0f,
                     kInheritPrecision};
  }
  /// Element-wise sum of two layers of the same shape (Darknet [shortcut]), then `act`.
  static constexpr LayerSpec add(int first, int second, Activation act = Activation::Linear) {
    return LayerSpec{LayerType::Add, act, 0, 0, 0, 0,
                     {first, second}, 0, 0, nullptr, false, YoloDecode::V5, 1.0f,
                     kInheritPrecision};
  }

  /// Copy of this spec that reads its primary input from layer `index`.
  constexpr LayerSpec input(int index) const {
    return LayerSpec{type,      activation,       filters,    kernel,   stride,
                     pad,       {index, from[1]}, classes,    num_anchors,
                     anchors,   batch_norm,       decode,     scale_xy,
                     precision_override};
  }

  /// Copy of this spec with its own weight precision (conv layers only). This overrides the
  /// model's choice, including the rule that keeps YOLO head inputs in float32.
  constexpr LayerSpec withPrecision(WeightPrecision p) const {
    return LayerSpec{type,      activation,       filters,    kernel,   stride,
                     pad,       {from[0], from[1]}, classes,  num_anchors,
                     anchors,   batch_norm,       decode,     scale_xy,
                     static_cast<int>(p)};
  }
};

/// Whole-network description: input size, layer list and optional class labels.
struct NetworkSpec {
  int input_h;
  int input_w;
  int input_c;
  const LayerSpec* layers;
  int layer_count;
  const char* const* labels;
  int label_count;
};

}  // namespace tinyyolo
