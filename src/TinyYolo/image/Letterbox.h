#pragma once
/// @file Letterbox.h  Mapping between source image and network input coordinates.

namespace tinyyolo {

/// Describes how a source image was scaled and padded into the network input.
struct LetterboxInfo {
  float scale;
  float pad_x;
  float pad_y;
  int source_w;
  int source_h;

  LetterboxInfo() : scale(1.0f), pad_x(0.0f), pad_y(0.0f), source_w(0), source_h(0) {}

  /// Maps a point in network-input pixels back to source-image pixels (in place).
  void toSource(float& x, float& y) const {
    x = (x - pad_x) / scale;
    y = (y - pad_y) / scale;
  }
};

}  // namespace tinyyolo
