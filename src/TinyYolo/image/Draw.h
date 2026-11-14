#pragma once
/// @file Draw.h  Draws detection boxes onto an RGB888 frame (3 bytes per pixel, row-major).
///
/// Works on any writable buffer, including a camera frame that has been converted to RGB888.
/// Pixels outside the buffer are skipped, so boxes at the image border are safe.

#include <stdint.h>

#include "TinyYolo/postprocess/Detection.h"

namespace tinyyolo {

struct Rgb {
  uint8_t r;
  uint8_t g;
  uint8_t b;
};

inline void setRgbPixel(uint8_t* rgb, int width, int height, int x, int y, Rgb color) {
  if (x < 0 || y < 0 || x >= width || y >= height) return;
  uint8_t* p = rgb + ((size_t)y * (size_t)width + (size_t)x) * 3;
  p[0] = color.r;
  p[1] = color.g;
  p[2] = color.b;
}

/// Outlines the rectangle [x0, x1] x [y0, y1] (inclusive pixel coordinates).
/// `thickness` pixels are drawn inwards from the edge.
inline void drawRectangle(uint8_t* rgb, int width, int height, int x0, int y0, int x1, int y1,
                          Rgb color, int thickness = 1) {
  if (rgb == nullptr || width <= 0 || height <= 0 || thickness <= 0) return;
  if (x1 < x0) {
    const int t = x0;
    x0 = x1;
    x1 = t;
  }
  if (y1 < y0) {
    const int t = y0;
    y0 = y1;
    y1 = t;
  }
  for (int t = 0; t < thickness; ++t) {
    if (x0 + t > x1 - t || y0 + t > y1 - t) break;  // the box is too small for more lines
    for (int x = x0 + t; x <= x1 - t; ++x) {
      setRgbPixel(rgb, width, height, x, y0 + t, color);
      setRgbPixel(rgb, width, height, x, y1 - t, color);
    }
    for (int y = y0 + t; y <= y1 - t; ++y) {
      setRgbPixel(rgb, width, height, x0 + t, y, color);
      setRgbPixel(rgb, width, height, x1 - t, y, color);
    }
  }
}

/// Outlines a detection box given in source-image pixels, as returned by Detector::detect().
inline void drawDetection(uint8_t* rgb, int width, int height, const Detection& d, Rgb color,
                          int thickness = 1) {
  const int x0 = (int)d.x;
  const int y0 = (int)d.y;
  const int x1 = (int)(d.x + d.width) - 1;
  const int y1 = (int)(d.y + d.height) - 1;
  drawRectangle(rgb, width, height, x0, y0, x1, y1, color, thickness);
}

}  // namespace tinyyolo
