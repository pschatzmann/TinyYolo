#pragma once
/// @file LcdOverlay.h  Shows a frame and its detection boxes on a TinyGPU LCD board.
///
/// Requires the TinyGPU library. Include this header after <TinyGPU.h> and
/// <TinyGPU/Boards/LCDBoards.h>.

#if !__has_include(<TinyGPU.h>)
#error "TinyYolo/TinyGPU/LcdOverlay.h needs the TinyGPU library: install it (https://github.com/pschatzmann/TinyGPU) or remove this include."
#endif

#include <TinyGPU.h>
#include <TinyGPU/Boards/LCDBoards.h>

#include "TinyYolo/image/ImageView.h"
#include "TinyYolo/postprocess/Detection.h"

namespace tinyyolo {

class LcdOverlay {
 public:
  /// `board` must outlive the overlay.
  explicit LcdOverlay(LCDBoard& board)
      : board_(board), screen_(board.width(), board.height(), FontRGB565) {}

  /// Starts the board and the frame buffer. Call once in setup().
  void begin() {
    board_.begin();
    screen_.begin();
  }

  /// Scales `frame` to fit the panel (centred, integer scale), outlines each detection in red
  /// and pushes the result to the display.
  void show(const ImageView& frame, const Detections& detections) {
    const int panel_w = (int)board_.width();
    const int panel_h = (int)board_.height();
    int scale = panel_w / frame.width() < panel_h / frame.height() ? panel_w / frame.width()
                                                                    : panel_h / frame.height();
    if (scale < 1) scale = 1;
    const int x0 = (panel_w - frame.width() * scale) / 2;
    const int y0 = (panel_h - frame.height() * scale) / 2;

    screen_.clear(RGB565(0, 0, 0));
    for (int y = 0; y < frame.height(); ++y) {
      for (int x = 0; x < frame.width(); ++x) {
        float rgb[3];
        frame.rgbAt(x, y, rgb);
        screen_.fillRect((size_t)(x0 + x * scale), (size_t)(y0 + y * scale), (size_t)scale,
                         (size_t)scale,
                         RGB565((uint8_t)(rgb[0] * 255.0f + 0.5f), (uint8_t)(rgb[1] * 255.0f + 0.5f),
                                (uint8_t)(rgb[2] * 255.0f + 0.5f)));
      }
    }
    for (const Detection& d : detections) {
      screen_.drawRect((size_t)(x0 + (int)(d.x * scale)), (size_t)(y0 + (int)(d.y * scale)),
                       (size_t)((int)(d.width * scale)), (size_t)((int)(d.height * scale)),
                       RGB565(255, 0, 0));
    }
    board_.display().writeData(screen_);
  }

 private:
  LCDBoard& board_;
  FrameBuffer<RGB565> screen_;
};

}  // namespace tinyyolo
