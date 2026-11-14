/// Second translation unit that includes the header-only library. Linking it with the
/// test program proves that all definitions are inline (no multiple-definition errors).
/// It is also compiled alone with the strict Arduino-like flags (C++11, no exceptions/RTTI).

#include "TinyYolo.h"

namespace tinyyolo_second_tu {

int touchLibrary() {
  using namespace tinyyolo;
  static const float anchors[] = {8, 8};
  static const LayerSpec layers[] = {LayerSpec::conv(6, 1, 1), LayerSpec::yolo(1, anchors, 1)};
  static const NetworkSpec spec = {8, 8, 3, layers, 2, nullptr, 0};
  Detector detector;
  ArrayWeightSource weights(nullptr, 0);
  Detections out;
  (void)out;
  (void)spec;
  TINYYOLO_LOGI("SecondTU", "linked %s", platformName());
  return detector.isReady() ? 1 : 0;
}

}  // namespace tinyyolo_second_tu
