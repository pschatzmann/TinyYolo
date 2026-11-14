#pragma once
/// @file Platform.h  Platform selection point for the optimised kernels (step 2).

#include "TinyYolo/backend/DotProductBackend.h"
#include "TinyYolo/backend/Esp32Backend.h"
#include "TinyYolo/backend/ReferenceBackend.h"
#include "TinyYolo/backend/cmsis/CmsisBackend.h"
#include "TinyYolo/core/Config.h"

namespace tinyyolo {

/// Backend used when Detector::begin() receives none:
///  - ESP32: Esp32Backend (esp-dsp dot products, both cores)
///  - RP2040/RP2350: CmsisBackend (CMSIS-NN on the Cortex-M33; the portable kernels elsewhere)
///  - everything else: DotProductBackend (portable, threads on hosts)
/// ReferenceBackend stays the correctness baseline the tests compare against.
#if defined(ESP32)
using DefaultBackend = Esp32Backend;
#elif defined(ARDUINO_ARCH_RP2040)
using DefaultBackend = CmsisBackend;
#else
using DefaultBackend = DotProductBackend;
#endif

inline const char* platformName() { return TINYYOLO_PLATFORM; }

}  // namespace tinyyolo
