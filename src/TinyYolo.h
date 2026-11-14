#pragma once
/// @file TinyYolo.h  Umbrella header. Include this one file in your sketch.
///
/// TinyYolo: header-only YOLO object detection for Arduino-class boards.

#include "TinyYolo/core/Config.h"
#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Types.h"
#include "TinyYolo/core/Tensor.h"
#include "TinyYolo/core/Vector.h"
#include "TinyYolo/core/BufferStream.h"
#include "TinyYolo/core/WeightSource.h"
#include "TinyYolo/image/ImageView.h"
#include "TinyYolo/image/Letterbox.h"
#include "TinyYolo/image/Preprocessor.h"
#include "TinyYolo/image/Draw.h"
#include "TinyYolo/backend/Backend.h"
#include "TinyYolo/backend/ReferenceBackend.h"
#include "TinyYolo/backend/DotProductBackend.h"
#include "TinyYolo/backend/Esp32Backend.h"
#include "TinyYolo/platform/Features.h"
#include "TinyYolo/platform/Memory.h"
#include "TinyYolo/platform/Parallel.h"
#include "TinyYolo/platform/Task.h"
#include "TinyYolo/model/LayerSpec.h"
#include "TinyYolo/model/Layer.h"
#include "TinyYolo/model/Model.h"
#include "TinyYolo/model/ModelBlob.h"
#include "TinyYolo/model/DarknetWeights.h"
#include "TinyYolo/model/NetworkDescription.h"
#include "TinyYolo/postprocess/Detection.h"
#include "TinyYolo/postprocess/Postprocessor.h"
#include "TinyYolo/platform/Platform.h"
#include "TinyYolo/Detector.h"


// High-level helpers: model loading, frame generation and text reports. The camera and
// display integrations are separate headers: TinyYolo/TinyCamera and TinyYolo/TinyGPU.
#include "TinyYolo/app/ModelSource.h"
#include "TinyYolo/app/TinyYoloModel.h"
#include "TinyYolo/app/SyntheticFrame.h"
#include "TinyYolo/core/Output.h"

// Arduino sketches get the tinyyolo namespace automatically, so Detector and friends
// can be used without a namespace prefix or a using-directive.
#if defined(ARDUINO)
using namespace tinyyolo;
#endif
