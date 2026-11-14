#pragma once
/// @file Features.h  Which optional platform features are available in this build.
///
/// Each feature is detected at compile time and falls back to portable code when missing.
/// Nothing here is a hard dependency: a plain desktop or Arduino build compiles unchanged.

// Raspberry Pi RP2040/RP2350 (Arduino-Pico) with PSRAM: __psram_malloc() allocates from the board PSRAM.
#if defined(ARDUINO_ARCH_RP2040) && __has_include("psram.h")
#define TINYYOLO_HAS_RP_PSRAM 1  // Arduino-Pico: __psram_malloc() allocates from the board PSRAM
#endif

// ESP32 family: Xtensa or RISC-V cores with FreeRTOS and esp-dsp in the Arduino core.
#if defined(ESP32)
#define TINYYOLO_HAS_FREERTOS 1
#define TINYYOLO_HAS_HEAP_CAPS 1  // heap_caps_* allocates from PSRAM when present
#endif

#if defined(ESP32) && __has_include(<dsps_dotprod.h>)
#define TINYYOLO_HAS_ESP_DSP 1  // esp-dsp SIMD dot products
#endif

// ESP-NN int8 convolution kernels: ESP32-S3 (Xtensa, esp32s3 kernels) and ESP32-P4 (RISC-V PIE
// kernels). The sdkconfig macro tells the chip apart (the core ships ESP-NN headers for other
// chips too). The library must also be linked; see docs/Quantization.md.
#if defined(ESP32) && __has_include(<sdkconfig.h>)
#include <sdkconfig.h>
#endif
// The S3 ESP-NN kernels are vendored in backend/esp32s3/esp-nn (no extra library or include path).
#if defined(ESP32) && defined(CONFIG_IDF_TARGET_ESP32S3)
#define TINYYOLO_HAS_ESP_NN 1
#define TINYYOLO_ESP_NN_S3 1
#endif
#if defined(ESP32) && defined(CONFIG_IDF_TARGET_ESP32P4) && __has_include(<esp_nn_riscv_pie.h>)
#define TINYYOLO_HAS_ESP_NN 1
#define TINYYOLO_ESP_NN_P4 1
#endif

// Host builds (Linux, macOS, Windows, Raspberry Pi OS) have std::thread.
#if !defined(TINYYOLO_HAS_FREERTOS) && (defined(__linux__) || defined(__APPLE__) || defined(_WIN32))
#define TINYYOLO_HAS_THREADS 1
#endif
