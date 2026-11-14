#pragma once
/// @file Config.h
/// Compile-time configuration of TinyYolo. Every macro can be overridden with a
/// -D compiler flag or by defining it before including TinyYolo.h.

#include <stddef.h>
#include <stdint.h>

#define TINYYOLO_VERSION "0.1.0"

/// Log messages below this level are removed at compile time (saves flash).
/// 0 = Debug, 1 = Info, 2 = Warning, 3 = Error, 4 = Off
#ifndef TINYYOLO_MIN_LOG_LEVEL
#define TINYYOLO_MIN_LOG_LEVEL 0
#endif

/// Size of the stack buffer used to format one log line.
#ifndef TINYYOLO_LOG_BUFFER_SIZE
#define TINYYOLO_LOG_BUFFER_SIZE 160
#endif

#if defined(__GNUC__)
#define TINYYOLO_PRINTF_CHECK(fmt_index, first_arg) \
  __attribute__((format(printf, fmt_index, first_arg)))
#else
#define TINYYOLO_PRINTF_CHECK(fmt_index, first_arg)
#endif

// Platform identification. The optimised backends of step 2 select on these.
#if defined(ESP32)
#define TINYYOLO_PLATFORM "esp32"
#elif defined(ARDUINO_ARCH_NRF52)
#define TINYYOLO_PLATFORM "nrf52"
#elif defined(ARDUINO_ARCH_RP2040)
#define TINYYOLO_PLATFORM "rp2040"
#elif defined(ARDUINO_ARCH_SAMD)
#define TINYYOLO_PLATFORM "samd"
#elif defined(ARDUINO_ARCH_STM32)
#define TINYYOLO_PLATFORM "stm32"
#elif defined(ARDUINO)
#define TINYYOLO_PLATFORM "arduino-generic"
#elif defined(__linux__) || defined(__APPLE__) || defined(_WIN32)
#define TINYYOLO_PLATFORM "host"
#else
#define TINYYOLO_PLATFORM "generic"
#endif
