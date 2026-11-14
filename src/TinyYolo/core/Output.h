#pragma once
/// @file Output.h  Formatted text output: one printf-style call per line.

#include <stdarg.h>
#include <stdio.h>

#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

/// Prints formatted lines to one Print target. On Arduino the target defaults to Serial.
///
///   Output out;                 // global in a sketch: prints to Serial
///   out.printf("%s %.2f", label, score);
class Output {
 public:
#if defined(ARDUINO)
  explicit Output(Print& target = Serial) : target_(&target) {}
#else
  explicit Output(Print& target) : target_(&target) {}
#endif

  /// Waits, at most `timeout_ms`, until the serial port is open, so that the first lines are
  /// not lost on boards with native USB. Returns true when the port is open. On other platforms
  /// it returns true at once.
  bool waitForSerialPort(unsigned long timeout_ms) const {
#if defined(ARDUINO)
    const unsigned long start = millis();
    while (!Serial && millis() - start < timeout_ms) {
    }
    return static_cast<bool>(Serial);
#else
    (void)timeout_ms;
    return true;
#endif
  }

  /// Formats the arguments like printf and prints the result as one line (the line end is
  /// added). A line longer than 255 characters is cut.
  __attribute__((format(printf, 2, 3))) void printf(const char* format, ...) {
    char line[kMaxLine];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);
    target_->println(line);
  }

 private:
  static constexpr int kMaxLine = 256;
  Print* target_;
};

}  // namespace tinyyolo
