#pragma once
/// @file Logger.h
/// Level-filtered logger that writes to any Arduino Print target.

#include <stdarg.h>
#include <stdio.h>

#include "TinyYolo/core/Config.h"
#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

enum class TinyYoloLogLevel : uint8_t { Debug = 0, Info = 1, Warning = 2, Error = 3, Off = 4 };

/// Process-wide logger. Access it via Logger::instance() or TinyYoloLogger.
/// Nothing is printed until begin() has been called with an output.
class Logger {
 public:
  /// Returns the single logger object.
  static Logger& instance() {
    static Logger logger;
    return logger;
  }

  /// Sends log lines to `output` (e.g. Serial) and sets the minimum level.
  void begin(Print& output, TinyYoloLogLevel level = TinyYoloLogLevel::Info) {
    output_ = &output;
    level_ = level;
  }

  /// Stops all output.
  void end() { output_ = nullptr; }

  void setLevel(TinyYoloLogLevel level) { level_ = level; }
  TinyYoloLogLevel level() const { return level_; }

  bool isEnabled(TinyYoloLogLevel level) const {
    return output_ != nullptr && level_ != TinyYoloLogLevel::Off && level >= level_;
  }

  /// Formats and writes one line: "[E] module: message".
  TINYYOLO_PRINTF_CHECK(4, 5)
  void log(TinyYoloLogLevel level, const char* module, const char* format, ...) {
    if (!isEnabled(level)) return;
    char message[TINYYOLO_LOG_BUFFER_SIZE];
    va_list args;
    va_start(args, format);
    vsnprintf(message, sizeof(message), format, args);
    va_end(args);
    output_->print("[");
    output_->print(tag(level));
    output_->print("] ");
    output_->print(module);
    output_->print(": ");
    output_->println(message);
    output_->flush();
  }

 private:
  Logger() {}
  Logger(const Logger&) = delete;
  Logger& operator=(const Logger&) = delete;

  static const char* tag(TinyYoloLogLevel level) {
    switch (level) {
      case TinyYoloLogLevel::Debug: return "D";
      case TinyYoloLogLevel::Info: return "I";
      case TinyYoloLogLevel::Warning: return "W";
      case TinyYoloLogLevel::Error: return "E";
      default: return "?";
    }
  }

  Print* output_ = nullptr;
  TinyYoloLogLevel level_ = TinyYoloLogLevel::Info;
};

/// The logger, for setup code: TinyYoloLogger.begin(Serial, TinyYoloLogLevel::Info). Every translation unit
/// refers to the same object (Logger::instance()). The attribute keeps a file that does not
/// use it from warning.
static Logger& TinyYoloLogger __attribute__((unused)) = Logger::instance();

/// True when a level is not removed at compile time (TINYYOLO_MIN_LOG_LEVEL).
inline bool logLevelCompiledIn(int level) { return level >= TINYYOLO_MIN_LOG_LEVEL; }

}  // namespace tinyyolo

#define TINYYOLO_LOG(level, module, ...)                                                  \
  do {                                                                                    \
    if (::tinyyolo::logLevelCompiledIn(static_cast<int>(level))) {                        \
      ::tinyyolo::Logger::instance().log((level), (module), __VA_ARGS__);                 \
    }                                                                                     \
  } while (0)

#define TINYYOLO_LOGD(module, ...) TINYYOLO_LOG(::tinyyolo::TinyYoloLogLevel::Debug, module, __VA_ARGS__)
#define TINYYOLO_LOGI(module, ...) TINYYOLO_LOG(::tinyyolo::TinyYoloLogLevel::Info, module, __VA_ARGS__)
#define TINYYOLO_LOGW(module, ...) TINYYOLO_LOG(::tinyyolo::TinyYoloLogLevel::Warning, module, __VA_ARGS__)
#define TINYYOLO_LOGE(module, ...) TINYYOLO_LOG(::tinyyolo::TinyYoloLogLevel::Error, module, __VA_ARGS__)
