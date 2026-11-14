#pragma once
/// @file Task.h  A background task: runs one function on its own thread (host) or
/// FreeRTOS task (ESP32).
///
/// Usage:
///   Task task;                       // optional: stack size, priority, core
///   task.begin([]() { work(); });    // starts running; returns false if already running
///   ...
///   task.end();                      // waits for the function to finish, then releases
///
/// A Task runs its function once per begin(). end() blocks until that run is complete.

#include <atomic>
#include <functional>
#include <utility>

#include "TinyYolo/platform/Features.h"

#if defined(TINYYOLO_HAS_THREADS)
#include <thread>
#endif

#if defined(TINYYOLO_HAS_FREERTOS)
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#endif

namespace tinyyolo {

class Task {
 public:
  /// stack_bytes: stack size (bytes on ESP32; ignored on hosts). core: ESP32 core, -1 = any.
  explicit Task(int stack_bytes = 8192, int priority = 1, int core = -1)
      : stack_bytes_(stack_bytes), priority_(priority), core_(core) {}
  ~Task() { end(); }
  Task(const Task&) = delete;
  Task& operator=(const Task&) = delete;

  /// Starts `fn` on a new thread. Returns false if the task is already running or the
  /// thread could not be created.
  bool begin(std::function<void()> fn) {
    if (isRunning()) return false;
    end();  // release a previous run, if any
    fn_ = std::move(fn);
    state_.store(kRunning, std::memory_order_release);
#if defined(TINYYOLO_HAS_THREADS)
    thread_ = std::thread([this]() {
      fn_();
      state_.store(kEnded, std::memory_order_release);
    });
    return true;
#elif defined(TINYYOLO_HAS_FREERTOS)
    done_ = xSemaphoreCreateBinary();
    if (done_ == nullptr) {
      state_.store(kIdle, std::memory_order_release);
      return false;
    }
    BaseType_t ok = xTaskCreatePinnedToCore(&Task::freeRtosEntry, "tinyyolo", stack_bytes_, this,
                                            priority_, &handle_, core_ < 0 ? tskNO_AFFINITY : core_);
    if (ok != pdPASS) {
      vSemaphoreDelete(done_);
      done_ = nullptr;
      state_.store(kIdle, std::memory_order_release);
      return false;
    }
    return true;
#else
    // No threads on this platform: run synchronously so callers still get the result.
    fn_();
    state_.store(kEnded, std::memory_order_release);
    return true;
#endif
  }

  /// Waits until the function has finished, then releases the thread. Safe to call when idle.
  void end() {
#if defined(TINYYOLO_HAS_THREADS)
    if (thread_.joinable()) thread_.join();
#elif defined(TINYYOLO_HAS_FREERTOS)
    if (done_ != nullptr) {
      xSemaphoreTake(done_, portMAX_DELAY);
      vSemaphoreDelete(done_);
      done_ = nullptr;
      handle_ = nullptr;
    }
#endif
  }

  /// True from begin() until the function returns.
  bool isRunning() const { return state_.load(std::memory_order_acquire) == kRunning; }
  /// True after the function has finished, until the next begin().
  bool isEnded() const { return state_.load(std::memory_order_acquire) == kEnded; }

 private:
  enum : int { kIdle = 0, kRunning = 1, kEnded = 2 };

#if defined(TINYYOLO_HAS_FREERTOS)
  static void freeRtosEntry(void* param) {
    Task* self = static_cast<Task*>(param);
    self->fn_();
    self->state_.store(kEnded, std::memory_order_release);
    xSemaphoreGive(self->done_);  // end() waits on this; the task deletes itself after it
    vTaskDelete(nullptr);
  }
#endif

  std::function<void()> fn_;
  std::atomic<int> state_{kIdle};
  int stack_bytes_;
  int priority_;
  int core_;
#if defined(TINYYOLO_HAS_THREADS)
  std::thread thread_;
#elif defined(TINYYOLO_HAS_FREERTOS)
  TaskHandle_t handle_ = nullptr;
  SemaphoreHandle_t done_ = nullptr;
#endif
};

}  // namespace tinyyolo
