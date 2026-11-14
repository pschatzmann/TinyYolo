#pragma once
/// @file Parallel.h  Splits a loop range across threads (host) or FreeRTOS tasks (ESP32).
///
/// Lessons from TinyTTS's TileSplitter, applied here:
///  - the extra worker runs on the other ESP32 core, not on the core the caller runs on
///    (Arduino's loop task runs on core 1 by default, so a fixed "core 1" would collide);
///  - the caller does one share of the work itself and only waits for the others;
///  - each worker writes only data that belongs to its own worker index.

#include "TinyYolo/platform/Features.h"

#if defined(TINYYOLO_HAS_THREADS)
#include <thread>
#include <vector>
#endif

#if defined(TINYYOLO_HAS_FREERTOS)
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>
#endif

namespace tinyyolo {

/// Work function: processes [begin, end) as worker number `worker` (0 = calling thread).
using RangeFn = void (*)(void* ctx, int begin, int end, int worker);

constexpr int kMaxWorkers = 4;

/// Hardware threads usable for parallel work: the host's count, 2 on ESP32, 1 otherwise.
inline int hardwareWorkers() {
#if defined(TINYYOLO_HAS_THREADS)
  const unsigned int n = std::thread::hardware_concurrency();
  int workers = n > 0 ? (int)n : 1;
  return workers > kMaxWorkers ? kMaxWorkers : workers;
#elif defined(TINYYOLO_HAS_FREERTOS)
  return 2;
#else
  return 1;
#endif
}

namespace detail {

struct RangeJob {
  RangeFn fn;
  void* ctx;
  int begin;
  int end;
  int worker;
};

inline void rangeFor(int count, int workers, int w, int& begin, int& end) {
  const int chunk = count / workers;
  const int rem = count % workers;
  begin = w * chunk + (w < rem ? w : rem);
  end = begin + chunk + (w < rem ? 1 : 0);
}

#if defined(TINYYOLO_HAS_FREERTOS)
struct FreeRtosJob {
  RangeJob job;
  SemaphoreHandle_t done;
};

inline void freeRtosEntry(void* param) {
  FreeRtosJob* j = static_cast<FreeRtosJob*>(param);
  j->job.fn(j->job.ctx, j->job.begin, j->job.end, j->job.worker);
  xSemaphoreGive(j->done);
  vTaskDelete(nullptr);
}
#endif

}  // namespace detail

/// Runs fn over [0, count) split into `workers` contiguous ranges and returns when all are done.
/// `workers` is clamped to [1, min(count, kMaxWorkers)]. With one worker, fn runs inline.
inline void parallelRange(int count, int workers, RangeFn fn, void* ctx) {
  if (count <= 0) return;
  if (workers > count) workers = count;
  if (workers > kMaxWorkers) workers = kMaxWorkers;
  if (workers <= 1) {
    fn(ctx, 0, count, 0);
    return;
  }

#if defined(TINYYOLO_HAS_THREADS)
  std::vector<std::thread> threads;
  threads.reserve((size_t)workers - 1);
  int caller_begin = 0, caller_end = 0;
  for (int w = 1; w < workers; ++w) {
    int begin, end;
    detail::rangeFor(count, workers, w, begin, end);
    threads.emplace_back([=]() { fn(ctx, begin, end, w); });
  }
  detail::rangeFor(count, workers, 0, caller_begin, caller_end);
  fn(ctx, caller_begin, caller_end, 0);
  for (auto& t : threads) t.join();

#elif defined(TINYYOLO_HAS_FREERTOS)
  detail::FreeRtosJob jobs[kMaxWorkers];
  SemaphoreHandle_t done = xSemaphoreCreateCounting(kMaxWorkers, 0);
  if (done == nullptr) {
    fn(ctx, 0, count, 0);  // no semaphore: run everything on this core
    return;
  }
  const int caller_core = xPortGetCoreID();
  bool started[kMaxWorkers] = {false, false, false, false};
  for (int w = 1; w < workers; ++w) {
    detail::FreeRtosJob& j = jobs[w];
    j.done = done;
    j.job = detail::RangeJob{fn, ctx, 0, 0, w};
    detail::rangeFor(count, workers, w, j.job.begin, j.job.end);
    // Two cores: the first worker takes the other core; any further one is left to the scheduler.
    const BaseType_t core = (w == 1) ? (1 - caller_core) : tskNO_AFFINITY;
    // Idle priority: the idle task still gets CPU time on this core, so the task watchdog
    // does not fire during long convolutions. The task needs internal RAM for its stack; when
    // that is not available the share runs on this core instead (see below).
    started[w] = xTaskCreatePinnedToCore(detail::freeRtosEntry, "tinyyolo", 8192, &j,
                                         tskIDLE_PRIORITY, nullptr, core) == pdPASS;
  }
  int begin, end;
  detail::rangeFor(count, workers, 0, begin, end);
  fn(ctx, begin, end, 0);
  for (int w = 1; w < workers; ++w) {
    if (started[w]) {
      xSemaphoreTake(done, portMAX_DELAY);
    } else {
      detail::rangeFor(count, workers, w, begin, end);
      fn(ctx, begin, end, w);
    }
  }
  vSemaphoreDelete(done);

#else
  // Unknown platform: no threads available, run everything inline.
  fn(ctx, 0, count, 0);
#endif
}

}  // namespace tinyyolo
