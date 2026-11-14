#pragma once
/// @file Memory.h  Placement of the library's buffers: internal RAM first, PSRAM as fallback.
///
/// On ESP32 builds with heap_caps, every buffer is first requested from internal RAM. When
/// that fails (large activations, or a full heap), the buffer goes to external PSRAM if the
/// board has it. Placement can be reversed with setMemoryPlacement(PsramFirst) for speed
/// tests, and PSRAM can be disabled altogether with setPsramAllocation(false).

#include <stdlib.h>
#include <string.h>

#include "TinyYolo/platform/Features.h"

#if defined(TINYYOLO_HAS_HEAP_CAPS)
#include <esp_heap_caps.h>
#endif
#if defined(TINYYOLO_HAS_RP_PSRAM)
#include "psram.h"
#endif

namespace tinyyolo {

enum class MemoryPlacement : uint8_t {
  InternalFirst,  ///< internal RAM, then PSRAM when internal RAM cannot hold the buffer (default)
  PsramFirst      ///< PSRAM, then internal RAM (for comparing speed)
};

inline MemoryPlacement& placementRef() {
  static MemoryPlacement placement = MemoryPlacement::InternalFirst;
  return placement;
}
inline void setMemoryPlacement(MemoryPlacement placement) { placementRef() = placement; }
inline MemoryPlacement memoryPlacement() { return placementRef(); }

inline bool& psramAllocationRef() {
  static bool enabled = true;
  return enabled;
}

/// Allows (true, default) or forbids (false) placing buffers in PSRAM.
inline void setPsramAllocation(bool enabled) { psramAllocationRef() = enabled; }
inline bool psramAllocation() { return psramAllocationRef(); }

/// Allocates `bytes` (zeroed when `zeroed`). `in_psram` reports where the memory came from;
/// pass it back to releaseMemory(). Returns nullptr when no placement can hold the request.
inline void* allocateMemory(size_t bytes, bool zeroed, bool& in_psram, MemoryPlacement first) {
  in_psram = false;
#if defined(TINYYOLO_HAS_HEAP_CAPS)
  const bool psram_first = first == MemoryPlacement::PsramFirst;
  for (int attempt = 0; attempt < 2; ++attempt) {
    const bool try_psram = (attempt == 0) == psram_first;
    if (try_psram && !psramAllocation()) continue;
    const uint32_t caps = try_psram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL;
    void* p = zeroed ? heap_caps_calloc(1, bytes, MALLOC_CAP_8BIT | caps)
                     : heap_caps_malloc(bytes, MALLOC_CAP_8BIT | caps);
    if (p != nullptr) {
      in_psram = try_psram;
      return p;
    }
  }
  return nullptr;
#elif defined(TINYYOLO_HAS_RP_PSRAM)
  // RP2040/RP2350 with PSRAM: internal RAM (malloc) and PSRAM (__psram_malloc), in the same order.
  const bool psram_first = first == MemoryPlacement::PsramFirst;
  for (int attempt = 0; attempt < 2; ++attempt) {
    const bool try_psram = (attempt == 0) == psram_first;
    if (try_psram && !psramAllocation()) continue;
    void* p = try_psram ? __psram_malloc(bytes) : malloc(bytes);
    if (p != nullptr) {
      if (zeroed) memset(p, 0, bytes);
      in_psram = try_psram;
      return p;
    }
  }
  return nullptr;
#else
  (void)first;  // one memory region on hosts
  return zeroed ? calloc(1, bytes) : malloc(bytes);
#endif
}

/// Allocates with the global placement (see setMemoryPlacement).
inline void* allocateMemory(size_t bytes, bool zeroed, bool& in_psram) {
  return allocateMemory(bytes, zeroed, in_psram, memoryPlacement());
}

/// Returns zeroed memory (see allocateMemory).
inline void* allocateZeroed(size_t bytes, bool& in_psram) {
  return allocateMemory(bytes, true, in_psram);
}

inline void releaseMemory(void* p, bool in_psram) {
#if defined(TINYYOLO_HAS_HEAP_CAPS)
  // heap_caps_free() releases memory from either region; free() does the same for internal RAM.
  if (in_psram) {
    heap_caps_free(p);
    return;
  }
#elif defined(TINYYOLO_HAS_RP_PSRAM)
  if (in_psram) {
    __psram_free(p);
    return;
  }
#endif
  (void)in_psram;
  free(p);
}

}  // namespace tinyyolo
