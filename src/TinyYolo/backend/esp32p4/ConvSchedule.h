#pragma once
/// @file ConvSchedule.h  Work split and tap addressing of the int8 convolutions that use the ESP32-P4
/// kernels (Esp32P4Conv.h). Platform-independent: the kernel is a parameter, so the schedule is
/// tested on the desktop with a reference kernel.
///
/// The input is the unpadded picture. A 3x3 window that lies inside the picture (with one pixel of
/// margin on the last row and column, because the kernels may read up to 16 bytes past the last
/// tap) is read in place. A window that reaches outside the picture is gathered into a small
/// buffer with zeros outside, so no padded copy of the picture is made.
///
/// The output rows are split over the workers (Parallel.h, one of them on the second core). Each
/// worker writes only its own rows. The packed filter is read-only while the workers run. A block of
/// 16 output channels is read from the staged whole filter when the whole filter fits, otherwise it
/// is copied into a buffer that belongs to the worker.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "TinyYolo/core/Status.h"
#include "TinyYolo/platform/Parallel.h"

namespace tinyyolo {
namespace esp32p4 {

/// Taps of a 3x3 kernel.
static constexpr int K3 = 9;
/// Largest input channel count of the kernels (the filter block is kept in a fixed buffer).
static constexpr int kMaxChannels = 256;
/// Bytes of the packed filter that are kept in internal RAM for the whole layer (pixels outside).
static constexpr int kWholeFilterBytes = 16384;
/// The kernels may read this many bytes past the end of a buffer: the buffers have this spare space.
static constexpr int kFilterTail = 64;
/// Staging buffers, one set per worker. parallelRange never uses more workers than this.
static constexpr int kMaxConvWorkers = 2;

/// Kernel signature: one output pixel, 16 output channels (see Esp32P4Conv.h).
typedef void (*ConvKernel)(int8_t* output_ptr, int8_t* input_ptr, void* args);

/// The argument block of the kernels. Word i is at byte offset 4 * i.
struct ConvArgs {
  static constexpr int kWordCount = 44;
  static constexpr int kFilter = 12;          ///< byte 48: packed filter pointer
  static constexpr int kMacShift = 16;        ///< byte 64: right shift of the accumulator
  static constexpr int kBias = 17;            ///< byte 68: int32 bias per output channel
  static constexpr int kOutputGroups = 24;    ///< byte 96: output channels / 16
  static constexpr int kInputGroupsM1 = 25;   ///< byte 100: input channels / 16 - 1
  static constexpr int kChannelShifts = 26;   ///< byte 104: int8 right shift per output channel
  static constexpr int kDilationX = 27;       ///< byte 108: convolution: 0 (taps are contiguous in x)
  static constexpr int kDilationY = 28;       ///< byte 112: convolution: input step after the 3 taps of a row
  // Depthwise kernel only:
  static constexpr int kDepthDilationX = 31;  ///< byte 124: channel step between taps in x (= C)
  static constexpr int kDepthDilationY = 32;  ///< byte 128: step from the last tap of a row to the next row
  static constexpr int kDepthNextGroup = 33;  ///< byte 132: step from the last tap to the next channel group

  uint32_t words[kWordCount] = {};
  /// Full-width copies of the pointers in the words above. The kernels read only the words (they
  /// are 32-bit); the host stand-in reads these, because a 64-bit host cannot keep a pointer in a word.
  const void* filter_ptr = nullptr;
  const void* bias_ptr = nullptr;
  const void* shift_ptr = nullptr;  ///< per-channel shifts, when used
};

/// The whole packed filter, staged once per call (shared read-only by the workers).
inline int8_t* wholeFilterBuffer() {
  static int8_t buf[kWholeFilterBytes + kFilterTail] __attribute__((aligned(16)));
  return buf;
}

/// The filter block of one worker.
inline int8_t* blockFilterBuffer(int worker) {
  static int8_t buf[kMaxConvWorkers][K3 * kMaxChannels * 16 + kFilterTail] __attribute__((aligned(16)));
  return buf[worker];
}

/// The 3x3 window of one worker when it reaches outside the picture: taps in rows of 3C bytes.
inline int8_t* windowBuffer(int worker) {
  static int8_t buf[kMaxConvWorkers][K3 * kMaxChannels + kFilterTail] __attribute__((aligned(16)));
  return buf[worker];
}

/// Address of the 3x3 window whose top-left tap is (top, left) in the unpadded picture (H x W x C).
/// Returns the picture itself when the window is inside it (`interior` set), otherwise a gathered
/// copy with zeros outside the picture (`interior` cleared). The tap steps depend on the case; the
/// caller sets them.
inline const int8_t* gatherWindow(const int8_t* src, int H, int W, int C, int top, int left, int worker,
                                  bool& interior) {
  interior = top >= 0 && left >= 0 && top + 2 <= H - 2 && left + 2 <= W - 2;
  if (interior) return src + ((size_t)top * W + (size_t)left) * C;
  int8_t* win = windowBuffer(worker);
  for (int ty = 0; ty < 3; ++ty) {
    for (int tx = 0; tx < 3; ++tx) {
      const int iy = top + ty, ix = left + tx;
      int8_t* dst = win + (size_t)(ty * 3 + tx) * C;
      if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
        memcpy(dst, src + ((size_t)iy * W + (size_t)ix) * C, (size_t)C);
      } else {
        memset(dst, 0, (size_t)C);
      }
    }
  }
  return win;
}

inline void setGroupFilter(ConvArgs& args, int g, const int8_t* filter, const int32_t* bias,
                           const int8_t* channel_shifts) {
  args.words[ConvArgs::kFilter] = (uint32_t)(uintptr_t)filter;
  args.filter_ptr = filter;
  if (bias != nullptr) {
    args.words[ConvArgs::kBias] = (uint32_t)(uintptr_t)(bias + 16 * g);
    args.bias_ptr = bias + 16 * g;
  }
  if (channel_shifts != nullptr) {
    args.words[ConvArgs::kChannelShifts] = (uint32_t)(uintptr_t)(channel_shifts + 16 * g);
    args.shift_ptr = channel_shifts + 16 * g;
    args.words[ConvArgs::kMacShift] = 0x80000000u;  // INT_MIN: the per-channel kernels use the array
  }
}

/// What the workers need of a convolution, shared by all of them.
struct ConvBlockJob {
  ConvKernel kernel;
  bool whole;           ///< the filter is staged in wholeFilterBuffer()
  bool three_by_three;  ///< 3x3 windows (with the edge gather) or 1x1 pixels
  int groups;           ///< output channel blocks of 16
  int C, OC, H, W, pad, stride, OW;
  size_t block_bytes;   ///< taps * C * 16
  const int8_t* src;    ///< the unpadded input (H x W x C)
  const int8_t* packed_filter;
  int8_t* output;
  int mac_shift;
  const int32_t* bias;
  const int8_t* channel_shifts;
};

/// Worker body: output rows [begin, end). `worker` selects the worker's own buffers.
inline void convBlockRows(void* ctx, int begin, int end, int worker) {
  const ConvBlockJob& j = *static_cast<const ConvBlockJob*>(ctx);
  ConvArgs args;
  args.words[ConvArgs::kMacShift] = (uint32_t)j.mac_shift;
  args.words[ConvArgs::kOutputGroups] = 1;  // one block of 16 output channels per call
  args.words[ConvArgs::kInputGroupsM1] = (uint32_t)(j.C / 16 - 1);
  args.words[ConvArgs::kDilationX] = 0;

  // The input of output pixel (oy, ox) and its tap step in y.
  auto pixelInput = [&](int oy, int ox, bool& interior) -> int8_t* {
    if (!j.three_by_three) {
      interior = true;
      args.words[ConvArgs::kDilationY] = 0;
      return (int8_t*)(j.src + ((size_t)(oy * j.stride) * j.W + (size_t)(ox * j.stride)) * j.C);
    }
    const int top = oy * j.stride - j.pad, left = ox * j.stride - j.pad;
    const int8_t* in = gatherWindow(j.src, j.H, j.W, j.C, top, left, worker, interior);
    // Interior: the rest of the picture's row after the 3 taps. Gathered: the rows are 3C apart.
    args.words[ConvArgs::kDilationY] = interior ? (uint32_t)(j.W * j.C - 3 * j.C) : 0;
    return (int8_t*)in;
  };

  if (j.whole) {
    const int8_t* whole = wholeFilterBuffer();
    for (int oy = begin; oy < end; ++oy) {
      for (int ox = 0; ox < j.OW; ++ox) {
        bool interior;
        int8_t* in = pixelInput(oy, ox, interior);
        for (int g = 0; g < j.groups; ++g) {
          setGroupFilter(args, g, whole + (size_t)g * j.block_bytes, j.bias, j.channel_shifts);
          j.kernel(j.output + ((size_t)oy * j.OW + ox) * j.OC + 16 * g, in, &args);
        }
      }
    }
  } else {
    int8_t* block = blockFilterBuffer(worker);
    for (int g = 0; g < j.groups; ++g) {
      memcpy(block, j.packed_filter + (size_t)g * j.block_bytes, j.block_bytes);
      memset(block + j.block_bytes, 0, kFilterTail);
      setGroupFilter(args, g, block, j.bias, j.channel_shifts);
      for (int oy = begin; oy < end; ++oy) {
        for (int ox = 0; ox < j.OW; ++ox) {
          bool interior;
          int8_t* in = pixelInput(oy, ox, interior);
          j.kernel(j.output + ((size_t)oy * j.OW + ox) * j.OC + 16 * g, in, &args);
        }
      }
    }
  }
}

/// Runs `kernel` over an OH x OW output with `workers` threads (clamped to kMaxConvWorkers).
/// `src` is the unpadded input (H x W x C, 16-byte aligned); `pad` is 0 or 1 (1 for 3x3 with padding
/// one). The filter is packed as (N/16)HWC16 (see Esp32P4Conv.h). When the whole filter fits in
/// internal RAM, the pixels are the outer loop (the filter is read once); otherwise the blocks are the
/// outer loop and each block's filter is copied to the worker's buffer once per worker.
inline Status runConvBlocks(ConvKernel kernel, bool three_by_three, int C, int OC, int H, int W, int pad,
                            int stride, int OH, int OW, const int8_t* src, const int8_t* packed_filter,
                            int mac_shift, const int32_t* bias, const int8_t* channel_shifts, int8_t* output,
                            int workers) {
  const int taps = three_by_three ? K3 : 1;
  if (C > kMaxChannels || C % 16 != 0 || OC % 16 != 0 || stride < 1) return Status::Unsupported;
  if (pad < 0 || pad > 1 || (!three_by_three && pad != 0)) return Status::Unsupported;
  const size_t block_bytes = (size_t)taps * C * 16;
  const size_t whole_bytes = (size_t)taps * C * OC;
  const bool whole = whole_bytes <= (size_t)kWholeFilterBytes;
  if (whole) {
    int8_t* staged = wholeFilterBuffer();
    memcpy(staged, packed_filter, whole_bytes);
    memset(staged + whole_bytes, 0, kFilterTail);
  }
  ConvBlockJob job;
  job.kernel = kernel;
  job.whole = whole;
  job.three_by_three = three_by_three;
  job.groups = OC / 16;
  job.C = C;
  job.OC = OC;
  job.H = H;
  job.W = W;
  job.pad = pad;
  job.stride = stride;
  job.OW = OW;
  job.block_bytes = block_bytes;
  job.src = src;
  job.packed_filter = packed_filter;
  job.output = output;
  job.mac_shift = mac_shift;
  job.bias = bias;
  job.channel_shifts = channel_shifts;
  if (workers < 1) workers = 1;
  if (workers > kMaxConvWorkers) workers = kMaxConvWorkers;
  parallelRange(OH, workers, &convBlockRows, &job);
  return Status::Ok;
}

/// Depthwise 3x3 (stride 1, padding 1) job: one kernel call per pixel covers all channels.
struct DepthwiseJob {
  ConvKernel kernel;
  int C, H, W;
  const int8_t* src;     ///< the unpadded input (H x W x C)
  const int8_t* packed;  ///< [C/16][9][16] (depthwise3x3Pack)
  int8_t* output;
  int mac_shift;
  const int32_t* bias;   ///< int32 per channel, 16-byte aligned
};

inline void depthwiseRows(void* ctx, int begin, int end, int worker) {
  const DepthwiseJob& j = *static_cast<const DepthwiseJob*>(ctx);
  ConvArgs args;
  args.words[ConvArgs::kFilter] = (uint32_t)(uintptr_t)j.packed;
  args.filter_ptr = j.packed;
  args.words[ConvArgs::kMacShift] = (uint32_t)j.mac_shift;
  args.words[ConvArgs::kBias] = (uint32_t)(uintptr_t)j.bias;
  args.bias_ptr = j.bias;
  args.words[ConvArgs::kInputGroupsM1] = (uint32_t)(j.C / 16 - 1);
  const int C = j.C;
  for (int oy = begin; oy < end; ++oy) {
    for (int ox = 0; ox < j.W; ++ox) {
      bool interior;
      const int8_t* in = gatherWindow(j.src, j.H, j.W, C, oy - 1, ox - 1, worker, interior);
      // The taps of a group: x steps one channel block (C bytes); the kernel moves from the last tap
      // of a group to the next channel group by the step in words[kDepthNextGroup].
      if (interior) {
        args.words[ConvArgs::kDepthDilationX] = (uint32_t)C;
        args.words[ConvArgs::kDepthDilationY] = (uint32_t)(j.W * C - 2 * C);
        args.words[ConvArgs::kDepthNextGroup] = (uint32_t)(16 - 2 * C - 2 * j.W * C);
      } else {
        args.words[ConvArgs::kDepthDilationX] = (uint32_t)C;
        args.words[ConvArgs::kDepthDilationY] = (uint32_t)C;  // rows are 3C apart: 3C - 2C
        args.words[ConvArgs::kDepthNextGroup] = (uint32_t)(16 - 8 * C);
      }
      j.kernel(j.output + ((size_t)oy * j.W + ox) * C, (int8_t*)in, &args);
    }
  }
}

/// Depthwise 3x3 over an H x W x C picture with `workers` threads (rows are split, as for conv).
inline Status runDepthwiseBlocks(ConvKernel kernel, int C, int H, int W, const int8_t* src,
                                 const int8_t* packed, int mac_shift, const int32_t* bias, int8_t* output,
                                 int workers) {
  if (C % 16 != 0 || C > kMaxChannels || H < 1 || W < 1) return Status::Unsupported;
  DepthwiseJob job = {kernel, C, H, W, src, packed, output, mac_shift, bias};
  if (workers < 1) workers = 1;
  if (workers > kMaxConvWorkers) workers = kMaxConvWorkers;
  parallelRange(H, workers, &depthwiseRows, &job);
  return Status::Ok;
}

}  // namespace esp32p4
}  // namespace tinyyolo
