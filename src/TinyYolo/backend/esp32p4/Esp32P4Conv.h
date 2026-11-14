#pragma once
/// @file Esp32P4Conv.h  int8 3x3 convolution with the ESP32-P4 PIE assembly kernels (no esp-dl
/// dependency). The kernels are tinyyolo_esp32p4_s8_conv2d_33cn and its _bias variant from Espressif's
/// esp-dl (MIT License, see LICENSE.esp-dl.txt in this folder).
///
/// Requirements of the kernels:
///  - input and output channels are multiples of 16; the input and output pointers are 16-byte
///    aligned;
///  - the filter is packed as (N/16)HWC16: for each block of 16 output channels, the taps (H, W),
///    then the input channels (C); inside a block the 16 output channels are the fastest index;
///  - the accumulator is an int32 sum of int8 products plus the optional bias (one int32 per
///    output channel, in the accumulator scale); it is shifted right by `mac_shift` (mac_shift =
///    out - filter - input exponent, so a positive value is a right shift), truncated (floor), and
///    saturated to int8. Linear activation only.
///
/// The kernels read their arguments from a block of 32-bit words. Only the words below are used;
/// their byte offsets are fixed by the assembly (the macros in dl_esp32p4_s8.S).

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "sdkconfig.h"
#include "TinyYolo/core/Buffer.h"
#include "TinyYolo/backend/esp32p4/ConvSchedule.h"
#include "TinyYolo/core/Status.h"

#if defined(CONFIG_IDF_TARGET_ESP32P4)
// The P4 assembly kernels, as header-only assembly (generated from the esp-dl .S sources).
#include "TinyYolo/backend/esp32p4/dl_esp32p4_common.h"
#include "TinyYolo/backend/esp32p4/dl_esp32p4_s8.h"
#include "TinyYolo/backend/esp32p4/dl_esp32p4_s8_conv2d.h"
#include "TinyYolo/backend/esp32p4/dl_esp32p4_s8_depthwise_conv2d.h"

namespace tinyyolo {
namespace esp32p4 {

extern "C" void tinyyolo_esp32p4_s8_conv2d_33cn(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_33cn_bias(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_per_channel_33cn(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_per_channel_33cn_bias(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_depthwise_conv2d_33c1_bias(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_11cn(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_11cn_bias(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_per_channel_11cn(int8_t* output_ptr, int8_t* input_ptr, void* args);
extern "C" void tinyyolo_esp32p4_s8_conv2d_per_channel_11cn_bias(int8_t* output_ptr, int8_t* input_ptr, void* args);

/// Pointer into `buf` rounded up to 16 bytes (the buffer has 16 bytes to spare).
template <typename T>
inline T* alignTo16(Buffer<int8_t>& buf) {
  uint8_t* p = (uint8_t*)buf.data();
  return (T*)(p + ((16 - ((uintptr_t)p & 15)) & 15));
}

/// Runs the kernel that matches the arguments over an OH x OW output (see ConvSchedule.h for the
/// work split and the edge windows). The kernel is chosen here: per-channel or one shift, with or
/// without bias.
inline Status runBlocks(bool three_by_three, int C, int OC, int H, int W, int pad, int stride, int OH, int OW,
                        const int8_t* src, const int8_t* packed_filter, int mac_shift, const int32_t* bias,
                        const int8_t* channel_shifts, int8_t* output, int workers) {
  ConvKernel kernel;
  if (three_by_three) {
    kernel = channel_shifts != nullptr
                 ? (bias != nullptr ? tinyyolo_esp32p4_s8_conv2d_per_channel_33cn_bias : tinyyolo_esp32p4_s8_conv2d_per_channel_33cn)
                 : (bias != nullptr ? tinyyolo_esp32p4_s8_conv2d_33cn_bias : tinyyolo_esp32p4_s8_conv2d_33cn);
  } else {
    kernel = channel_shifts != nullptr
                 ? (bias != nullptr ? tinyyolo_esp32p4_s8_conv2d_per_channel_11cn_bias : tinyyolo_esp32p4_s8_conv2d_per_channel_11cn)
                 : (bias != nullptr ? tinyyolo_esp32p4_s8_conv2d_11cn_bias : tinyyolo_esp32p4_s8_conv2d_11cn);
  }
  return runConvBlocks(kernel, three_by_three, C, OC, H, W, pad, stride, OH, OW, src, packed_filter, mac_shift, bias,
                       channel_shifts, output, workers);
}

/// 3x3 convolution of `input` (H x W x C, int8, HWC, 16-byte aligned) with the packed filter into
/// `output` (OH x OW x OC), where OH = (H + 2 pad - 3) / stride + 1 (the same for OW). Zero padding
/// `pad` (0 or 1) is applied by the schedule at the picture edges: the picture is not copied.
/// `bias` (OC int32 values, or nullptr) is added to the accumulator. `channel_shifts` (OC int8 values,
/// 16-byte aligned) replaces `mac_shift` with one right shift per output channel, for weights with a
/// scale per channel. Returns false when the shape or the alignment does not fit.
inline Status conv3x3(const int8_t* input, int H, int W, int C, int stride, int pad, const int8_t* packed_filter,
                      int OC, int mac_shift, const int32_t* bias, int8_t* output,
                      const int8_t* channel_shifts = nullptr, int workers = 1) {
  // The kernels read 16 input channels at a time: C must be a multiple of 16 (the caller pads).
  if (C % 16 != 0) return Status::Unsupported;
  if (OC % 16 != 0 || stride < 1 || pad < 0 || pad > 1 || H + 2 * pad < 3 || W + 2 * pad < 3) {
    return Status::Unsupported;
  }
  if (((uintptr_t)packed_filter & 15) || ((uintptr_t)output & 15) || ((uintptr_t)input & 15)) {
    return Status::InvalidArgument;
  }
  const int OH = (H + 2 * pad - 3) / stride + 1, OW = (W + 2 * pad - 3) / stride + 1;
  return runBlocks(true, C, OC, H, W, pad, stride, OH, OW, input, packed_filter, mac_shift, bias, channel_shifts,
                   output, workers);
}

/// 1x1 convolution (no padding, stride `stride`) of `input` (H x W x C) into `output`
/// (OH x OW x OC), OH = (H - 1) / stride + 1. The same packing and rules as conv3x3.
inline Status conv1x1(const int8_t* input, int H, int W, int C, int stride, const int8_t* packed_filter, int OC,
                      int mac_shift, const int32_t* bias, int8_t* output, const int8_t* channel_shifts = nullptr,
                      int workers = 1) {
  if (C % 16 != 0 || OC % 16 != 0 || stride < 1 || H < 1 || W < 1) return Status::Unsupported;
  if (((uintptr_t)input & 15) || ((uintptr_t)output & 15) || ((uintptr_t)packed_filter & 15)) {
    return Status::InvalidArgument;
  }
  if (channel_shifts != nullptr && ((uintptr_t)channel_shifts & 15)) return Status::InvalidArgument;
  const int OH = (H - 1) / stride + 1, OW = (W - 1) / stride + 1;
  return runBlocks(false, C, OC, H, W, 0, stride, OH, OW, input, packed_filter, mac_shift, bias, channel_shifts,
                   output, workers);
}

/// Depthwise 3x3 convolution, stride 1, padding 1, with bias (int32 per channel) and one right shift
/// for all channels. `packed_weights` is [C/16][9][16] (see depthwise3x3Pack); `input` and `output`
/// are H x W x C (C a multiple of 16). The edge windows are gathered by the schedule; the picture is
/// not copied. Returns Unsupported for other shapes.
inline Status depthwise3x3(const int8_t* input, int H, int W, int C, const int8_t* packed_weights, int mac_shift,
                           const int32_t* bias, int8_t* output, int workers = 1) {
  if (C % 16 != 0 || H < 1 || W < 1) return Status::Unsupported;
  if (((uintptr_t)packed_weights & 15) || ((uintptr_t)output & 15) || ((uintptr_t)bias & 15) ||
      ((uintptr_t)input & 15)) {
    return Status::InvalidArgument;
  }
  return runDepthwiseBlocks(tinyyolo_esp32p4_s8_depthwise_conv2d_33c1_bias, C, H, W, input, packed_weights, mac_shift,
                            bias, output, workers);
}

/// Packs depthwise weights [C][9] (one row of 9 taps per channel) into [C/16][9][16].
inline void depthwise3x3Pack(const int8_t* weights, int C, int8_t* packed) {
  for (int c = 0; c < C; ++c) {
    for (int t = 0; t < 9; ++t) packed[((c / 16) * 9 + t) * 16 + c % 16] = weights[c * 9 + t];
  }
}

}  // namespace esp32p4
}  // namespace tinyyolo

#endif  // CONFIG_IDF_TARGET_ESP32P4
