#pragma once
#ifndef ARM_NN_ENABLE_F32
#define ARM_NN_ENABLE_F32 0  // the float kernels are not used
#endif
// CMSIS-NN kernels are used on Arm cores with the DSP extension (Cortex-M33 of the RP2350)
// with GCC 12 or newer. Other cores use the portable reference kernels of this library.
#if !defined(CMSIS_NN) &&                                          \
    ((defined(__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1)) ||    \
     (defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1))) &&    \
    !defined(__clang__) && defined(__GNUC__) && (__GNUC__ >= 12)
#define CMSIS_NN 1
#endif
