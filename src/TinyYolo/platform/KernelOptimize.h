#pragma once
/// @file KernelOptimize.h  Compiles the compute kernels with -O2 on the ESP32.
///
/// The Arduino ESP32 core builds sketches with -Os. The convolution and quantization loops
/// are the time-critical code, and -O2 is measurably faster for them (about 3-4% on the
/// whole YOLOv4-tiny frame, see docs/Quantization.md). Kernel headers include this file, so
/// the pragma applies to the functions that follow it in those headers. Other compilers and
/// host builds are not affected.

#if defined(ESP32) && defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize("O2")
#endif
