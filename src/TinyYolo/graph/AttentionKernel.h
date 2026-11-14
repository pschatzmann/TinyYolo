#pragma once
/// @file AttentionKernel.h  The score and value products of the attention in Graph (C2PSA).
///
/// The work is split into blocks of 4 queries of one head. Each block is independent: it reads the
/// shared per-head arrays and writes its own positions of the output, so the blocks are spread over
/// the workers (Parallel.h). Each worker has its own scores buffer (4 * N floats), so the sums of
/// every query are the same as on one thread.

#include <math.h>
#include <stddef.h>

#include "TinyYolo/platform/Parallel.h"

namespace tinyyolo {

/// Shared inputs and outputs of the attention blocks.
struct AttentionJob {
  const float* q;   ///< [head][position][kd], scaled later by `scale`
  const float* k;   ///< [head][position][kd]
  const float* v;   ///< [head][position][hd]
  float* scores;    ///< workers * 4 * N floats: one scores buffer per worker
  float* y;         ///< position-major output with C = heads * hd channels per position
  int N;            ///< positions
  int heads, kd, hd, C;
  float scale;
  int blocks_per_head;  ///< ceil(N / 4)
};

/// Work body for parallelRange: items [begin, end) are (head, block of 4 queries) pairs, numbered
/// head-major. `worker` selects the scores buffer.
inline void attentionBlocks(void* ctx, int begin, int end, int worker) {
  const AttentionJob& j = *static_cast<const AttentionJob*>(ctx);
  const int N = j.N, kd = j.kd, hd = j.hd, C = j.C;
  float* sp = j.scores + (size_t)worker * 4 * (size_t)N;
  for (int item = begin; item < end; ++item) {
    const int h = item / j.blocks_per_head;
    const int nq0 = (item % j.blocks_per_head) * 4;
    const int nb = N - nq0 < 4 ? N - nq0 : 4;
    float max_s[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
    // Four queries at a time: each key row is loaded once for all four (the sums per query are
    // the same as one query at a time).
    for (int nk = 0; nk < N; ++nk) {
      const float* k = j.k + ((size_t)h * N + nk) * kd;
      for (int b = 0; b < nb; ++b) {
        const float* q = j.q + ((size_t)h * N + nq0 + b) * kd;
        // Four partial sums (kd is a multiple of 4 for the attention blocks of YOLO11).
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        int d = 0;
        for (; d + 3 < kd; d += 4) {
          d0 += q[d] * k[d];
          d1 += q[d + 1] * k[d + 1];
          d2 += q[d + 2] * k[d + 2];
          d3 += q[d + 3] * k[d + 3];
        }
        for (; d < kd; ++d) d0 += q[d] * k[d];
        float* sb = sp + (size_t)b * N;
        sb[nk] = ((d0 + d1) + (d2 + d3)) * j.scale;
        if (sb[nk] > max_s[b]) max_s[b] = sb[nk];
      }
    }
    for (int b = 0; b < nb; ++b) {
      const int nq = nq0 + b;
      float* spb = sp + (size_t)b * N;
      // Softmax over the positions.
      float sum = 0.0f;
      for (int nk = 0; nk < N; ++nk) {
        spb[nk] = expf(spb[nk] - max_s[b]);
        sum += spb[nk];
      }
      // Weighted sum of the values (the channels of one position are contiguous in y).
      float* out = j.y + (size_t)nq * C + h * hd;
      const float inv = 1.0f / sum;
      // Blocks of 8 output channels in local accumulators (the sums are in the same order as before).
      int d0 = 0;
      for (; d0 + 8 <= hd; d0 += 8) {
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (int nk = 0; nk < N; ++nk) {
          const float s = spb[nk];
          const float* v = j.v + ((size_t)h * N + nk) * hd + d0;
          a0 += s * v[0]; a1 += s * v[1]; a2 += s * v[2]; a3 += s * v[3];
          a4 += s * v[4]; a5 += s * v[5]; a6 += s * v[6]; a7 += s * v[7];
        }
        out[d0] = a0 * inv; out[d0 + 1] = a1 * inv; out[d0 + 2] = a2 * inv; out[d0 + 3] = a3 * inv;
        out[d0 + 4] = a4 * inv; out[d0 + 5] = a5 * inv; out[d0 + 6] = a6 * inv; out[d0 + 7] = a7 * inv;
      }
      for (; d0 < hd; ++d0) {
        float a = 0;
        for (int nk = 0; nk < N; ++nk) a += spb[nk] * j.v[((size_t)h * N + nk) * hd + d0];
        out[d0] = a * inv;
      }
    }
  }
}

/// Number of worker scores buffers needed for `workers` threads (parallelRange clamps the same way).
inline int attentionWorkers(int workers, int items) {
  if (workers < 1) workers = 1;
  if (workers > kMaxWorkers) workers = kMaxWorkers;
  if (workers > items) workers = items;
  return workers < 1 ? 1 : workers;
}

}  // namespace tinyyolo
