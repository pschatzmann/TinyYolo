/// make_darknet_weights: writes a Darknet .weights file that matches a .cfg.
///
/// The values are deterministic pseudo-random numbers, not a trained model. Use the tool to
/// produce sample model files for examples and tests. Real models come from Darknet.
///
/// usage: tinyyolo_make_weights model.cfg model.weights

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "TinyYolo.h"

namespace {

class StdoutPrint : public tinyyolo::Print {
 public:
  size_t write(uint8_t c) override {
    fputc(c, stderr);
    return 1;
  }
  void flush() override { fflush(stderr); }
};

bool readFile(const char* path, std::vector<char>& bytes) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) return false;
  char buffer[4096];
  size_t n;
  while ((n = fread(buffer, 1, sizeof(buffer), f)) > 0) bytes.insert(bytes.end(), buffer, buffer + n);
  fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  using namespace tinyyolo;
  StdoutPrint log_output;
  TinyYoloLogger.begin(log_output, TinyYoloLogLevel::Info);
  if (argc != 3) {
    fprintf(stderr, "usage: %s model.cfg model.weights\n", argv[0]);
    return 2;
  }

  std::vector<char> cfg_bytes;
  if (!readFile(argv[1], cfg_bytes)) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  BufferStream cfg(cfg_bytes.data(), cfg_bytes.size());
  NetworkDescription desc;
  if (desc.parseCfg(cfg) != Status::Ok) return 1;

  Model model;
  ReferenceBackend backend;
  if (model.build(desc.spec(), backend) != Status::Ok) return 1;

  // Darknet v0.2 header: version (3 x int32), then "seen" as uint64.
  std::vector<uint8_t> file;
  const uint32_t version[3] = {0, 2, 0};
  const uint64_t seen = 0;
  file.insert(file.end(), (const uint8_t*)version, (const uint8_t*)version + sizeof(version));
  file.insert(file.end(), (const uint8_t*)&seen, (const uint8_t*)&seen + sizeof(seen));

  uint32_t state = 42;
  auto random01 = [&state]() -> float {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) & 0xFFFF) / 65535.0f;
  };
  auto append = [&file](float v) {
    const uint8_t* p = (const uint8_t*)&v;
    file.insert(file.end(), p, p + sizeof(float));
  };

  const NetworkSpec& spec = desc.spec();
  for (int i = 0; i < spec.layer_count; ++i) {
    const LayerSpec& layer = spec.layers[i];
    if (layer.type != LayerType::Conv) continue;
    const int out = layer.filters;
    const size_t weight_count = model.layer(i).paramCount() - (size_t)out;
    if (layer.batch_norm) {
      for (int o = 0; o < out; ++o) append((random01() - 0.5f) * 0.2f);  // beta
      for (int o = 0; o < out; ++o) append(0.8f + 0.4f * random01());     // gamma
      for (int o = 0; o < out; ++o) append((random01() - 0.5f) * 0.2f);  // running mean
      for (int o = 0; o < out; ++o) append(0.5f + random01());            // running variance
    } else {
      for (int o = 0; o < out; ++o) append((random01() - 0.5f) * 0.2f);  // bias
    }
    for (size_t w = 0; w < weight_count; ++w) append((random01() - 0.5f) * 0.2f);
  }

  FILE* out = fopen(argv[2], "wb");
  if (out == nullptr) {
    fprintf(stderr, "cannot write %s\n", argv[2]);
    return 1;
  }
  fwrite(file.data(), 1, file.size(), out);
  fclose(out);
  fprintf(stderr, "wrote %s: %lu bytes for %d layers\n", argv[2], (unsigned long)file.size(),
          spec.layer_count);
  return 0;
}
