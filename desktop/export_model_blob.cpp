/// export_model_blob: converts a Darknet model into a model blob (see TinyYolo/model/ModelBlob.h).
///
/// Convolutions are stored as int8 (per output channel), except the ones that feed a YOLO
/// head, which stay float32. The blob does not depend on the input size: the same weights work
/// at any input size the network allows.
///
/// usage: tinyyolo_export_blob model.cfg model.weights model.blob

#include <stdio.h>
#include <stdint.h>

#include <fstream>
#include <iterator>
#include <string>
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

std::vector<uint8_t> readFile(const char* path, bool& ok) {
  std::ifstream in(path, std::ios::binary);
  ok = (bool)in;
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/// Writes blob bytes to a file.
class FileSink {
 public:
  explicit FileSink(FILE* f) : f_(f) {}
  size_t write(const void* data, size_t n) { return fwrite(data, 1, n, f_); }

 private:
  FILE* f_;
};

}  // namespace

int main(int argc, char** argv) {
  using namespace tinyyolo;
  StdoutPrint log_output;
  TinyYoloLogger.begin(log_output, TinyYoloLogLevel::Info);
  if (argc != 4) {
    fprintf(stderr, "usage: %s model.cfg model.weights model.blob\n", argv[0]);
    return 2;
  }

  bool ok = false;
  std::vector<uint8_t> cfg_bytes = readFile(argv[1], ok);
  if (!ok) return 1;
  std::vector<uint8_t> weight_bytes = readFile(argv[2], ok);
  if (!ok) return 1;

  BufferStream cfg(cfg_bytes.data(), cfg_bytes.size());
  NetworkDescription desc;
  if (desc.parseCfg(cfg) != Status::Ok) return 1;

  ReferenceBackend backend;
  Model model;
  if (model.build(desc.spec(), backend, WeightPrecision::Int8) != Status::Ok) return 1;

  BufferStream weight_stream(weight_bytes.data(), weight_bytes.size());
  DarknetWeightSource weights(weight_stream);
  if (weights.open() != Status::Ok || model.loadWeights(weights) != Status::Ok) return 1;
  if (weights.finish() != Status::Ok) return 1;

  FILE* out = fopen(argv[3], "wb");
  if (out == nullptr) {
    fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  FileSink sink(out);
  const Status status = writeModelBlob(model, sink);
  fclose(out);
  if (status != Status::Ok) return 1;
  fprintf(stderr, "wrote %s: %.2f MB of weights (%lu parameters)\n", argv[3],
          model.weightBytes() / 1e6, (unsigned long)model.paramCount());
  return 0;
}
