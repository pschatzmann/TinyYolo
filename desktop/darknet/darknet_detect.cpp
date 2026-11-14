// darknet_detect: runs a Darknet model (.cfg, .names, .weights) on a PPM image on the desktop,
// through the same TinyYoloModel as the boards. Prints the detections and the mean detection time.
//
// usage: darknet_detect <model base> <image.ppm> [--size N] [--runs N] [--float]
//   <model base>  files <base>.cfg, <base>.names and <base>.weights
//   --size N      the network input (the cfg's width and height must match it: write a copy of the
//                 cfg with the new size; this tool only checks it)
//   --runs N      detections to time (default 3); the detections of the last run are printed
//   --float       float32 weights (default int8)
//   --calibrate   int8 activations too (calibrated on this image): the int8 convolution kernel
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <vector>
#include <TinyYolo.h>
#include <TinyYolo/app/TinyYoloModel.h>

using namespace tinyyolo;

namespace {

/// A model read from files: <base>.cfg, <base>.names and <base>.weights.
class FileModelSource : public ModelSource {
 public:
  bool load(const std::string& base) {
    return read(base + ".cfg", cfg_) && read(base + ".names", names_) && read(base + ".weights", weights_);
  }
  bool open() override {
    cfg_stream_.emplace(cfg_.data(), cfg_.size());
    names_stream_.emplace(names_.data(), names_.size());
    weights_stream_.emplace(weights_.data(), weights_.size());
    return true;
  }
  void close() override {}
  Stream& cfg() override { return *cfg_stream_; }
  Stream& names() override { return *names_stream_; }
  Stream& weights() override { return *weights_stream_; }

 private:
  static bool read(const std::string& path, std::vector<uint8_t>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      fprintf(stderr, "cannot read %s\n", path.c_str());
      return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
  }
  std::vector<uint8_t> cfg_, names_, weights_;
  std::optional<BufferStream> cfg_stream_, names_stream_, weights_stream_;
};

/// Binary P6 PPM with maxval 255.
bool readPpm(const char* path, std::vector<uint8_t>& pixels, int& width, int& height) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  if (!(in >> magic >> width >> height >> maxval) || magic != "P6" || maxval != 255) return false;
  in.get();
  pixels.resize((size_t)width * height * 3);
  in.read(reinterpret_cast<char*>(pixels.data()), (std::streamsize)pixels.size());
  return (bool)in;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <model base> <image.ppm> [--size N] [--runs N] [--float]\n", argv[0]);
    return 2;
  }
  int size = 0, runs = 3;
  bool use_int8 = true, calibrate = false;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--calibrate") calibrate = true;
    const std::string a = argv[i];
    if (a == "--float") use_int8 = false;
    if (a == "--size" && i + 1 < argc) size = atoi(argv[++i]);
    if (a == "--runs" && i + 1 < argc) runs = atoi(argv[++i]);
  }
  FileModelSource source;
  if (!source.load(argv[1])) return 1;
  std::vector<uint8_t> pixels;
  int width = 0, height = 0;
  if (!readPpm(argv[2], pixels, width, height)) {
    fprintf(stderr, "cannot read %s (binary PPM, P6, maxval 255)\n", argv[2]);
    return 1;
  }
  DetectorConfig config;
  config.weight_precision = use_int8 ? WeightPrecision::Int8 : WeightPrecision::Float32;
  TinyYoloModel model;
  if (model.begin(source, config) != Status::Ok) {
    fprintf(stderr, "model setup failed\n");
    return 1;
  }
  if (size > 0 && size != model.inputWidth()) {
    fprintf(stderr, "the cfg input is %d; write a cfg with width and height %d\n",
            model.inputWidth(), size);
    return 1;
  }
  const ImageView image(pixels.data(), width, height, PixelFormat::RGB888);
  if (calibrate && model.detector().calibrate(image) != Status::Ok) {
    fprintf(stderr, "calibration failed\n");
    return 1;
  }
  double total_ms = 0;
  for (int r = 0; r < runs; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    if (model.detect(image) != Status::Ok) {
      fprintf(stderr, "detection failed\n");
      return 1;
    }
    total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  const Detections& dets = model.detections();
  printf("%d detections, mean %.1f ms over %d runs (%s%s)\n", (int)dets.size(), total_ms / runs, runs,
         use_int8 ? "int8" : "float32", calibrate ? ", int8 activations" : "");
  for (int i = 0; i < dets.size(); ++i) {
    const Detection& d = dets[i];
    printf("class %d %s score %.3f box %.1f %.1f %.1f %.1f\n", d.class_id, model.detector().label(d.class_id),
           d.score, d.x, d.y, d.x + d.width, d.y + d.height);
  }
  return 0;
}
