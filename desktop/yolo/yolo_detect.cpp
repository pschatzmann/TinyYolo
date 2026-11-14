// yolo_detect: a YOLO model on the desktop, through the same GraphYolo11 class as the boards.
// The targets yolo11_detect and yolo26_detect (CMakeLists.txt) set the model with macros:
//   YOLO_NS (the table namespace), YOLO_BASE (the export's file base name), YOLO_TABLE (the table).
//
// usage: yolo11_detect <export folder> <image.ppm> [--size N] [--per-tensor] [--float]
//   The export folder holds <base>_graph_weights.bin, <base>_int8_weights.bin and
//   <base>_int8_scales.bin (tools/yolo/yolo_export.py writes them). The image is a binary PPM (P6).
//   The default is the int8 graph with the shared-scale table; --per-tensor selects the per-tensor
//   table, --float runs the float graph. Boxes are printed in the image's pixels.

#include <stdio.h>

#include <string>
#include <vector>

#include "TinyYolo.h"
#include "TinyYolo/app/GraphYolo11.h"
#include "TinyYolo/backend/DotProductBackend.h"
#include YOLO_TABLE  // the table: YOLO_NS::kModel and the other tables

using namespace tinyyolo;

static bool readBytes(const std::string& path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (f == nullptr) return false;
  fseek(f, 0, SEEK_END);
  const long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  out.resize((size_t)bytes);
  const size_t got = fread(out.data(), 1, out.size(), f);
  fclose(f);
  return got == out.size();
}

/// Reads a binary PPM (P6, maxval 255). Returns false for anything else.
static bool readPpm(const std::string& path, std::vector<uint8_t>& pixels, int& width, int& height) {
  std::vector<uint8_t> file;
  if (!readBytes(path, file) || file.size() < 15 || file[0] != 'P' || file[1] != '6') return false;
  size_t pos = 2;
  int numbers[3] = {0, 0, 0};
  for (int i = 0; i < 3; ++i) {
    while (pos < file.size() && (file[pos] == ' ' || file[pos] == '\n' || file[pos] == '\r' || file[pos] == '\t')) ++pos;
    if (pos >= file.size() || file[pos] < '0' || file[pos] > '9') return false;
    while (pos < file.size() && file[pos] >= '0' && file[pos] <= '9') numbers[i] = numbers[i] * 10 + (file[pos++] - '0');
  }
  if (numbers[2] != 255) return false;
  ++pos;  // the single whitespace byte after maxval
  width = numbers[0];
  height = numbers[1];
  const size_t bytes = (size_t)width * height * 3;
  if (file.size() - pos < bytes) return false;
  pixels.assign(file.begin() + pos, file.begin() + pos + bytes);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <export folder> <image.ppm> [--size N] [--per-tensor] [--float]\n", YOLO_BASE);
    return 2;
  }
  const std::string dir = argv[1];
  bool per_tensor = false, use_int8 = true;
  int input_size = 640;  // the network input (the table's default); --size N sets another square size
  for (int i = 3; i < argc; ++i) {
    per_tensor = per_tensor || std::string(argv[i]) == "--per-tensor";
    use_int8 = use_int8 && std::string(argv[i]) != "--float";
    if (std::string(argv[i]) == "--size" && i + 1 < argc) input_size = atoi(argv[++i]);
  }

  std::vector<uint8_t> weight_bytes, qweight_bytes, qscale_bytes, pixels;
  int width = 0, height = 0;
  const std::string base = YOLO_BASE;
  if (!readBytes(dir + "/" + base + "_graph_weights.bin", weight_bytes) ||
      (use_int8 && (!readBytes(dir + "/" + base + "_int8_weights.bin", qweight_bytes) ||
                    !readBytes(dir + "/" + base + "_int8_scales.bin", qscale_bytes)))) {
    fprintf(stderr, "cannot read the weights from %s\n", dir.c_str());
    return 1;
  }
  if (!readPpm(argv[2], pixels, width, height)) {
    fprintf(stderr, "cannot read %s (binary PPM, P6, maxval 255)\n", argv[2]);
    return 1;
  }

  GraphModel model = YOLO_NS::kModel;
  model.input_size = input_size;
  if (use_int8 && per_tensor) model.quant = YOLO_NS::kQuantPerTensor;
  if (!use_int8) model.quant = nullptr;
  DotProductBackend backend;
  GraphYolo11 detector;
  Detections detections;
  if (detections.reserve(100) != Status::Ok ||
      detector.begin(model, backend, reinterpret_cast<const float*>(weight_bytes.data()),
                     use_int8 ? reinterpret_cast<const int8_t*>(qweight_bytes.data()) : nullptr,
                     use_int8 ? reinterpret_cast<const float*>(qscale_bytes.data()) : nullptr) != Status::Ok) {
    fprintf(stderr, "detector setup failed\n");
    return 1;
  }
  const ImageView image(pixels.data(), width, height, PixelFormat::RGB888);
  if (detector.detect(image, detections) != Status::Ok) {
    fprintf(stderr, "detection failed\n");
    return 1;
  }
  printf("%d detections (graph %.1f ms, head %.1f ms)\n", detections.size(),
         detector.lastGraphMicros() / 1000.0, detector.lastHeadMicros() / 1000.0);
  for (int i = 0; i < detections.size(); ++i) {
    const Detection& d = detections[i];
    printf("class %d score %.3f box %.1f %.1f %.1f %.1f\n", d.class_id, d.score, d.x, d.y, d.x + d.width,
           d.y + d.height);
  }
  return 0;
}
