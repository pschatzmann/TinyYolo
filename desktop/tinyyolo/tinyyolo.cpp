// tinyyolo: one desktop tool for the models of this repository, on a webcam or still pictures.
//
// Models (choose one; default: the YOLO26n export in models/yolo26n/export of this repository, the
// fastest model on the desktop, or the built-in Darknet sample when that folder is not there):
//   --model <base>     a Darknet model: <base>.cfg, <base>.names, <base>.weights
//                      (e.g. models/yolov4-tiny/yolov4-tiny; the cfg sets the input size)
//   --yolo11 <folder>  a YOLO11 export (models/yolo11n/export)
//   --yolo26 <folder>  a YOLO26 export (models/yolo26n/export)
// Pictures (default: the webcam):
//   --device <path>    V4L2 device (default: TINY_CAMERA_DEVICE, else /dev/video0)
//   --image <ppm>      a binary PPM picture; repeat the option for several (they are shown in turn)
//   --serial <port>    the stream of a board running examples/CameraStream (921600 baud); the board
//                      does the detection, and the window shows its pictures and detections
//   --rotate <deg>     rotate the picture of a board stream: 0 (default), 90, 180 or 270, clockwise
// Options:
//   --float            float32 weights (default: int8)
//   --calibrate        Darknet: int8 activations, calibrated on the first picture
//   --per-tensor       YOLO exports: per-tensor activation scales (default: shared scales)
//   --size N           YOLO exports: network input, square (default 320)
//   --score T          score threshold (default 0.25)
//   --names <file>     class names of a YOLO export (default: <base>.names in the folder or its parent)
//   --frames N         stop after N recognitions (default: no limit in the window, 1 without a window)
//   --no-window        no display: print each recognition and its time on stdout
// The window shows the picture with the boxes and the results below it. Close it to exit.
//
// Recognition runs on a worker thread in the window; the picture stays live. Without a window the
// recognitions run in turn.

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <deque>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <TinyCamera.h>
#include <TinyGPU.h>
#include <TinyGPU/Boards/LCDBoards.h>
#include <TinyMaterialDesign.h>
#include <TinyYolo.h>
#include <TinyYolo/TinyCamera/CameraFrame.h>
#include <TinyYolo/app/GraphYolo11.h>
#include <TinyYolo/backend/DotProductBackend.h>
#include <TinyYolo/app/TinyYoloModel.h>
#include <TinyYolo/platform/Task.h>
#include <TinyYolo/TinyMaterialDesign/DetectionView.h>
#include "DarknetModel.h"  // the built-in Darknet sample
#include "yolo11n_graph.inc"
#include "yolo26n_graph.inc"

using namespace tinygpu;  // TinyGPU's display types
using namespace tinyyolo;

namespace {

constexpr int kCameraWidth = 320;   // QVGA
constexpr int kCameraHeight = 240;
constexpr int kListHeight = 120;    // space below the picture for the results (the card)

/// Log output on stderr.
class StderrPrint : public tinyyolo::Print {
 public:
  size_t write(uint8_t c) override {
    fputc(c, stderr);
    return 1;
  }
  void flush() override { fflush(stderr); }
};

/// A Darknet model read from files: <base>.cfg, <base>.names and <base>.weights.
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
      fprintf(stderr, "tinyyolo: cannot read %s\n", path.c_str());
      return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
  }
  std::vector<uint8_t> cfg_, names_, weights_;
  std::optional<BufferStream> cfg_stream_, names_stream_, weights_stream_;
};

/// Reads a binary P6 PPM with maxval 255 into RGB888.
bool readPpm(const std::string& path, std::vector<uint8_t>& rgb, int& width, int& height) {
  std::ifstream in(path, std::ios::binary);
  std::string magic;
  int maxval = 0;
  if (!(in >> magic >> width >> height >> maxval) || magic != "P6" || maxval != 255) return false;
  in.get();
  rgb.resize((size_t)width * height * 3);
  in.read(reinterpret_cast<char*>(rgb.data()), (std::streamsize)rgb.size());
  return (bool)in;
}

/// RGB565 little-endian pixels to RGB888, in integers. Only the detection path needs RGB888.
void rgb565ToRgb888(const uint8_t* src, uint8_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i, src += 2, dst += 3) {
    const unsigned pixel = src[0] | (src[1] << 8);
    dst[0] = (uint8_t)((((pixel >> 11) & 31) * 255 + 15) / 31);
    dst[1] = (uint8_t)((((pixel >> 5) & 63) * 255 + 31) / 63);
    dst[2] = (uint8_t)(((pixel & 31) * 255 + 15) / 31);
  }
}

/// RGB888 to RGB565 little-endian (for the still pictures, once at load).
void rgb888ToRgb565(const uint8_t* src, uint8_t* dst, size_t count) {
  for (size_t i = 0; i < count; ++i, src += 3, dst += 2) {
    const unsigned packed = ((src[0] >> 3) << 11) | ((src[1] >> 2) << 5) | (src[2] >> 3);
    dst[0] = (uint8_t)(packed & 0xFF);
    dst[1] = (uint8_t)(packed >> 8);
  }
}

bool readFile(const std::string& path, std::vector<uint8_t>& bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  return true;
}

std::vector<std::string> readNames(const std::string& path) {
  std::vector<std::string> names;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    names.push_back(line);
  }
  return names;
}

// ---------------------------------------------------------------- recognition

struct Options {
  bool use_int8 = true;
  bool calibrate = false;
  bool per_tensor = false;
  int size = 320;
  float score = 0.25f;
  std::string names_path;
};

/// One recognizer for every model: a Darknet model or a YOLO export.
class Recognizer {
 public:
  virtual ~Recognizer() {}
  virtual const char* name() const = 0;
  /// The model, e.g. "YOLO11n" or "yolov4-tiny", and its settings, e.g. "int8 weights".
  virtual std::string description() const = 0;
  virtual std::string detail() const = 0;
  virtual bool ready() const = 0;
  virtual Status detect(const ImageView& image) = 0;
  virtual const Detections& detections() const = 0;
  virtual const char* label(int class_id) const = 0;
  virtual double lastMs() const = 0;
};

/// A Darknet model (any cfg), through TinyYoloModel.
class DarknetRecognizer : public Recognizer {
 public:
  DarknetRecognizer(ModelSource& source, const Options& options, std::string model_name)
      : options_(options), modelName_(std::move(model_name)) {
    DetectorConfig config;
    config.score_threshold = options.score;
    config.weight_precision = options.use_int8 ? WeightPrecision::Int8 : WeightPrecision::Float32;
    ready_ = model_.begin(source, config) == Status::Ok;
    if (!ready_) fprintf(stderr, "tinyyolo: the Darknet model could not be set up\n");
  }
  const char* name() const override { return "Darknet"; }
  std::string description() const override { return modelName_; }
  std::string detail() const override {
    if (!options_.use_int8) return "float32 weights";
    if (!options_.calibrate) return "int8 weights";
    return calibrated_ ? "int8 activations, calibrated" : "int8 activations (calibrated on the first picture)";
  }
  bool ready() const override { return ready_; }
  Status detect(const ImageView& image) override {
    if (options_.calibrate && !calibrated_) {
      calibrated_ = true;
      if (model_.detector().calibrate(image) != Status::Ok) return Status::InvalidArgument;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const Status status = model_.detect(image);
    ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return status;
  }
  const Detections& detections() const override { return model_.detections(); }
  const char* label(int class_id) const override { return model_.label(class_id); }
  double lastMs() const override { return ms_; }

 private:
  const Options& options_;
  TinyYoloModel model_;
  bool ready_ = false;
  std::atomic<bool> calibrated_{false};  // set by the worker thread
  double ms_ = 0;
  std::string modelName_;
};

/// Threads for the recognition: all cores but one, so that the display keeps its turns.
int freeCores() {
  const int cores = (int)std::thread::hardware_concurrency();
  return cores > 2 ? cores - 1 : 1;
}

/// A YOLO export (YOLO11 or YOLO26), through GraphYolo11: the weights from the export folder.
class GraphRecognizer : public Recognizer {
 public:
  GraphRecognizer(const char* family, const GraphModel& table, const GraphQuant* per_tensor,
                  const std::string& folder, const std::string& base, const Options& options)
      : family_(family), base_(base), options_(options) {
    if (!readFile(folder + "/" + base + "_graph_weights.bin", weights_) ||
        (options.use_int8 && (!readFile(folder + "/" + base + "_int8_weights.bin", qweights_) ||
                              !readFile(folder + "/" + base + "_int8_scales.bin", qscales_)))) {
      fprintf(stderr, "tinyyolo: cannot read the %s weights from %s\n", family, folder.c_str());
      return;
    }
    names_ = readNames(options.names_path.empty() ? folder + "/" + base + ".names" : options.names_path);
    if (names_.empty()) names_ = readNames(folder + "/../" + base + ".names");
    GraphModel model = table;
    model.input_size = options.size;
    if (options.use_int8 && options.per_tensor) model.quant = per_tensor;
    if (!options.use_int8) model.quant = nullptr;
    ready_ = detector_.begin(model, backend_, reinterpret_cast<const float*>(weights_.data()),
                             options.use_int8 ? reinterpret_cast<const int8_t*>(qweights_.data()) : nullptr,
                             options.use_int8 ? reinterpret_cast<const float*>(qscales_.data()) : nullptr,
                             options.score) == Status::Ok &&
             detections_.reserve(100) == Status::Ok;
    if (!ready_) fprintf(stderr, "tinyyolo: the %s graph could not be set up\n", family);
  }
  const char* name() const override { return family_; }
  std::string description() const override {
    std::string model = base_;  // "yolo11n" -> "YOLO11n"
    for (size_t i = 0; i < 4 && i < model.size(); ++i) model[i] = (char)toupper((unsigned char)model[i]);
    return model;
  }
  std::string detail() const override {
    return options_.use_int8 ? (options_.per_tensor ? "int8, per-tensor scales" : "int8, shared scales")
                             : "float32 weights";
  }
  bool ready() const override { return ready_; }
  Status detect(const ImageView& image) override {
    const Status status = detector_.detect(image, detections_);
    ms_ = detector_.lastGraphMicros() / 1000.0 + detector_.lastHeadMicros() / 1000.0;
    return status;
  }
  const Detections& detections() const override { return detections_; }
  const char* label(int class_id) const override {
    return class_id >= 0 && class_id < (int)names_.size() ? names_[class_id].c_str() : "?";
  }
  double lastMs() const override { return ms_; }

 private:
  const char* family_;
  std::string base_;
  const Options& options_;
  DotProductBackend backend_{freeCores()};
  GraphYolo11 detector_;
  Detections detections_;
  std::vector<uint8_t> weights_, qweights_, qscales_;
  std::vector<std::string> names_;
  bool ready_ = false;
  double ms_ = 0;
};

// ---------------------------------------------------------------- pictures

/// A picture source: the webcam, or still pictures in turn. Pictures are RGB888.
class PictureSource {
 public:
  virtual ~PictureSource() {}
  virtual bool open() = 0;
  virtual int width() const = 0;
  virtual int height() const = 0;
  virtual bool next(std::vector<uint8_t>& rgb) = 0;  // false: no picture yet
  /// The source, for the header of the result card, e.g. "webcam /dev/video0".
  virtual std::string description() const = 0;
};

class CameraSource : public PictureSource {
 public:
  explicit CameraSource(const char* device) : device_(device) {}
  bool open() override {
    camera_config_t config = tiny_camera::TinyCamera::defaultConfig();
    config.frame_size = FRAMESIZE_QVGA;
    config.device = device_;
    if (!frame_.begin(config)) {
      fprintf(stderr, "tinyyolo: cannot start the camera\n");
      return false;
    }
    return true;
  }
  int width() const override { return kCameraWidth; }
  int height() const override { return kCameraHeight; }
  std::string description() const override { return std::string("webcam ") + device_; }
  bool next(std::vector<uint8_t>& rgb) override {
    if (!frame_.next()) return false;
    // The camera's RGB565 bytes as they are (the display uses them; the detection converts).
    const uint8_t* src = frame_.image().data();
    rgb.assign(src, src + (size_t)kCameraWidth * kCameraHeight * 2);
    return true;
  }

 private:
  const char* device_;
  CameraFrame frame_{kCameraWidth, kCameraHeight, PixelFormat::RGB565Little};
};

class ImageSource : public PictureSource {
 public:
  explicit ImageSource(std::vector<std::string> paths) : paths_(std::move(paths)) {}
  bool open() override {
    if (paths_.empty()) return false;
    for (const std::string& path : paths_) {
      Picture picture;
      std::vector<uint8_t> rgb;
      if (!readPpm(path, rgb, picture.width, picture.height)) {
        fprintf(stderr, "tinyyolo: cannot read %s (binary PPM, P6, maxval 255)\n", path.c_str());
        return false;
      }
      picture.rgb.resize((size_t)picture.width * picture.height * 2);
      rgb888ToRgb565(rgb.data(), picture.rgb.data(), (size_t)picture.width * picture.height);
      pictures_.push_back(std::move(picture));
    }
    return true;
  }
  int width() const override { return pictures_.empty() ? 0 : pictures_[0].width; }
  int height() const override { return pictures_.empty() ? 0 : pictures_[0].height; }
  std::string description() const override {
    if (paths_.empty()) return "";
    const std::string& first = paths_[0];
    const size_t slash = first.find_last_of('/');
    std::string name = slash == std::string::npos ? first : first.substr(slash + 1);
    return paths_.size() > 1 ? name + " (+" + std::to_string(paths_.size() - 1) + ")" : name;
  }
  bool next(std::vector<uint8_t>& rgb) override {
    if (pictures_.empty()) return false;
    const Picture& picture = pictures_[index_];
    index_ = (index_ + 1) % pictures_.size();
    rgb = picture.rgb;
    return true;
  }

 private:
  struct Picture {
    std::vector<uint8_t> rgb;
    int width = 0;
    int height = 0;
  };
  std::vector<std::string> paths_;
  std::vector<Picture> pictures_;
  size_t index_ = 0;
};

// ---------------------------------------------------------------- the program

StderrPrint log_output;
Options options;
std::unique_ptr<Recognizer> recognizer;
std::unique_ptr<PictureSource> source;
std::optional<FileModelSource> file_source;
FlashModelSource builtin_source(kModelCfg, kModelNames, kModelWeights);
ModelSource* darknet_source = &builtin_source;
int max_frames = 0;
bool use_window = true;
int frames_done = 0;

/// The camera feeds the display with every picture; the recognition takes the newest picture
/// when its previous recognition has completed. One buffer each: nothing is queued.
std::mutex picture_mutex;
std::vector<uint8_t> latest_rgb;      // the newest picture (written by the display loop)
bool has_latest = false;
bool new_latest = false;              // a picture that the recognition has not taken yet
std::vector<Detection> latest_results;
bool has_new_results = false;
bool stop_worker = false;
Task worker_task;

void recognizeInWorker() {
  std::vector<uint8_t> input;      // the recognition's own copy of the picture (RGB565)
  std::vector<uint8_t> detect_rgb; // the same picture as RGB888, for the detector
  while (true) {
    {
      std::lock_guard<std::mutex> lock(picture_mutex);
      if (stop_worker) return;
      if (!new_latest) {
        // nothing new: wait for the next picture
      } else {
        input = latest_rgb;
        new_latest = false;
      }
    }
    if (input.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    // The detection path converts the picture to RGB888 (the display works on the RGB565 bytes).
    detect_rgb.resize((size_t)source->width() * source->height() * 3);
    rgb565ToRgb888(input.data(), detect_rgb.data(), (size_t)source->width() * source->height());
    const ImageView image(detect_rgb.data(), source->width(), source->height(), PixelFormat::RGB888);
    if (recognizer->detect(image) == Status::Ok) {
      const Detections& found = recognizer->detections();
      std::lock_guard<std::mutex> lock(picture_mutex);
      latest_results.assign(found.begin(), found.end());
      has_new_results = true;
    }
    input.clear();
  }
}

struct WorkerStopper {
  ~WorkerStopper() {
    {
      std::lock_guard<std::mutex> lock(picture_mutex);
      stop_worker = true;
    }
    worker_task.end();
  }
};

/// Picture buffer and the display objects (created in setupWindow).
std::vector<uint8_t> current_rgb;
std::unique_ptr<LCDBoardDesktopSDL> board;
std::unique_ptr<tinymd::App<RGB565>> app;
std::unique_ptr<DetectionView<RGB565>> view;

void printUsage(const char* program) {
  fprintf(stderr,
          "usage: %s [--model <base> | --yolo11 <folder> | --yolo26 <folder>] [--device <path> | --image <ppm>]...\n"
          "       [--serial <port> [--rotate <deg>]] [--float] [--calibrate] [--per-tensor] [--size N] [--score T]\n"
          "       [--names <file>] [--frames N] [--no-window]\n",
          program);
}

}  // namespace


// ---------------------------------------------------------------- a board stream (CameraStream)
//
// A board running examples/CameraStream sends each picture (RGB565) with its detections over the
// serial port (921600 baud). The board does the detection; the window only shows the stream. The
// reader runs on a task of its own, and the display takes the newest picture.

namespace {

constexpr int kBoardWidth = 320;   // the window's picture size for a board stream
constexpr int kBoardHeight = 240;

std::string board_port;
int board_rotation = 0;           // degrees clockwise: 0, 90, 180 or 270
std::vector<std::string> board_names;
int board_fd = -1;
std::mutex board_mutex;
std::vector<uint8_t> board_picture;   // the newest picture: RGB565 little-endian, 320x240
std::vector<Detection> board_boxes;   // its detections (in the 320x240 picture space)
bool board_new = false;
bool board_failed = false;
Task* board_task = new Task();        // never destroyed: the reader may be blocked in read()

bool openBoard(const std::string& path) {
  board_fd = open(path.c_str(), O_RDWR | O_NOCTTY);
  if (board_fd < 0) return false;
  termios tio{};
  tcgetattr(board_fd, &tio);
  cfmakeraw(&tio);
  cfsetispeed(&tio, B921600);
  cfsetospeed(&tio, B921600);
  tio.c_cflag |= CLOCAL | CREAD;
  tio.c_cc[VMIN] = 1;
  tio.c_cc[VTIME] = 0;
  return tcsetattr(board_fd, TCSANOW, &tio) == 0;
}

bool readBoardBytes(uint8_t* p, size_t n) {
  while (n > 0) {
    const ssize_t got = read(board_fd, p, n);
    if (got <= 0) return false;
    p += got;
    n -= (size_t)got;
  }
  return true;
}

bool readBoardLine(std::string& line) {
  line.clear();
  char c;
  while (true) {
    if (read(board_fd, &c, 1) <= 0) return false;
    if (c == '\n') return true;
    if (c != '\r') line += c;
  }
}

/// Reads one frame of the CameraStream protocol: the header "FRAME w h bytes detections", the
/// detections (one line each), then the picture bytes. Lines that do not parse are skipped.
bool readBoardFrame(std::vector<uint8_t>& raw, int& w, int& h, std::vector<Detection>& boxes) {
  std::string line;
  int nbytes = 0, ndet = 0;
  while (true) {
    if (!readBoardLine(line)) return false;
    // The picture bytes are not newline-terminated, so the header can follow them on the same line.
    const size_t at = line.find("FRAME");
    if (at != std::string::npos && sscanf(line.c_str() + at, "FRAME %d %d %d %d", &w, &h, &nbytes, &ndet) == 4) break;
  }
  if (w <= 0 || h <= 0 || nbytes != w * h * 2 || ndet < 0) return true;  // skip this frame
  boxes.clear();
  for (int i = 0; i < ndet; ++i) {
    if (!readBoardLine(line)) return false;
    Detection d{};
    if (sscanf(line.c_str(), "%f %f %f %f %f %d", &d.x, &d.y, &d.width, &d.height, &d.score, &d.class_id) == 6) {
      boxes.push_back(d);
    }
  }
  raw.resize((size_t)nbytes);
  return readBoardBytes(raw.data(), raw.size());
}

/// The picture scaled (nearest neighbour) and rotated to the window, still RGB565.
void rotateToWindow(const std::vector<uint8_t>& raw, int w, int h, std::vector<uint8_t>& out) {
  out.resize((size_t)kBoardWidth * kBoardHeight * 2);
  for (int y = 0; y < kBoardHeight; ++y) {
    for (int x = 0; x < kBoardWidth; ++x) {
      int sx, sy;
      switch (board_rotation) {
        case 90:  sx = y * w / kBoardHeight;               sy = h - 1 - x * h / kBoardWidth; break;
        case 180: sx = w - 1 - x * w / kBoardWidth;        sy = h - 1 - y * h / kBoardHeight; break;
        case 270: sx = w - 1 - y * w / kBoardHeight;       sy = x * h / kBoardWidth; break;
        default:  sx = x * w / kBoardWidth;                sy = y * h / kBoardHeight; break;
      }
      const uint8_t* src = &raw[((size_t)sy * w + sx) * 2];
      uint8_t* dst = &out[((size_t)y * kBoardWidth + x) * 2];
      dst[0] = src[0];
      dst[1] = src[1];
    }
  }
}

/// The reader task: one frame after the other, into the shared picture.
void readBoard() {
  std::vector<uint8_t> raw, picture;
  std::vector<Detection> boxes;
  while (true) {
    int w = 0, h = 0;
    if (!readBoardFrame(raw, w, h, boxes)) {
      std::lock_guard<std::mutex> lock(board_mutex);
      board_failed = true;
      return;
    }
    if (w <= 0 || h <= 0 || raw.size() != (size_t)w * h * 2) continue;
    rotateToWindow(raw, w, h, picture);
    std::lock_guard<std::mutex> lock(board_mutex);
    board_picture.swap(picture);
    board_boxes = boxes;
    board_new = true;
  }
}

const char* boardLabel(int class_id) {
  static char fallback[16];
  if (class_id >= 0 && class_id < (int)board_names.size()) return board_names[class_id].c_str();
  snprintf(fallback, sizeof(fallback), "class %d", class_id);
  return fallback;
}

/// The board stream in the window. Returns the exit status.
int runBoardStream() {
  if (!openBoard(board_port)) {
    fprintf(stderr, "tinyyolo: cannot open %s\n", board_port.c_str());
    return 1;
  }
  board.reset(new LCDBoardDesktopSDL(kBoardWidth, kBoardHeight + kListHeight));
  app.reset(new tinymd::App<RGB565>(*board));
  app->begin();
  view.reset(new DetectionView<RGB565>(tinymd::Bounds(0, 0, kBoardWidth, kBoardHeight + kListHeight), kBoardWidth,
                                       kBoardHeight, [](int class_id) { return boardLabel(class_id); }));
  char rotation[32];
  snprintf(rotation, sizeof(rotation), "rotation %d", board_rotation);
  view->setInfo("CameraStream board", rotation, board_port.c_str());
  app->screen().addFixedWidget(*view);
  board_task->begin(readBoard);

  while (true) {
    {
      std::lock_guard<std::mutex> lock(board_mutex);
      if (board_failed) {
        fprintf(stderr, "tinyyolo: the board stream stopped\n");
        return 1;
      }
      if (board_new) {
        view->setPicture(ImageView(board_picture.data(), kBoardWidth, kBoardHeight, PixelFormat::RGB565Little));
        view->setDetections(board_boxes.data(), (int)board_boxes.size());
        board_new = false;
      }
    }
    app->update();  // closing the window exits
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string darknet_base, yolo11_folder, yolo26_folder;
  const char* device = nullptr;
  std::vector<std::string> images;
  bool frames_given = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--model" && has_value) darknet_base = argv[++i];
    else if (arg == "--yolo11" && has_value) yolo11_folder = argv[++i];
    else if (arg == "--yolo26" && has_value) yolo26_folder = argv[++i];
    else if (arg == "--device" && has_value) device = argv[++i];
    else if (arg == "--image" && has_value) images.push_back(argv[++i]);
    else if (arg == "--names" && has_value) options.names_path = argv[++i];
    else if (arg == "--size" && has_value) options.size = atoi(argv[++i]);
    else if (arg == "--score" && has_value) options.score = (float)atof(argv[++i]);
    else if (arg == "--frames" && has_value) { max_frames = atoi(argv[++i]); frames_given = true; }
    else if (arg == "--float") options.use_int8 = false;
    else if (arg == "--calibrate") options.calibrate = true;
    else if (arg == "--per-tensor") options.per_tensor = true;
    else if (arg == "--no-window") use_window = false;
    else if (arg == "--serial" && has_value) board_port = argv[++i];
    else if (arg == "--rotate" && has_value) {
      board_rotation = atoi(argv[++i]);
      if (board_rotation != 0 && board_rotation != 90 && board_rotation != 180 && board_rotation != 270) {
        fprintf(stderr, "tinyyolo: --rotate must be 0, 90, 180 or 270\n");
        return 2;
      }
    }
    else {
      printUsage(argv[0]);
      return 2;
    }
  }
#ifdef TINYYOLO_DEFAULT_MODEL
  // No model given: the YOLO26n export of the repository, when it is there.
  if (darknet_base.empty() && yolo11_folder.empty() && yolo26_folder.empty()) {
    std::ifstream probe(std::string(TINYYOLO_DEFAULT_MODEL) + "/yolo26n_graph_weights.bin");
    if (probe) yolo26_folder = TINYYOLO_DEFAULT_MODEL;
  }
#endif
  const int models = (!darknet_base.empty()) + (!yolo11_folder.empty()) + (!yolo26_folder.empty());
  if (models > 1) {
    fprintf(stderr, "tinyyolo: choose one model\n");
    return 2;
  }
  if (!use_window && !frames_given) max_frames = 1;

  TinyYoloLogger.begin(log_output, TinyYoloLogLevel::Info);

  // A board stream: the board detects, the window shows its pictures and detections.
  if (!board_port.empty()) {
    if (!options.names_path.empty()) board_names = readNames(options.names_path);
    return runBoardStream();
  }

  // The picture source.
  if (images.empty()) {
    source.reset(new CameraSource(device ? device : getenv("TINY_CAMERA_DEVICE") ? getenv("TINY_CAMERA_DEVICE") : "/dev/video0"));
  } else {
    source.reset(new ImageSource(images));
  }
  if (!source->open()) return 1;

  // The model.
  if (!yolo11_folder.empty() || !yolo26_folder.empty()) {
    const bool is11 = !yolo11_folder.empty();
    recognizer.reset(is11 ? static_cast<Recognizer*>(new GraphRecognizer(
                                "YOLO11", yolo11::kModel, yolo11::kQuantPerTensor, yolo11_folder, "yolo11n", options))
                          : static_cast<Recognizer*>(new GraphRecognizer(
                                "YOLO26", yolo26::kModel, yolo26::kQuantPerTensor, yolo26_folder, "yolo26n", options)));
  } else {
    if (!darknet_base.empty()) {
      file_source.emplace();
      if (!file_source->load(darknet_base)) return 1;
      darknet_source = &*file_source;
    }
    std::string name = "built-in sample";
    if (!darknet_base.empty()) {
      const size_t slash = darknet_base.find_last_of('/');
      name = slash == std::string::npos ? darknet_base : darknet_base.substr(slash + 1);
    }
    recognizer.reset(new DarknetRecognizer(*darknet_source, options, name));
  }
  if (!recognizer->ready()) return 1;
  fprintf(stderr, "tinyyolo: %s, %s, pictures %dx%d\n", recognizer->name(),
          options.use_int8 ? "int8" : "float32", source->width(), source->height());

  if (!use_window) {
    // Without a window the recognitions run in turn and print their results.
    std::vector<uint8_t> rgb;
    for (int frame = 1; max_frames == 0 || frame <= max_frames; ++frame) {
      if (!source->next(rgb)) {
        if (images.empty()) { std::this_thread::sleep_for(std::chrono::milliseconds(10)); --frame; continue; }
        break;
      }
      std::vector<uint8_t> detect_rgb((size_t)source->width() * source->height() * 3);
      rgb565ToRgb888(rgb.data(), detect_rgb.data(), (size_t)source->width() * source->height());
      const ImageView input(detect_rgb.data(), source->width(), source->height(), PixelFormat::RGB888);
      if (recognizer->detect(input) != Status::Ok) {
        fprintf(stderr, "tinyyolo: detection failed\n");
        return 1;
      }
      const Detections& found = recognizer->detections();
      printf("frame %d: %d detections, %.1f ms\n", frame, found.size(), recognizer->lastMs());
      for (int i = 0; i < found.size(); ++i) {
        const Detection& d = found[i];
        printf("  class %d %s score %.3f box %.1f %.1f %.1f %.1f\n", d.class_id, recognizer->label(d.class_id),
               d.score, d.x, d.y, d.x + d.width, d.y + d.height);
      }
    }
    return 0;
  }

  // The window: picture, boxes and results.
  static WorkerStopper stopper;  // declared before the window objects, so it runs before them
  board.reset(new LCDBoardDesktopSDL(source->width(), source->height() + kListHeight));
  app.reset(new tinymd::App<RGB565>(*board));
  app->begin();
  Recognizer* rec = recognizer.get();
  view.reset(new DetectionView<RGB565>(
      tinymd::Bounds(0, 0, source->width(), source->height() + kListHeight), source->width(),
      source->height(), [rec](int class_id) { return rec->label(class_id); }));
  view->setInfo(recognizer->description().c_str(), recognizer->detail().c_str(), source->description().c_str());
  std::string shown_detail = recognizer->detail();
  app->screen().addFixedWidget(*view);
  worker_task.begin(recognizeInWorker);

  std::vector<uint8_t> rgb;
  for (;;) {
    if (source->next(rgb)) {
      {
        std::lock_guard<std::mutex> lock(picture_mutex);
        latest_rgb.swap(rgb);  // the newest picture; the recognition copies it when it is ready
        new_latest = true;
        has_latest = true;
      }
      // The display is fed from the camera with every picture.
      const ImageView picture(latest_rgb.data(), source->width(), source->height(), PixelFormat::RGB565Little);
      view->setPicture(picture);
    }
    // The model card follows the state of the recognizer (calibration finishes on the worker).
    const std::string detail_now = recognizer->detail();
    if (detail_now != shown_detail) {
      shown_detail = detail_now;
      view->setInfo(recognizer->description().c_str(), detail_now.c_str(), source->description().c_str());
    }
    {
      std::lock_guard<std::mutex> lock(picture_mutex);
      if (has_new_results) {
        view->setDetections(latest_results.data(), (int)latest_results.size());
        has_new_results = false;
        ++frames_done;
      }
    }
    if (max_frames > 0 && frames_done >= max_frames) break;
    app->update();  // closing the window exits
    // At most about 100 turns per second: the camera and the recognition get the rest of the time.
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return 0;
}
