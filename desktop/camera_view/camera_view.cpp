// camera_view: shows a desktop webcam with TinyYolo detections, using TinyMaterialDesign.
//
// Capture: TinyCamera (V4L2 on Linux; the test pattern if no webcam can be opened).
// Detection: TinyYolo with a Darknet model.
// Display: TinyMaterialDesign's App on a TinyGPU SDL window: the picture with the detection
//          boxes, and the results listed below it.
//
// usage: camera_view [--model <base>] [--device <path>] [--serial <port>]
//   --model <base>  model files <base>.cfg, <base>.names and <base>.weights, e.g.
//                   models/yolov4-tiny/yolov4-tiny. Default: the built-in sample model.
//   --device <path> V4L2 device, e.g. /dev/video1. Default: TINY_CAMERA_DEVICE, else
//                   /dev/video0.
//   --serial <port> show the stream of a board running examples/CameraStream (921600 baud),
//                   instead of a local webcam.
//   --rotate <deg>  rotate the picture of a --serial stream: 0 (default), 90, 180 or 270,
//                   clockwise.
// Close the window to exit.
//
// Written in the Arduino style (setup/loop). The main() at the end calls them.

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <termios.h>
#include <unistd.h>

#include <chrono>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

#include <TinyCamera.h>
#include <TinyGPU.h>
#include <TinyGPU/Boards/LCDBoards.h>
#include <TinyMaterialDesign.h>
#include <TinyYolo.h>
#include <TinyYolo/TinyCamera/CameraFrame.h>
#include "DetectionView.h"
#include <TinyYolo/platform/Task.h>
#include "DarknetModel.h"  // built-in sample model, the built-in sample model

void setup();
void loop();

using namespace tinygpu;  // TinyGPU's display types

namespace {

constexpr int kPictureWidth = 320;   // QVGA
constexpr int kPictureHeight = 240;
constexpr int kListHeight = 80;      // space below the picture for the results

/// Log output on stderr.
class StderrPrint : public tinyyolo::Print {
 public:
  size_t write(uint8_t c) override {
    fputc(c, stderr);
    return 1;
  }
  void flush() override { fflush(stderr); }
};

/// A model read from files on the desktop: <base>.cfg, <base>.names and <base>.weights.
class FileModelSource : public tinyyolo::ModelSource {
 public:
  /// Reads the three files into memory. Returns false (after logging) if one is missing.
  bool load(const std::string& base) {
    return read(base + ".cfg", cfg_) && read(base + ".names", names_) &&
           read(base + ".weights", weights_);
  }
  bool open() override {
    cfg_stream_.emplace(cfg_.data(), cfg_.size());
    names_stream_.emplace(names_.data(), names_.size());
    weights_stream_.emplace(weights_.data(), weights_.size());
    return true;
  }
  void close() override {}
  tinyyolo::Stream& cfg() override { return *cfg_stream_; }
  tinyyolo::Stream& names() override { return *names_stream_; }
  tinyyolo::Stream& weights() override { return *weights_stream_; }

 private:
  static bool read(const std::string& path, std::vector<uint8_t>& bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      fprintf(stderr, "camera_view: cannot read %s\n", path.c_str());
      return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
  }

  std::vector<uint8_t> cfg_;
  std::vector<uint8_t> names_;
  std::vector<uint8_t> weights_;
  std::optional<tinyyolo::BufferStream> cfg_stream_;
  std::optional<tinyyolo::BufferStream> names_stream_;
  std::optional<tinyyolo::BufferStream> weights_stream_;
};

StderrPrint log_output;
LCDBoardDesktopSDL board(kPictureWidth, kPictureHeight + kListHeight);
tinymd::App<RGB565> app(board);
tinyyolo::TinyYoloModel model;
tinyyolo::FlashModelSource builtin_source(kModelCfg, kModelNames, kModelWeights);
FileModelSource file_source;
tinyyolo::ModelSource* model_source = &builtin_source;
std::string serial_port;             // non-empty: show the stream of a board on this port
int serial_rotation = 0;             // degrees clockwise: 0, 90, 180 or 270
int serial_fd = -1;
std::vector<uint8_t> serial_raw;     // one received picture (RGB565, little-endian)
std::vector<uint8_t> serial_rgb;     // the picture as RGB888, 320x240 for the window
std::vector<tinyyolo::Detection> serial_boxes;
const char* camera_device = nullptr;  // null: TINY_CAMERA_DEVICE, else /dev/video0
tinyyolo::CameraFrame camera_frame(kPictureWidth, kPictureHeight, tinyyolo::PixelFormat::RGB565Little);
tinyyolo::DetectionView<RGB565> view(
    tinymd::Bounds(0, 0, kPictureWidth, kPictureHeight + kListHeight), kPictureWidth,
    kPictureHeight, model);

// Recognition queue: the display pushes camera frames, a background worker takes them.
// It holds at most two unprocessed frames; when it is full, the oldest one is overwritten.
constexpr size_t kQueueCapacity = 2;

struct QueuedFrame {
  std::vector<uint8_t> rgb;  // RGB888, kPictureWidth x kPictureHeight
};

std::mutex queue_mutex;
std::deque<QueuedFrame> queue;
std::vector<tinyyolo::Detection> latest_results;  // written by the worker, read by the display
bool has_new_results = false;
bool stop_worker = false;

tinyyolo::Task recognition_task;  // declared after the model, so it is destroyed first

void pushFrame(const tinyyolo::ImageView& picture) {
  QueuedFrame frame;
  frame.rgb.resize((size_t)kPictureWidth * kPictureHeight * 3);
  uint8_t* p = frame.rgb.data();
  for (int y = 0; y < kPictureHeight; ++y) {
    for (int x = 0; x < kPictureWidth; ++x) {
      float rgb[3];
      picture.rgbAt(x, y, rgb);
      p[0] = (uint8_t)(rgb[0] * 255.0f + 0.5f);
      p[1] = (uint8_t)(rgb[1] * 255.0f + 0.5f);
      p[2] = (uint8_t)(rgb[2] * 255.0f + 0.5f);
      p += 3;
    }
  }
  std::lock_guard<std::mutex> lock(queue_mutex);
  if (queue.size() >= kQueueCapacity) queue.pop_front();  // overwrite the oldest unprocessed frame
  queue.push_back(std::move(frame));
}

/// Opens the serial port of a CameraStream board: 921600 baud, raw.
bool openSerial(const std::string& path) {
  serial_fd = open(path.c_str(), O_RDWR | O_NOCTTY);
  if (serial_fd < 0) return false;
  termios tio{};
  tcgetattr(serial_fd, &tio);
  cfmakeraw(&tio);
  cfsetispeed(&tio, B921600);
  cfsetospeed(&tio, B921600);
  tio.c_cflag |= CLOCAL | CREAD;
  tio.c_cc[VMIN] = 1;
  tio.c_cc[VTIME] = 0;
  return tcsetattr(serial_fd, TCSANOW, &tio) == 0;
}

bool readSerialBytes(uint8_t* p, size_t n) {
  while (n > 0) {
    const ssize_t got = read(serial_fd, p, n);
    if (got <= 0) return false;
    p += got;
    n -= (size_t)got;
  }
  return true;
}

bool readSerialLine(std::string& line) {
  line.clear();
  char c;
  while (true) {
    if (read(serial_fd, &c, 1) <= 0) return false;
    if (c == '\n') return true;
    if (c != '\r') line += c;
  }
}

/// Reads one frame of the CameraStream protocol and shows it. Lines that do not parse are
/// skipped until the next "FRAME" header.
void serialStep() {
  std::string line;
  int w = 0, h = 0, nbytes = 0, ndet = 0;
  // The picture bytes are not newline-terminated, so the header can follow them on the same
  // line: look for "FRAME" anywhere in the line.
  while (true) {
    if (!readSerialLine(line)) return;
    // The line can hold picture bytes, which include zeros: search the whole string.
    const size_t at = line.find("FRAME");
    if (at != std::string::npos &&
        sscanf(line.c_str() + at, "FRAME %d %d %d %d", &w, &h, &nbytes, &ndet) == 4) {
      break;
    }
  }
  if (w <= 0 || h <= 0 || nbytes != w * h * 2 || ndet < 0) return;

  serial_boxes.clear();
  for (int i = 0; i < ndet; ++i) {
    if (!readSerialLine(line)) return;
    tinyyolo::Detection d{};
    if (sscanf(line.c_str(), "%f %f %f %f %f %d", &d.x, &d.y, &d.width, &d.height, &d.score,
               &d.class_id) == 6) {
      serial_boxes.push_back(d);
    }
  }
  serial_raw.resize((size_t)nbytes);
  if (!readSerialBytes(serial_raw.data(), serial_raw.size())) {
    fprintf(stderr, "serial: picture read failed\n");
    return;
  }

  // Upscale the picture to the window size (nearest neighbour), rotate it, and unpack
  // RGB565 to RGB888. For 90 and 270 degrees the picture is stretched to the window.
  const int out_w = kPictureWidth, out_h = kPictureHeight;
  serial_rgb.resize((size_t)out_w * out_h * 3);
  for (int y = 0; y < out_h; ++y) {
    for (int x = 0; x < out_w; ++x) {
      int sx, sy;
      switch (serial_rotation) {
        case 90:  sx = y * w / out_h;             sy = h - 1 - x * h / out_w; break;
        case 180: sx = w - 1 - x * w / out_w;     sy = h - 1 - y * h / out_h; break;
        case 270: sx = w - 1 - y * w / out_h;     sy = x * h / out_w;         break;
        default:  sx = x * w / out_w;             sy = y * h / out_h;         break;
      }
      const uint8_t* src = &serial_raw[((size_t)sy * w + sx) * 2];
      const uint16_t v = (uint16_t)(src[0] | (src[1] << 8));
      uint8_t* dst = &serial_rgb[((size_t)y * out_w + x) * 3];
      dst[0] = (uint8_t)(((v >> 11) & 0x1F) * 255 / 31);
      dst[1] = (uint8_t)(((v >> 5) & 0x3F) * 255 / 63);
      dst[2] = (uint8_t)((v & 0x1F) * 255 / 31);
    }
  }
  // Box coordinates come in the 320x240 picture space, which is the window's picture space.
  view.setPicture(tinyyolo::ImageView(serial_rgb.data(), out_w, out_h, tinyyolo::PixelFormat::RGB888));
  view.setDetections(serial_boxes.data(), (int)serial_boxes.size());
}

/// Worker: takes frames from the queue, runs the model and posts the results.
void recognize() {
  while (true) {
    QueuedFrame frame;
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (stop_worker) return;
      if (!queue.empty()) {
        frame = std::move(queue.front());
        queue.pop_front();
      }
    }
    if (frame.rgb.empty()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      continue;
    }
    const tinyyolo::ImageView input(frame.rgb.data(), kPictureWidth, kPictureHeight,
                                    tinyyolo::PixelFormat::RGB888);
    if (model.detect(input) == tinyyolo::Status::Ok) {
      const tinyyolo::Detections& found = model.detections();
      std::lock_guard<std::mutex> lock(queue_mutex);
      latest_results.assign(found.begin(), found.end());
      has_new_results = true;
    }
  }
}

/// Stops the worker when the program exits (static objects are destroyed on exit()).
struct WorkerStopper {
  ~WorkerStopper() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      stop_worker = true;
    }
    recognition_task.end();
  }
};
WorkerStopper worker_stopper;  // declared after the task, so it runs first on exit

}  // namespace

void setup() {
  TinyYoloLogger.begin(log_output, tinyyolo::TinyYoloLogLevel::Info);
  app.begin();
  app.screen().addFixedWidget(view);

  if (model.begin(*model_source) != tinyyolo::Status::Ok) {
    fprintf(stderr, "camera_view: model setup failed; showing the camera only\n");
  }
  if (!serial_port.empty()) {
    if (!openSerial(serial_port)) {
      fprintf(stderr, "camera_view: cannot open %s\n", serial_port.c_str());
      exit(1);
    }
    return;  // the board does the detection; the camera is not used
  }
  if (model.isReady()) recognition_task.begin(recognize);

  camera_config_t config = tiny_camera::TinyCamera::defaultConfig();
  config.frame_size = FRAMESIZE_QVGA;
  config.device = camera_device;
  if (!camera_frame.begin(config)) {
    fprintf(stderr, "camera_view: cannot start the camera\n");
    exit(1);
  }
}

void loop() {
  if (!serial_port.empty()) {
    serialStep();
    app.update();
    return;
  }
  // The display: every camera frame is shown and queued for recognition.
  if (camera_frame.next()) {
    const tinyyolo::ImageView picture = camera_frame.image();
    if (model.isReady()) pushFrame(picture);
    view.setPicture(picture);
  }
  // Results that the worker has finished since the last loop.
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    if (has_new_results) {
      view.setDetections(latest_results.data(), (int)latest_results.size());
      has_new_results = false;
    }
  }
  // Redraws only when the picture or the results changed. Closing the window exits.
  app.update();
}

int main(int argc, char** argv) {
  std::string model_base;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      model_base = argv[++i];
    } else if (arg == "--device" && i + 1 < argc) {
      camera_device = argv[++i];
    } else if (arg == "--serial" && i + 1 < argc) {
      serial_port = argv[++i];
    } else if (arg == "--rotate" && i + 1 < argc) {
      serial_rotation = atoi(argv[++i]);
      if (serial_rotation != 0 && serial_rotation != 90 && serial_rotation != 180 &&
          serial_rotation != 270) {
        fprintf(stderr, "camera_view: --rotate must be 0, 90, 180 or 270\n");
        return 2;
      }
    } else {
      fprintf(stderr, "usage: %s [--model <base>] [--device <path>] [--serial <port>] [--rotate <deg>]\n", argv[0]);
      return 2;
    }
  }
  if (!model_base.empty()) {
    if (!file_source.load(model_base)) return 1;
    model_source = &file_source;
  }
  setup();
  for (;;) loop();
}
