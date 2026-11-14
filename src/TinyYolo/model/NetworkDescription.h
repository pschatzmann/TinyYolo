#pragma once
/// @file NetworkDescription.h  Builds a NetworkSpec from a Darknet .cfg (and .names) file.
///
/// Works with any Stream: File on an SD card, or BufferStream over a model embedded in
/// flash. The description owns all the storage the NetworkSpec points to, so it must
/// outlive the Detector that uses spec().
///
/// Supported sections: [net], [convolutional] (leaky, linear, relu), [maxpool],
/// [upsample], [route] (one or two inputs), [yolo]. Anything else is rejected with an error.

#include <stdlib.h>
#include <string.h>

#include "TinyYolo/core/Logger.h"
#include "TinyYolo/core/Status.h"
#include "TinyYolo/core/Vector.h"
#include "TinyYolo/model/LayerSpec.h"
#include "TinyYolo/platform/Io.h"

namespace tinyyolo {

class NetworkDescription {
 public:
  NetworkDescription() {
    spec_ = NetworkSpec{0, 0, 0, nullptr, 0, nullptr, 0};
  }
  NetworkDescription(const NetworkDescription&) = delete;
  NetworkDescription& operator=(const NetworkDescription&) = delete;

  /// Parses a Darknet .cfg. Replaces any previously parsed network.
  Status parseCfg(Stream& stream) {
    reset();
    Section section;
    bool in_section = false;
    int line_no = 0;
    char line[kMaxLine];
    for (;;) {
      size_t length = 0;
      bool truncated = false;
      if (!readLine(stream, line, sizeof(line), length, truncated)) break;
      ++line_no;
      if (truncated) {
        TINYYOLO_LOGW("Darknet", "cfg line %d is longer than %d characters and was cut", line_no,
                      (int)kMaxLine - 1);
      }
      char* text = trim(line);
      if (*text == 0 || *text == '#' || *text == ';') continue;

      if (*text == '[') {
        if (in_section) TINYYOLO_CHECK(finishSection(section, line_no));
        TINYYOLO_CHECK(startSection(section, text, line_no));
        in_section = true;
        continue;
      }
      if (!in_section) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: key/value pair outside of a [section]", line_no);
        return Status::InvalidModel;
      }
      char* equals = strchr(text, '=');
      if (equals == nullptr) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: expected key=value, got '%s'", line_no, text);
        return Status::InvalidModel;
      }
      *equals = 0;
      TINYYOLO_CHECK(applyKey(section, trim(text), trim(equals + 1), line_no));
    }
    if (in_section) TINYYOLO_CHECK(finishSection(section, line_no));
    if (!net_seen_) {
      TINYYOLO_LOGE("Darknet", "cfg has no [net] section");
      return Status::InvalidModel;
    }
    if (layers_.empty()) {
      TINYYOLO_LOGE("Darknet", "cfg defines no layers");
      return Status::InvalidModel;
    }
    TINYYOLO_CHECK(finalize());
    valid_ = true;
    TINYYOLO_LOGI("Darknet", "cfg parsed: %dx%dx%d input, %d layers", spec_.input_w, spec_.input_h,
                  spec_.input_c, spec_.layer_count);
    return Status::Ok;
  }

  /// Parses a Darknet .names file: one label per line. Call after parseCfg().
  Status parseNames(Stream& stream) {
    label_text_.clear();
    label_offsets_.clear();
    label_pointers_.clear();
    char line[kMaxLine];
    for (;;) {
      size_t length = 0;
      bool truncated = false;
      if (!readLine(stream, line, sizeof(line), length, truncated)) break;
      char* text = trim(line);
      if (*text == 0) continue;
      TINYYOLO_CHECK(label_offsets_.push_back(label_text_.size()));
      for (const char* c = text; *c != 0; ++c) TINYYOLO_CHECK(label_text_.push_back(*c));
      TINYYOLO_CHECK(label_text_.push_back(0));
    }
    for (size_t i = 0; i < label_offsets_.size(); ++i) {
      TINYYOLO_CHECK(label_pointers_.push_back(label_text_.data() + label_offsets_[i]));
    }
    if (label_pointers_.empty()) {
      TINYYOLO_LOGW("Darknet", "names file has no labels");
    }
    spec_.labels = label_pointers_.empty() ? nullptr : label_pointers_.data();
    spec_.label_count = (int)label_pointers_.size();
    return Status::Ok;
  }

  /// The parsed network. Valid after a successful parseCfg().
  const NetworkSpec& spec() const { return spec_; }
  bool isValid() const { return valid_; }

 private:
  static constexpr int kMaxLine = 160;
  static constexpr int kMaxList = 16;     // route / mask entries
  static constexpr int kMaxAnchors = 32;  // floats (16 pairs)

  enum class Kind : uint8_t { None, Net, Convolutional, MaxPool, Upsample, Route, Yolo, Shortcut };

  struct Section {
    Kind kind;
    int line;
    int width, height, channels;     // [net]
    int filters, size, stride, pad;  // [convolutional] / [maxpool] / [upsample]
    int padding;                     // explicit padding, -1 = not given
    int batch_normalize, groups;
    char activation[16];
    int list[kMaxList];              // [route] layers / [yolo] mask
    int list_count;
    float anchors[kMaxAnchors];      // [yolo]
    int anchor_count;
    int classes;
    int group_id;
    float scale_xy;
  };

  void reset() {
    layers_.clear();
    anchor_pool_.clear();
    anchor_offsets_.clear();
    net_seen_ = false;
    valid_ = false;
    spec_ = NetworkSpec{0, 0, 0, nullptr, 0, spec_.labels, spec_.label_count};
  }

  static void resetSection(Section& s, Kind kind, int line) {
    memset(&s, 0, sizeof(s));
    s.kind = kind;
    s.line = line;
    s.padding = -1;
    s.stride = kind == Kind::Upsample ? 2 : 1;
    s.scale_xy = 1.0f;
  }

  Status startSection(Section& section, char* text, int line) {
    char* end = strchr(text, ']');
    if (end == nullptr) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: section header without ']'", line);
      return Status::InvalidModel;
    }
    *end = 0;
    const char* name = trim(text + 1);
    Kind kind = Kind::None;
    if (streq(name, "net") || streq(name, "network")) kind = Kind::Net;
    else if (streq(name, "convolutional")) kind = Kind::Convolutional;
    else if (streq(name, "maxpool")) kind = Kind::MaxPool;
    else if (streq(name, "upsample")) kind = Kind::Upsample;
    else if (streq(name, "route")) kind = Kind::Route;
    else if (streq(name, "yolo")) kind = Kind::Yolo;
    else if (streq(name, "shortcut")) kind = Kind::Shortcut;

    if (kind == Kind::None) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: section [%s] is not supported", line, name);
      return Status::Unsupported;
    }
    if (kind == Kind::Net) {
      if (net_seen_) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: second [net] section", line);
        return Status::InvalidModel;
      }
      net_seen_ = true;
    } else if (!net_seen_) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: [%s] appears before [net]", line, name);
      return Status::InvalidModel;
    }
    resetSection(section, kind, line);
    return Status::Ok;
  }

  Status applyKey(Section& s, const char* key, const char* value, int line) {
    if (s.kind == Kind::Net) {
      if (streq(key, "width")) return intValue(value, s.width, key, line);
      if (streq(key, "height")) return intValue(value, s.height, key, line);
      if (streq(key, "channels")) return intValue(value, s.channels, key, line);
      return Status::Ok;
    }
    if (streq(key, "filters")) return intValue(value, s.filters, key, line);
    if (streq(key, "size")) return intValue(value, s.size, key, line);
    if (streq(key, "stride")) return intValue(value, s.stride, key, line);
    if (streq(key, "pad")) return intValue(value, s.pad, key, line);
    if (streq(key, "padding")) return intValue(value, s.padding, key, line);
    if (streq(key, "batch_normalize")) return intValue(value, s.batch_normalize, key, line);
    if (streq(key, "groups")) return intValue(value, s.groups, key, line);
    if (streq(key, "group_id")) return intValue(value, s.group_id, key, line);
    if (streq(key, "classes")) return intValue(value, s.classes, key, line);
    if (streq(key, "activation")) {
      strncpy(s.activation, value, sizeof(s.activation) - 1);
      s.activation[sizeof(s.activation) - 1] = 0;
      return Status::Ok;
    }
    if (streq(key, "scale_x_y")) {
      s.scale_xy = (float)strtod(value, nullptr);
      return Status::Ok;
    }
    if (streq(key, "layers") || streq(key, "mask") || streq(key, "from")) {
      return intList(value, s, key, line);
    }
    if (streq(key, "anchors")) return anchorList(value, s, line);
    return Status::Ok;  // unknown keys (learning rate, ...) do not affect inference
  }

  Status finishSection(const Section& s, int line) {
    switch (s.kind) {
      case Kind::Net:
        if (s.width <= 0 || s.height <= 0 || s.channels <= 0) {
          TINYYOLO_LOGE("Darknet", "cfg line %d: [net] needs width, height and channels", line);
          return Status::InvalidModel;
        }
        spec_.input_w = s.width;
        spec_.input_h = s.height;
        spec_.input_c = s.channels;
        return Status::Ok;
      case Kind::Convolutional:
        return emitConvolutional(s, line);
      case Kind::MaxPool:
        return emitMaxPool(s, line);
      case Kind::Upsample:
        return addLayer(LayerSpec::upsample(s.stride), -1);
      case Kind::Route:
        return emitRoute(s, line);
      case Kind::Yolo:
        return emitYolo(s, line);
      case Kind::Shortcut:
        return emitShortcut(s, line);
      case Kind::None:
        break;
    }
    return Status::Ok;
  }

  Status emitConvolutional(const Section& s, int line) {
    if (s.filters <= 0 || s.size <= 0 || s.stride <= 0) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: convolutional needs filters, size and stride > 0", line);
      return Status::InvalidModel;
    }
    if (s.groups > 1) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: grouped convolution (groups=%d) is not supported", line,
                    s.groups);
      return Status::Unsupported;
    }
    Activation act;
    if (!activationFrom(s.activation, act)) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: activation '%s' is not supported", line, s.activation);
      return Status::Unsupported;
    }
    // Darknet: pad=1 means "same" padding (size/2); otherwise the padding key (default 0).
    const int pad = s.pad != 0 ? s.size / 2 : (s.padding > 0 ? s.padding : 0);
    return addLayer(LayerSpec::conv(s.filters, s.size, s.stride, act, pad, s.batch_normalize != 0), -1);
  }

  /// Maps a Darknet activation name. An empty name means linear (Darknet's shortcut default).
  static bool activationFrom(const char* name, Activation& act) {
    if (name[0] == 0 || streq(name, "linear")) act = Activation::Linear;
    else if (streq(name, "leaky")) act = Activation::Leaky;
    else if (streq(name, "relu")) act = Activation::ReLU;
    else if (streq(name, "mish")) act = Activation::Mish;
    else return false;
    return true;
  }

  /// [shortcut]: adds the previous layer and the layer named by `from` (relative or absolute).
  Status emitShortcut(const Section& s, int line) {
    if (s.list_count != 1) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: shortcut needs exactly one 'from' layer", line);
      return Status::InvalidModel;
    }
    Activation act;
    if (!activationFrom(s.activation, act)) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: shortcut activation '%s' is not supported", line,
                    s.activation);
      return Status::Unsupported;
    }
    const int current = (int)layers_.size();
    const int other = s.list[0] < 0 ? current + s.list[0] : s.list[0];
    return addLayer(LayerSpec::add(kPrevious, other, act), -1);
  }

  Status emitMaxPool(const Section& s, int line) {
    if (s.size <= 0 || s.stride <= 0) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: maxpool needs size and stride > 0", line);
      return Status::InvalidModel;
    }
    // Darknet default: total padding = size - 1 (so stride 1 keeps the resolution).
    const int pad_total = s.padding >= 0 ? s.padding : s.size - 1;
    return addLayer(LayerSpec::maxPool(s.size, s.stride, pad_total), -1);
  }

  Status emitRoute(const Section& s, int line) {
    if (s.groups > 1) {
      // Grouped route: one channel group of a single layer.
      if (s.list_count != 1 || s.group_id < 0 || s.group_id >= s.groups) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: grouped route needs one layer and 0 <= group_id < groups",
                      line);
        return Status::InvalidModel;
      }
      const int current = (int)layers_.size();
      const int from = s.list[0] < 0 ? current + s.list[0] : s.list[0];
      return addLayer(LayerSpec::split(from, s.groups, s.group_id), -1);
    }
    if (s.list_count < 1 || s.list_count > 2) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: route with %d inputs is not supported (use 1 or 2)",
                    line, s.list_count);
      return Status::Unsupported;
    }
    const int current = (int)layers_.size();
    int first = s.list[0] < 0 ? current + s.list[0] : s.list[0];
    if (s.list_count == 1) return addLayer(LayerSpec::route(first), -1);
    int second = s.list[1] < 0 ? current + s.list[1] : s.list[1];
    return addLayer(LayerSpec::route(first, second), -1);
  }

  Status emitYolo(const Section& s, int line) {
    if (s.classes <= 0 || s.list_count <= 0) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: yolo needs classes > 0 and a mask", line);
      return Status::InvalidModel;
    }
    const int anchor_pairs = s.anchor_count / 2;
    const int offset = (int)anchor_pool_.size();
    for (int k = 0; k < s.list_count; ++k) {
      const int m = s.list[k];
      if (m < 0 || m >= anchor_pairs) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: mask entry %d selects anchor %d of %d", line, k, m,
                      anchor_pairs);
        return Status::InvalidModel;
      }
      TINYYOLO_CHECK(anchor_pool_.push_back(s.anchors[2 * m]));
      TINYYOLO_CHECK(anchor_pool_.push_back(s.anchors[2 * m + 1]));
    }
    return addLayer(LayerSpec::yolo(s.classes, nullptr, s.list_count, YoloDecode::Darknet, s.scale_xy),
                    offset);
  }

  /// Appends a layer. `anchor_offset` is the offset into anchor_pool_, or -1.
  Status addLayer(const LayerSpec& spec, int anchor_offset) {
    TINYYOLO_CHECK(layers_.push_back(spec));
    return anchor_offsets_.push_back(anchor_offset);
  }

  /// Points the spec at its storage once every section has been read.
  Status finalize() {
    for (size_t i = 0; i < layers_.size(); ++i) {
      if (anchor_offsets_[i] >= 0) layers_[i].anchors = anchor_pool_.data() + anchor_offsets_[i];
    }
    spec_.layers = layers_.data();
    spec_.layer_count = (int)layers_.size();
    return Status::Ok;
  }

  // ----------------------------------------------------------------- value parsing

  static bool streq(const char* a, const char* b) { return strcmp(a, b) == 0; }

  Status intValue(const char* value, int& out, const char* key, int line) {
    char* end = nullptr;
    const long v = strtol(value, &end, 10);
    if (end == value || *trim(end) != 0) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: '%s' must be an integer, got '%s'", line, key, value);
      return Status::InvalidModel;
    }
    out = (int)v;
    return Status::Ok;
  }

  Status intList(const char* value, Section& s, const char* key, int line) {
    s.list_count = 0;
    const char* p = value;
    while (*p != 0) {
      char* end = nullptr;
      const long v = strtol(p, &end, 10);
      if (end == p) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: '%s' has a non-integer entry in '%s'", line, key,
                      value);
        return Status::InvalidModel;
      }
      if (s.list_count >= kMaxList) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: '%s' has more than %d entries", line, key, kMaxList);
        return Status::Unsupported;
      }
      s.list[s.list_count++] = (int)v;
      p = end;
      while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    }
    return Status::Ok;
  }

  Status anchorList(const char* value, Section& s, int line) {
    s.anchor_count = 0;
    const char* p = value;
    while (*p != 0) {
      char* end = nullptr;
      const double v = strtod(p, &end);
      if (end == p) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: anchors has a non-number in '%s'", line, value);
        return Status::InvalidModel;
      }
      if (s.anchor_count >= kMaxAnchors) {
        TINYYOLO_LOGE("Darknet", "cfg line %d: more than %d anchor values", line, kMaxAnchors);
        return Status::Unsupported;
      }
      s.anchors[s.anchor_count++] = (float)v;
      p = end;
      while (*p == ',' || *p == ' ' || *p == '\t') ++p;
    }
    if (s.anchor_count % 2 != 0) {
      TINYYOLO_LOGE("Darknet", "cfg line %d: anchors must come in width,height pairs", line);
      return Status::InvalidModel;
    }
    return Status::Ok;
  }

  // ----------------------------------------------------------------- text handling

  /// Reads one line (without the terminator). Returns false at end of stream.
  static bool readLine(Stream& stream, char* buffer, size_t capacity, size_t& length,
                       bool& truncated) {
    length = 0;
    bool any = false;
    char c;
    while (stream.readBytes(&c, 1) == 1) {
      any = true;
      if (c == '\n') break;
      if (c == '\r') continue;
      if (length + 1 < capacity) {
        buffer[length++] = c;
      } else {
        truncated = true;
      }
    }
    buffer[length] = 0;
    return any;
  }

  static char* trim(char* s) {
    while (*s == ' ' || *s == '\t') ++s;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = 0;
    return s;
  }

  NetworkSpec spec_;
  Vector<LayerSpec> layers_;
  Vector<float> anchor_pool_;
  Vector<int> anchor_offsets_;  ///< per layer: offset into anchor_pool_, or -1
  Vector<char> label_text_;
  Vector<size_t> label_offsets_;
  Vector<const char*> label_pointers_;
  bool net_seen_ = false;
  bool valid_ = false;
};

}  // namespace tinyyolo
