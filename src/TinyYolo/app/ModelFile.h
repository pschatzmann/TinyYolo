#pragma once
/// @file ModelFile.h  Reader for the single-file model format (docs/ModelFormat.md).
///
/// The file holds a whole model: the graph, the quantization, the int8 weights, optionally the
/// float weights, and the class names. The reader checks the file (magic, version, section
/// bounds, CRC32, and every index of the graph) before it returns a GraphModel. The weights are
/// used in place: the pointers refer to the bytes passed to begin(), which must stay valid while
/// the model is used. The graph records are converted into arrays owned by the reader.
///
/// The weights are read in place, so the host must be little-endian (ESP32, RP2040, x86, ARM).

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "TinyYolo/app/GraphYolo11.h"
#include "TinyYolo/core/Status.h"

namespace tinyyolo {
namespace tym {

/// Section types (docs/ModelFormat.md).
enum SectionType : uint32_t {
  kInfo = 1,
  kOps = 2,
  kParams = 3,
  kQuant = 4,
  kDetect = 5,
  kInt8Weights = 6,
  kInt8Scales = 7,
  kFloatWeights = 8,
  kNames = 9,
  kLayerOutputs = 10,  // optional: the tensor of each export layer (for the checks)
};

const uint32_t kFlagHasFloat = 1u << 0;
const uint32_t kFlagPerTensorQuant = 1u << 1;
const uint8_t kMajorVersion = 1;
const uint8_t kMinorVersion = 0;
const size_t kOpRecordBytes = 96;
const size_t kParamRecordBytes = 24;
const size_t kQuantRecordBytes = 16;
const size_t kDetectLevelBytes = 12;
const size_t kSectionAlign = 16;
const size_t kMaxDetectLevels = 3;

/// CRC32 (IEEE, as used by zlib), of data[0..size).
inline uint32_t crc32(const uint8_t* data, size_t size) {
  static uint32_t table[256];
  static bool built = false;
  if (!built) {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
      table[i] = c;
    }
    built = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
  return crc ^ 0xFFFFFFFFu;
}

/// Little-endian readers (the file's byte order, whatever the host).
inline uint32_t readU32(const uint8_t* p) {
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
inline int32_t readI32(const uint8_t* p) { return (int32_t)readU32(p); }

/// A model read from a .tym file. Call begin() once, then use model() and the weight pointers.
class ModelFile {
 public:
  /// Checks the bytes and sets the model up. The bytes must outlive this object.
  Status begin(const uint8_t* data, size_t size) {
    ready_ = false;
    if (data == nullptr || size < 24 + 4) return Status::InvalidArgument;
    if (memcmp(data, "TYM1", 4) != 0) return Status::InvalidModel;
    if (data[4] != kMajorVersion) return Status::Unsupported;  // a different major version
    const size_t header_size = readU32(data + 8);
    const uint32_t flags = readU32(data + 12);
    const uint32_t section_count = readU32(data + 16);
    if (section_count > 16 || header_size != 20 + 12 * (size_t)section_count || header_size + 4 > size) {
      return Status::InvalidModel;
    }
    // The CRC covers everything before the last four bytes.
    const size_t body = size - 4;
    if (crc32(data, body) != readU32(data + body)) return Status::InvalidModel;

    // The section table: the offset and size of each section, each on a 16-byte boundary.
    Section sections[16];
    for (uint32_t i = 0; i < section_count; ++i) {
      const uint8_t* entry = data + 20 + 12 * i;
      sections[i] = {readU32(entry), readU32(entry + 4), readU32(entry + 8)};
      if (sections[i].offset % kSectionAlign != 0 || sections[i].offset < header_size ||
          sections[i].offset > body || sections[i].size > body - sections[i].offset) {
        return Status::InvalidModel;
      }
    }
    const Section* info = find(sections, section_count, kInfo);
    const Section* ops = find(sections, section_count, kOps);
    const Section* params = find(sections, section_count, kParams);
    const Section* quant = find(sections, section_count, kQuant);
    const Section* detect = find(sections, section_count, kDetect);
    const Section* int8 = find(sections, section_count, kInt8Weights);
    const Section* scales = find(sections, section_count, kInt8Scales);
    const Section* floats = find(sections, section_count, kFloatWeights);
    const Section* names = find(sections, section_count, kNames);
    const Section* layer_outputs = find(sections, section_count, kLayerOutputs);
    if (!info || !ops || !params || !quant || !detect || !int8 || !scales || !names) return Status::InvalidModel;
    if (((flags & kFlagHasFloat) != 0) != (floats != nullptr)) return Status::InvalidModel;

    // INFO: name, input size, tensor count, input tensor, op, param and quant counts.
    size_t at = info->offset;
    const size_t info_end = info->offset + info->size;
    if (at + 1 > info_end) return Status::InvalidModel;
    const size_t name_len = data[at];
    at += 1;
    if (at + name_len + 24 > info_end) return Status::InvalidModel;
    name_.assign((const char*)data + at, name_len);
    at += name_len;
    const int input_size = (int)readU32(data + at);
    const int tensor_count = (int)readU32(data + at + 4);
    const int input_id = (int)readU32(data + at + 8);
    const int op_count = (int)readU32(data + at + 12);
    const int param_count = (int)readU32(data + at + 16);
    const int quant_count = (int)readU32(data + at + 20);
    if (input_size <= 0 || tensor_count <= 0 || op_count <= 0 || param_count < 0 || quant_count < 0) {
      return Status::InvalidModel;
    }
    if (input_id < 0 || input_id >= tensor_count) return Status::InvalidModel;

    // The graph records.
    if (ops->size != (size_t)op_count * kOpRecordBytes) return Status::InvalidModel;
    if (params->size != (size_t)param_count * kParamRecordBytes) return Status::InvalidModel;
    if (quant->size != (size_t)quant_count * kQuantRecordBytes) return Status::InvalidModel;
    ops_.resize((size_t)op_count);
    for (int i = 0; i < op_count; ++i) {
      if (!readOp(data + ops->offset + (size_t)i * kOpRecordBytes, ops_[i], tensor_count, param_count, quant_count)) {
        return Status::InvalidModel;
      }
    }
    params_.resize((size_t)param_count);
    for (int i = 0; i < param_count; ++i) {
      const uint8_t* p = data + params->offset + (size_t)i * kParamRecordBytes;
      GraphParam& param = params_[i];
      param.offset = readI32(p);
      param.rank = (int)readU32(p + 4);
      for (int d = 0; d < 4; ++d) param.dims[d] = readI32(p + 8 + 4 * d);
      if (param.rank < 0 || param.rank > 4 || param.offset < 0) return Status::InvalidModel;
    }
    quant_.resize((size_t)quant_count);
    for (int i = 0; i < quant_count; ++i) {
      const uint8_t* p = data + quant->offset + (size_t)i * kQuantRecordBytes;
      GraphQuant& q = quant_[i];
      q.w_offset = readI32(p);
      q.s_offset = readI32(p + 4);
      q.in_scale = floatAt(p + 8);
      q.out_scale = floatAt(p + 12);
    }

    // The detection heads.
    if (detect->size < 12) return Status::InvalidModel;
    const int levels = (int)readU32(data + detect->offset);
    const int classes = (int)readU32(data + detect->offset + 4);
    const int reg_max = (int)readU32(data + detect->offset + 8);
    if (levels < 1 || levels > (int)kMaxDetectLevels || classes <= 0 || reg_max <= 0) return Status::InvalidModel;
    if (detect->size != 12 + (size_t)levels * kDetectLevelBytes) return Status::InvalidModel;
    heads_.resize((size_t)levels * 2);
    strides_.resize((size_t)levels);
    for (int l = 0; l < levels; ++l) {
      const uint8_t* p = data + detect->offset + 12 + (size_t)l * kDetectLevelBytes;
      heads_[2 * l] = readI32(p);
      heads_[2 * l + 1] = readI32(p + 4);
      strides_[l] = readI32(p + 8);
      if (heads_[2 * l] < 0 || heads_[2 * l] >= tensor_count || heads_[2 * l + 1] < 0 ||
          heads_[2 * l + 1] >= tensor_count || strides_[l] <= 0) {
        return Status::InvalidModel;
      }
    }

    // The weights, in place.
    if (int8->size == 0 || scales->size % 4 != 0 || scales->size == 0) return Status::InvalidModel;
    int8_ = data + int8->offset;
    int8_count_ = int8->size;
    scales_ = (const float*)(data + scales->offset);
    scale_count_ = scales->size / 4;
    if (floats) {
      if (floats->size % 4 != 0) return Status::InvalidModel;
      floats_ = (const float*)(data + floats->offset);
      float_count_ = floats->size / 4;
    }

    // The class names.
    if (names->size < 4) return Status::InvalidModel;
    const size_t name_count = readU32(data + names->offset);
    if ((int)name_count != classes) return Status::InvalidModel;
    names_.clear();
    size_t p = names->offset + 4;
    const size_t names_end = names->offset + names->size;
    for (size_t i = 0; i < name_count; ++i) {
      if (p + 1 > names_end) return Status::InvalidModel;
      const size_t len = data[p];
      if (p + 1 + len > names_end) return Status::InvalidModel;
      names_.push_back(std::make_pair(p + 1, len));
      p += 1 + len;
    }

    // The layer outputs (optional): the number of pairs, then {layer, tensor id} pairs.
    outputs_.clear();
    if (layer_outputs) {
      if (layer_outputs->size < 4 || (layer_outputs->size - 4) % 8 != 0) return Status::InvalidModel;
      const size_t count = readU32(data + layer_outputs->offset);
      if (layer_outputs->size != 4 + 8 * count) return Status::InvalidModel;
      for (size_t i = 0; i < count; ++i) {
        const uint8_t* p = data + layer_outputs->offset + 4 + 8 * i;
        const int tensor = readI32(p + 4);
        if (tensor < 0 || tensor >= tensor_count) return Status::InvalidModel;
        outputs_.push_back(std::make_pair(readI32(p), tensor));
      }
    }

    // The model description for GraphYolo11.
    model_ = GraphModel();
    model_.ops = ops_.data();
    model_.op_count = op_count;
    model_.params = params_.data();
    model_.param_count = param_count;
    model_.tensor_count = tensor_count;
    model_.input_id = input_id;
    model_.quant = quant_.empty() ? nullptr : quant_.data();
    model_.detect_heads = (const int(*)[2])heads_.data();
    model_.detect_strides = strides_.data();
    model_.detect_levels = levels;
    model_.detect_classes = classes;
    model_.detect_reg_max = reg_max;
    model_.input_size = input_size;
    per_tensor_quant_ = (flags & kFlagPerTensorQuant) != 0;
    data_ = data;
    size_ = size;
    ready_ = true;
    return Status::Ok;
  }

  /// The length of the file that starts at `data`, from its section table, or 0 when the header
  /// is not a model file or the file does not fit into `available` bytes. Used when the file sits
  /// in a larger flash partition.
  static size_t fileSize(const uint8_t* data, size_t available) {
    if (data == nullptr || available < 24 + 4 || memcmp(data, "TYM1", 4) != 0) return 0;
    const size_t header_size = readU32(data + 8);
    const uint32_t section_count = readU32(data + 16);
    if (section_count > 16 || header_size != 20 + 12 * (size_t)section_count || header_size > available) return 0;
    size_t end = header_size;
    for (uint32_t i = 0; i < section_count; ++i) {
      const size_t offset = readU32(data + 20 + 12 * i + 4);
      const size_t size = readU32(data + 20 + 12 * i + 8);
      if (offset > available || size > available - offset) return 0;
      if (offset + size > end) end = offset + size;
    }
    const size_t total = end + 4;  // the CRC32
    return total <= available ? total : 0;
  }

  bool ready() const { return ready_; }
  const GraphModel& model() const { return model_; }
  /// The tensor of each export layer, as {layer, tensor id} (empty when the file has no such section).
  const std::vector<std::pair<int, int>>& layerOutputs() const { return outputs_; }
  const std::string& name() const { return name_; }
  bool hasFloatWeights() const { return floats_ != nullptr; }
  bool perTensorQuant() const { return per_tensor_quant_; }
  const float* floatWeights() const { return floats_; }
  size_t floatCount() const { return float_count_; }
  const int8_t* int8Weights() const { return (const int8_t*)int8_; }
  size_t int8Count() const { return int8_count_; }
  const float* int8Scales() const { return scales_; }
  size_t scaleCount() const { return scale_count_; }
  int classCount() const { return (int)names_.size(); }
  /// The name of a class, or "" for an id out of range.
  std::string className(int class_id) const {
    if (class_id < 0 || class_id >= (int)names_.size()) return std::string();
    return std::string((const char*)data_ + names_[class_id].first, names_[class_id].second);
  }

 private:
  struct Section {
    uint32_t type;
    size_t offset;
    size_t size;
  };

  static const Section* find(const Section* sections, uint32_t count, uint32_t type) {
    for (uint32_t i = 0; i < count; ++i) {
      if (sections[i].type == type) return &sections[i];
    }
    return nullptr;
  }

  static float floatAt(const uint8_t* p) {
    const uint32_t bits = readU32(p);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
  }

  /// Converts one OPS record. Returns false when an index is out of range.
  static bool readOp(const uint8_t* p, GraphOp& op, int tensor_count, int param_count, int quant_count) {
    const uint8_t kind = p[0];
    const uint8_t n_in = p[1];
    const uint8_t flags = p[2];
    if (kind > (uint8_t)GraphOpKind::Attention || n_in > 4) return false;
    op.kind = (GraphOpKind)kind;
    op.n_in = n_in;
    op.silu = (flags & 1) != 0;
    for (int i = 0; i < 4; ++i) op.in[i] = readI32(p + 4 + 4 * i);
    op.out = readI32(p + 20);
    op.w = readI32(p + 24);
    op.b = readI32(p + 28);
    op.k = readI32(p + 32);
    op.s = readI32(p + 36);
    op.p = readI32(p + 40);
    op.g = readI32(p + 44);
    op.index = readI32(p + 48);
    op.w2 = readI32(p + 52);
    op.b2 = readI32(p + 56);
    op.w3 = readI32(p + 60);
    op.b3 = readI32(p + 64);
    op.heads = readI32(p + 68);
    op.key_dim = readI32(p + 72);
    op.head_dim = readI32(p + 76);
    op.q = readI32(p + 80);
    op.q2 = readI32(p + 84);
    op.q3 = readI32(p + 88);
    for (int i = 0; i < n_in; ++i) {
      if (op.in[i] < 0 || op.in[i] >= tensor_count) return false;
    }
    if (op.out < 0 || op.out >= tensor_count) return false;
    const int params[] = {op.w, op.b, op.w2, op.b2, op.w3, op.b3};
    for (int id : params) {
      if (id < -1 || id >= param_count) return false;
    }
    const int quants[] = {op.q, op.q2, op.q3};
    for (int id : quants) {
      if (id < -1 || id >= quant_count) return false;
    }
    return true;
  }

  bool ready_ = false;
  bool per_tensor_quant_ = false;
  const uint8_t* data_ = nullptr;
  size_t size_ = 0;
  std::string name_;
  std::vector<GraphOp> ops_;
  std::vector<GraphParam> params_;
  std::vector<GraphQuant> quant_;
  std::vector<int> heads_;
  std::vector<int> strides_;
  std::vector<std::pair<size_t, size_t>> names_;  // offset and length of each class name
  std::vector<std::pair<int, int>> outputs_;      // {layer, tensor id}
  const uint8_t* int8_ = nullptr;
  size_t int8_count_ = 0;
  const float* scales_ = nullptr;
  size_t scale_count_ = 0;
  const float* floats_ = nullptr;
  size_t float_count_ = 0;
  GraphModel model_;
};

}  // namespace tym
}  // namespace tinyyolo
