#pragma once
/// @file ModelFileWriter.h  Writer for the single-file model format (docs/ModelFormat.md).
///
/// Used by the desktop tools and the tests. It lays the sections out as the reader expects:
/// each section starts on a 16-byte boundary, and the file ends with a CRC32 of everything before it.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <string>
#include <vector>

#include "TinyYolo/app/ModelFile.h"
#include "TinyYolo/app/GraphYolo11.h"

namespace tinyyolo {
namespace tym {

/// Everything that goes into a file. The weight pointers may be null (the float weights are optional).
struct Content {
  std::string name;  ///< at most 255 bytes
  int input_size = 640;
  int tensor_count = 0;
  int input_id = 0;
  std::vector<GraphOp> ops;
  std::vector<GraphParam> params;
  std::vector<GraphQuant> quant;
  bool per_tensor_quant = false;
  std::vector<int> detect_heads;   ///< box and class tensor id of each level, two per level
  std::vector<int> detect_strides; ///< one per level
  int detect_classes = 80;
  int detect_reg_max = 16;
  std::vector<std::string> names;  ///< one per class
  std::vector<std::pair<int, int>> layer_outputs;  ///< {layer, tensor id}; optional section
  const float* floats = nullptr;   ///< float parameters, or null (no FLOAT_WEIGHTS section)
  size_t float_count = 0;
  const int8_t* int8 = nullptr;    ///< int8 convolution weights
  size_t int8_count = 0;
  const float* scales = nullptr;   ///< per-output-channel weight scales
  size_t scale_count = 0;
};

namespace detail {

inline void putU32(std::vector<uint8_t>& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back((uint8_t)(v >> (8 * i)));
}
inline void putI32(std::vector<uint8_t>& out, int32_t v) { putU32(out, (uint32_t)v); }
inline void putF32(std::vector<uint8_t>& out, float v) {
  uint32_t bits;
  memcpy(&bits, &v, sizeof(bits));
  putU32(out, bits);
}
inline void padTo(std::vector<uint8_t>& out, size_t align) {
  while (out.size() % align != 0) out.push_back(0);
}

inline std::vector<uint8_t> encodeOps(const std::vector<GraphOp>& ops) {
  std::vector<uint8_t> out;
  for (const GraphOp& op : ops) {
    out.push_back((uint8_t)op.kind);
    out.push_back(op.n_in);
    out.push_back(op.silu ? 1 : 0);
    out.push_back(0);
    for (int i = 0; i < 4; ++i) putI32(out, op.in[i]);
    const int fields[] = {op.out, op.w, op.b, op.k, op.s, op.p, op.g, op.index, op.w2, op.b2, op.w3,
                          op.b3, op.heads, op.key_dim, op.head_dim, op.q, op.q2, op.q3};
    for (int v : fields) putI32(out, v);
    putU32(out, 0);  // reserved
  }
  return out;
}

}  // namespace detail

/// The file bytes for a model.
inline std::vector<uint8_t> write(const Content& c) {
  using namespace detail;
  std::vector<std::pair<uint32_t, std::vector<uint8_t>>> sections;

  // INFO
  std::vector<uint8_t> info;
  info.push_back((uint8_t)(c.name.size() > 255 ? 255 : c.name.size()));
  info.insert(info.end(), c.name.begin(), c.name.begin() + (c.name.size() > 255 ? 255 : c.name.size()));
  putU32(info, (uint32_t)c.input_size);
  putU32(info, (uint32_t)c.tensor_count);
  putU32(info, (uint32_t)c.input_id);
  putU32(info, (uint32_t)c.ops.size());
  putU32(info, (uint32_t)c.params.size());
  putU32(info, (uint32_t)c.quant.size());
  sections.push_back(std::make_pair((uint32_t)kInfo, info));

  // OPS, PARAMS, QUANT
  sections.push_back(std::make_pair((uint32_t)kOps, encodeOps(c.ops)));
  std::vector<uint8_t> params;
  for (const GraphParam& p : c.params) {
    putI32(params, p.offset);
    putU32(params, (uint32_t)p.rank);
    for (int d = 0; d < 4; ++d) putI32(params, p.dims[d]);
  }
  sections.push_back(std::make_pair((uint32_t)kParams, params));
  std::vector<uint8_t> quant;
  for (const GraphQuant& q : c.quant) {
    putI32(quant, q.w_offset);
    putI32(quant, q.s_offset);
    putF32(quant, q.in_scale);
    putF32(quant, q.out_scale);
  }
  sections.push_back(std::make_pair((uint32_t)kQuant, quant));

  // DETECT
  std::vector<uint8_t> detect;
  const int levels = (int)c.detect_strides.size();
  putU32(detect, (uint32_t)levels);
  putU32(detect, (uint32_t)c.detect_classes);
  putU32(detect, (uint32_t)c.detect_reg_max);
  for (int l = 0; l < levels; ++l) {
    putI32(detect, c.detect_heads[2 * l]);
    putI32(detect, c.detect_heads[2 * l + 1]);
    putI32(detect, c.detect_strides[l]);
  }
  sections.push_back(std::make_pair((uint32_t)kDetect, detect));

  // The weights
  sections.push_back(std::make_pair((uint32_t)kInt8Weights,
                                    std::vector<uint8_t>(c.int8, c.int8 + c.int8_count)));
  std::vector<uint8_t> scales;
  for (size_t i = 0; i < c.scale_count; ++i) putF32(scales, c.scales[i]);
  sections.push_back(std::make_pair((uint32_t)kInt8Scales, scales));
  if (c.floats != nullptr) {
    std::vector<uint8_t> floats;
    for (size_t i = 0; i < c.float_count; ++i) putF32(floats, c.floats[i]);
    sections.push_back(std::make_pair((uint32_t)kFloatWeights, floats));
  }

  // NAMES
  std::vector<uint8_t> names;
  putU32(names, (uint32_t)c.names.size());
  for (const std::string& n : c.names) {
    const size_t len = n.size() > 255 ? 255 : n.size();
    names.push_back((uint8_t)len);
    names.insert(names.end(), n.begin(), n.begin() + len);
  }
  sections.push_back(std::make_pair((uint32_t)kNames, names));

  // LAYER_OUTPUTS (optional)
  if (!c.layer_outputs.empty()) {
    std::vector<uint8_t> outputs;
    putU32(outputs, (uint32_t)c.layer_outputs.size());
    for (const auto& pair : c.layer_outputs) {
      putI32(outputs, pair.first);
      putI32(outputs, pair.second);
    }
    sections.push_back(std::make_pair((uint32_t)kLayerOutputs, outputs));
  }

  // Header and section table, then the sections, each on a 16-byte boundary.
  const size_t header_size = 20 + 12 * sections.size();
  uint32_t flags = 0;
  if (c.floats != nullptr) flags |= kFlagHasFloat;
  if (c.per_tensor_quant) flags |= kFlagPerTensorQuant;

  std::vector<uint8_t> file;
  file.insert(file.end(), {'T', 'Y', 'M', '1'});
  file.push_back(kMajorVersion);
  file.push_back(0);
  file.push_back(kMinorVersion);
  file.push_back(0);
  putU32(file, (uint32_t)header_size);
  putU32(file, flags);
  putU32(file, (uint32_t)sections.size());

  std::vector<uint8_t> body;  // the section data, placed after the header
  size_t position = header_size;
  std::vector<std::pair<uint32_t, size_t>> placed;
  for (const auto& section : sections) {
    position = (position + kSectionAlign - 1) / kSectionAlign * kSectionAlign;
    placed.push_back(std::make_pair(position, section.second.size()));
    body.resize(position - header_size, 0);
    body.insert(body.end(), section.second.begin(), section.second.end());
    position += section.second.size();
  }
  for (size_t i = 0; i < sections.size(); ++i) {
    putU32(file, sections[i].first);
    putU32(file, (uint32_t)placed[i].first);
    putU32(file, (uint32_t)placed[i].second);
  }
  file.insert(file.end(), body.begin(), body.end());
  const uint32_t crc = crc32(file.data(), file.size());
  putU32(file, crc);
  return file;
}

}  // namespace tym
}  // namespace tinyyolo
