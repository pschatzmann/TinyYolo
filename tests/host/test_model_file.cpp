/// Tests for the single-file model format (docs/ModelFormat.md): the round trip, and the checks
/// the reader makes. Exit code 0 means every check passed.

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "TinyYolo/app/ModelFile.h"
#include "TinyYolo/app/ModelFileWriter.h"

using namespace tinyyolo;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_failures;                                                    \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    }                                                                  \
  } while (0)

namespace {

/// A small model: one Conv op from tensor 0 to tensor 1, one detect level on tensor 1.
tym::Content smallModel(bool with_float) {
  tym::Content c;
  c.name = "tiny-test";
  c.input_size = 64;
  c.tensor_count = 2;
  c.input_id = 0;
  GraphOp conv{};
  conv.kind = GraphOpKind::Conv;
  conv.n_in = 1;
  conv.in[0] = 0;
  conv.out = 1;
  conv.w = 0;
  conv.b = 1;
  conv.k = 3;
  conv.s = 1;
  conv.p = 1;
  conv.g = 1;
  conv.silu = true;
  conv.q = 0;
  c.ops.push_back(conv);
  GraphParam w{0, 4, {2, 1, 3, 3}};
  GraphParam b{18, 1, {2, 0, 0, 0}};
  c.params.push_back(w);
  c.params.push_back(b);
  GraphQuant q{0, 0, 0.5f, 0.25f};
  c.quant.push_back(q);
  c.per_tensor_quant = true;
  c.detect_heads = {1, 1};
  c.detect_strides = {8};
  c.detect_classes = 2;
  c.detect_reg_max = 1;
  c.names = {"cat", "dog"};
  c.layer_outputs = {{0, 1}, {5, 1}};
  static const int8_t int8[18] = {1, -2, 3, -4, 5, -6, 7, -8, 9, 10, -11, 12, 13, -14, 15, -16, 17, -18};
  static const float scales[2] = {0.125f, 0.5f};
  static float floats[20];
  for (int i = 0; i < 20; ++i) floats[i] = 0.25f * i - 2.0f;
  c.int8 = int8;
  c.int8_count = 18;
  c.scales = scales;
  c.scale_count = 2;
  c.floats = with_float ? floats : nullptr;
  c.float_count = with_float ? 20 : 0;
  return c;
}

bool sameOp(const GraphOp& a, const GraphOp& b) {
  return a.kind == b.kind && a.n_in == b.n_in && a.silu == b.silu && memcmp(a.in, b.in, sizeof(a.in)) == 0 &&
         a.out == b.out && a.w == b.w && a.b == b.b && a.k == b.k && a.s == b.s && a.p == b.p && a.g == b.g &&
         a.index == b.index && a.w2 == b.w2 && a.b2 == b.b2 && a.w3 == b.w3 && a.b3 == b.b3 &&
         a.heads == b.heads && a.key_dim == b.key_dim && a.head_dim == b.head_dim && a.q == b.q &&
         a.q2 == b.q2 && a.q3 == b.q3;
}

void testRoundTrip(bool with_float) {
  const tym::Content c = smallModel(with_float);
  const std::vector<uint8_t> file = tym::write(c);
  tym::ModelFile model;
  CHECK(model.begin(file.data(), file.size()) == Status::Ok);
  CHECK(model.ready());
  CHECK(model.name() == "tiny-test");
  CHECK(model.hasFloatWeights() == with_float);
  CHECK(model.perTensorQuant());
  const GraphModel& g = model.model();
  CHECK(g.op_count == 1);
  CHECK(g.param_count == 2);
  CHECK(g.tensor_count == 2);
  CHECK(g.input_id == 0);
  CHECK(g.input_size == 64);
  CHECK(g.detect_levels == 1);
  CHECK(g.detect_classes == 2);
  CHECK(g.detect_reg_max == 1);
  CHECK(g.detect_heads[0][0] == 1 && g.detect_heads[0][1] == 1);
  CHECK(g.detect_strides[0] == 8);
  CHECK(sameOp(g.ops[0], c.ops[0]));
  CHECK(g.params[0].offset == 0 && g.params[0].rank == 4 && g.params[0].dims[0] == 2 && g.params[0].dims[3] == 3);
  CHECK(g.params[1].offset == 18 && g.params[1].rank == 1 && g.params[1].dims[0] == 2);
  CHECK(g.quant[0].w_offset == 0 && g.quant[0].in_scale == 0.5f && g.quant[0].out_scale == 0.25f);
  CHECK(model.layerOutputs().size() == 2);
  CHECK(model.layerOutputs()[1].first == 5 && model.layerOutputs()[1].second == 1);
  CHECK(model.classCount() == 2);
  CHECK(model.className(0) == "cat" && model.className(1) == "dog" && model.className(2).empty());
  CHECK(model.int8Count() == 18);
  CHECK(model.int8Weights()[0] == 1 && model.int8Weights()[17] == -18);
  CHECK(model.scaleCount() == 2 && model.int8Scales()[1] == 0.5f);
  if (with_float) {
    CHECK(model.floatCount() == 20);
    CHECK(model.floatWeights()[3] == 0.25f * 3 - 2.0f);
  } else {
    CHECK(model.floatWeights() == nullptr && model.floatCount() == 0);
  }
  // The int8 weights are used in place, and start on a 16-byte boundary.
  CHECK(reinterpret_cast<uintptr_t>(model.int8Weights()) - reinterpret_cast<uintptr_t>(file.data()) < file.size());
  CHECK((reinterpret_cast<uintptr_t>(model.int8Weights()) - reinterpret_cast<uintptr_t>(file.data())) % 16 == 0);
}

void testRejects() {
  const std::vector<uint8_t> good = tym::write(smallModel(true));
  tym::ModelFile model;

  // A single flipped byte breaks the CRC.
  std::vector<uint8_t> bad = good;
  bad[good.size() / 2] ^= 0x10;
  CHECK(model.begin(bad.data(), bad.size()) == Status::InvalidModel);

  // Bad magic.
  bad = good;
  bad[0] = 'X';
  CHECK(model.begin(bad.data(), bad.size()) == Status::InvalidModel);

  // A different major version is not supported.
  bad = good;
  bad[4] = 2;
  CHECK(model.begin(bad.data(), bad.size()) == Status::Unsupported);

  // Truncated.
  CHECK(model.begin(good.data(), good.size() - 9) == Status::InvalidModel);
  CHECK(model.begin(good.data(), 10) == Status::InvalidArgument);
  CHECK(model.begin(nullptr, 0) == Status::InvalidArgument);

  // An op that refers to a tensor that does not exist is rejected, even with a valid CRC.
  tym::Content c = smallModel(false);
  c.ops[0].in[0] = 7;
  const std::vector<uint8_t> out_of_range = tym::write(c);
  CHECK(model.begin(out_of_range.data(), out_of_range.size()) == Status::InvalidModel);

  // A parameter index out of range is rejected too.
  c = smallModel(false);
  c.ops[0].w = 5;
  const std::vector<uint8_t> bad_param = tym::write(c);
  CHECK(model.begin(bad_param.data(), bad_param.size()) == Status::InvalidModel);

  // A name count that differs from the number of classes is rejected.
  c = smallModel(false);
  c.detect_classes = 3;
  const std::vector<uint8_t> bad_names = tym::write(c);
  CHECK(model.begin(bad_names.data(), bad_names.size()) == Status::InvalidModel);
}

void testCrc() {
  const char* text = "123456789";
  CHECK(tym::crc32((const uint8_t*)text, 9) == 0xCBF43926u);
}

}  // namespace

int main() {
  testCrc();
  testRoundTrip(true);
  testRoundTrip(false);
  testRejects();
  if (g_failures == 0) {
    printf("model file tests: %d checks passed\n", g_checks);
    return 0;
  }
  printf("model file tests: %d of %d checks failed\n", g_failures, g_checks);
  return 1;
}
