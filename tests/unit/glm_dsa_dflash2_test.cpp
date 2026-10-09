// The DFlash2 drafter's GLM-5.3 pair checks (models/glm_dsa/dflash2.hpp):
// the target-matched checkpoint's cross-validation against
// GlmDsaTextConfig — the silent-acceptance-failure hazards (widths,
// vocab, tap indices, the block vs the window) as refusals, and the
// released GLM-5.3 drafter fixture passing a stock target.
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "engine/decode_outputs.hpp"
#include "loaders/minijson.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/dflash2.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// A trimmed fixture of the GLM-5.3 DFlash2 drafter's config.json (the
// qwen3-backbone drafter at the target's 6144 width, six layers, six
// taps, 154880 vocab).
const char* kFixture = R"({
  "architectures": ["DFlash2DraftModel"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_size": 6144,
  "num_hidden_layers": 6,
  "vocab_size": 154880,
  "num_attention_heads": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 12288,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {"rope_type": "default", "rope_theta": 1000000},
  "max_position_embeddings": 1048576,
  "use_sliding_window": true,
  "sliding_window": 2048,
  "is_causal": false,
  "tie_word_embeddings": false,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 154856,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [5, 19, 33, 47, 61, 75]
  }
})";

dgpp::DFlash2Config parse(const std::string& text) {
  return dgpp::DFlash2Config::parse(dgpp::minijson::parse(text).root);
}

std::string refusal_against(const dgpp::DFlash2Config& c, const dgpp::GlmDsaTextConfig& t) {
  try {
    dgpp::dflash2_validate_against(c, t);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(glm_dsa_dflash2_accepts_the_target_matched_pair) {
  const auto c = parse(kFixture);
  const dgpp::GlmDsaTextConfig target;  // the stock GLM-5.3 shape
  require(refusal_against(c, target).empty(), "the stock pair must pass");
  require(c.drafts() == 7 && c.query_rows() == 8, "the block");
  require(c.fc_in() == 6 * 6144, "the six taps' fc input");
  require(c.tp_divisible(4) && c.tp_divisible(2) && c.tp_divisible(1), "divides across the fabric worlds");
  require(!c.tp_divisible(3), "a non-dividing world must be refused by tp_divisible");
}

DGPP_TEST(glm_dsa_dflash2_refuses_the_mismatched_pair) {
  const auto c = parse(kFixture);
  dgpp::GlmDsaTextConfig t;
  require(!refusal_against(c, t).empty() == false, "sanity: the stock pair passes");
  t.hidden_size = 4096;
  require(!refusal_against(c, t).empty(), "a width mismatch must be refused (fc and the shared head)");
  t = dgpp::GlmDsaTextConfig();
  t.vocab_size = 151936;
  require(!refusal_against(c, t).empty(), "a vocab mismatch must be refused (the shared embeddings and head)");
  t = dgpp::GlmDsaTextConfig();
  t.num_hidden_layers = 61;
  require(!refusal_against(c, t).empty(), "a tap past the target stack must be refused");
  t = dgpp::GlmDsaTextConfig();
  t.max_position_embeddings = 131072;
  require(!refusal_against(c, t).empty(), "a draft longer than the target must be refused");
  auto narrow = c;
  narrow.sliding_window = 8;  // a window not covering the block
  require(!refusal_against(narrow, dgpp::GlmDsaTextConfig()).empty(),
          "a block past the window must be refused");
  auto deep = c;
  deep.block_size = dgpp::kSpecRows + 1;
  require(!refusal_against(deep, dgpp::GlmDsaTextConfig()).empty(),
          "an over-deep block must be refused (the speculative row budget)");
  auto out_of_vocab = c;
  out_of_vocab.mask_token_id = dgpp::GlmDsaTextConfig().vocab_size;
  require(!refusal_against(out_of_vocab, dgpp::GlmDsaTextConfig()).empty(),
          "a mask token outside the target vocabulary must be refused");
}
