// The DFlash2 drafter's GLM-5.3-Flash pair checks (models/glm/dflash2.hpp):
// the target-matched checkpoint's cross-validation against GlmTextConfig —
// the silent-acceptance-failure hazards (widths, vocab, tap indices) as
// refusals — and the released Flash drafter fixture
// (incoai/GLM-5.3-Flash-DFlash2) passing a stock target.
#include <stdexcept>
#include <string>

#include "common/test.hpp"
#include "engine/decode_outputs.hpp"
#include "loaders/minijson.hpp"
#include "models/glm/config.hpp"
#include "models/glm/dflash2.hpp"

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

// A trimmed fixture of incoai/GLM-5.3-Flash-DFlash2's config.json (the
// qwen3-backbone drafter at the target's 4096 width, five layers, five
// taps, 154880 vocab).
const char* kFixture = R"({
  "architectures": ["DFlash2DraftModel"],
  "model_type": "qwen3",
  "dtype": "bfloat16",
  "hidden_size": 4096,
  "num_hidden_layers": 5,
  "vocab_size": 154880,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 12288,
  "rms_norm_eps": 1e-05,
  "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
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
    "target_layer_ids": [5, 14, 24, 33, 42]
  }
})";

dgpp::DFlash2Config parse(const std::string& text) {
  return dgpp::DFlash2Config::parse(dgpp::minijson::parse(text).root);
}

std::string refusal_against(const dgpp::DFlash2Config& c, const dgpp::GlmTextConfig& t) {
  try {
    dgpp::dflash2_validate_against(c, t);
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

}  // namespace

DGPP_TEST(glm_dflash2_accepts_the_target_matched_pair) {
  const auto c = parse(kFixture);
  const dgpp::GlmTextConfig target;  // the stock GLM-5.3-Flash shape
  require(refusal_against(c, target).empty(), "the stock pair must pass");
  require(c.drafts() == 7 && c.query_rows() == 8, "the block");
  require(c.fc_in() == 5 * 4096, "the five taps' fc input");
  require(c.tp_divisible(4) && c.tp_divisible(2) && c.tp_divisible(1), "divides across the fabric worlds");
}

DGPP_TEST(glm_dflash2_refuses_the_mismatched_pair) {
  const auto c = parse(kFixture);
  dgpp::GlmTextConfig t;
  require(!refusal_against(c, t).empty() == false, "the stock pair passes first");
  t.hidden_size = 5120;
  require(refusal_against(c, t).find("hidden_size") != std::string::npos,
          "a width mismatch must be refused by name");
  t = dgpp::GlmTextConfig{};
  t.vocab_size = 151552;
  require(refusal_against(c, t).find("vocab_size") != std::string::npos,
          "a vocab mismatch must be refused by name");
  t = dgpp::GlmTextConfig{};
  t.num_hidden_layers = 42;
  require(refusal_against(c, t).find("target_layer_ids") != std::string::npos,
          "a tap past the stack must be refused by name");
  require(1 + c.drafts() <= dgpp::kSpecRows, "the block fits the speculative rows");
}

