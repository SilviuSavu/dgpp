// The DFlash2 drafter's GLM-5.3-Flash validation: the same hazards the
// glm_dsa port checks (models/glm_dsa/dflash2.cpp), against GlmTextConfig.
// The Flash family's config carries no positional ceiling of its own (the
// rope addresses the pool's capacity), so the drafter's context bound is
// the runtime check the draft path already holds (position + query rows
// within max_context).
#include <stdexcept>
#include <string>

#include "engine/decode_outputs.hpp"
#include "models/glm/config.hpp"
#include "models/glm/dflash2.hpp"

namespace dgpp {

namespace {
[[noreturn]] void reject(const std::string& field, const std::string& why) {
  throw std::invalid_argument("dflash2 (GLM-5.3-Flash): " + field + ": " + why);
}
}  // namespace

void dflash2_validate_against(const DFlash2Config& c, const GlmTextConfig& target) {
  if (c.hidden_size != target.hidden_size)
    reject("hidden_size", "the draft must share the target's width (fc and the shared head)");
  if (c.vocab_size != target.vocab_size)
    reject("vocab_size", "the draft shares the target's embeddings and lm head");
  if (c.mask_token_id < 0 || c.mask_token_id >= target.vocab_size)
    reject("dflash_config.mask_token_id", "outside the target vocabulary");
  if (c.drafts() > kSpecRows - 1)
    reject("dflash_config.block_size", "the draft depth exceeds the speculative row budget");
  for (size_t i = 0; i < c.target_layer_ids.size(); ++i) {
    if (c.target_layer_ids[i] < 0 || c.target_layer_ids[i] >= target.num_hidden_layers)
      reject("dflash_config.target_layer_ids",
             "layer " + std::to_string(c.target_layer_ids[i]) + " does not exist in the target");
    if (i && c.target_layer_ids[i] == c.target_layer_ids[i - 1])
      reject("dflash_config.target_layer_ids", "duplicate tap layer");
  }
  if (c.sliding_window <= c.block_size)
    reject("sliding_window", "the window must cover the whole query block");
}

}  // namespace dgpp
