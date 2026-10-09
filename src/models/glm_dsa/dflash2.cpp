// The DFlash2 drafter's GLM-5.3 validation (the port's only GLM-specific
// piece besides the planes): the silent-acceptance-failure hazards of a
// mismatched draft/target pair, as a free function against
// GlmDsaTextConfig (the member validate_against is bound to the Qwen
// target in the shared header).
#include <stdexcept>
#include <string>

#include "engine/decode_outputs.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/dflash2.hpp"

namespace dgpp {

namespace {
[[noreturn]] void reject(const std::string& field, const std::string& why) {
  throw std::invalid_argument("dflash2 (GLM): " + field + ": " + why);
}
}  // namespace

void dflash2_validate_against(const DFlash2Config& c, const GlmDsaTextConfig& target) {
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
  if (c.max_position_embeddings > target.max_position_embeddings)
    reject("max_position_embeddings", "the draft must not exceed the target's");
}

}  // namespace dgpp
