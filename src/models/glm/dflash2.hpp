#pragma once
// The DFlash2 drafter's GLM-5.3-Flash bindings (docs/
// performance_improvement_plan.md §7, the second GLM port after the full
// GLM/DSA model): the target-pair validation against GlmTextConfig. The
// drafter's own types, its checkpoint loader and the KV planes are shared
// (models/qwen/dflash2.hpp); the model integration lives in
// models/glm/dflash2_model.cpp and the Flash model's forward.hpp.
#include "models/glm/config.hpp"
#include "models/qwen/dflash2.hpp"

namespace dgpp {

// The silent-acceptance-failure hazards of a mismatched pair (§7 step 2:
// widths, vocab, tap indices) as free validation against the Flash target.
// Throws with a message naming the field.
void dflash2_validate_against(const DFlash2Config& c, const GlmTextConfig& target);

}  // namespace dgpp
