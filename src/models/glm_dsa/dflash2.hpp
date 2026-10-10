#pragma once
// The DFlash2 drafter's GLM-5.3 bindings (docs/performance_improvement_plan.md §7,
// the port of the Qwen3.5-family integration to the full GLM/DSA model).
//
// The drafter's own types — DFlash2Config, DFlash2Weights, the checkpoint
// loader and the KV planes — are model-agnostic and live in
// models/qwen/dflash2.hpp (their first home); this header adds the one
// thing the GLM side needs on top: the target-pair validation against
// GlmDsaTextConfig. The planes alias keeps the glm_dsa sources' name.
#include "models/glm_dsa/config.hpp"
#include "models/qwen/dflash2.hpp"

namespace dgpp {

// The silent-acceptance-failure hazards of a mismatched pair (§7 step 2:
// widths, vocab, tap indices) as free validation against the GLM target —
// DFlash2Config::validate_against is a member bound to the Qwen target.
// Throws with a message naming the field.
void dflash2_validate_against(const DFlash2Config& c, const GlmDsaTextConfig& target);

// The drafter's dedicated KV planes under their glm_dsa-era name (moved to
// the model-agnostic header for the GLM-5.3-Flash port; same class).
using GlmDFlash2Planes = DFlash2Planes;

}  // namespace dgpp
