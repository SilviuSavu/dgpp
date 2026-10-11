// The full GLM-5.3 TP forwards (2026-09-12, docs/glm53_plan.md D9):
// loopback worlds of 2 and 4 ranks in one process (real verbs QPs over
// 127.0.0.1, the bus the fabric uses) over the tiny fixture, against the
// world-1 model as the oracle. Asserted:
//   * cross-rank: every layer's residual, the final read, the routing
//     decisions, every indexed layer's selection and the lm-head slices'
//     top-1 are bitwise identical on every rank (the canonical rank-order
//     fold; the indexer is replicated, so the selections must agree
//     exactly; a divergence is the silent-corruption class);
//   * vs world 1: per-layer residuals within the numerics budget (the
//     partial sums fold in bf16 on the wire), the final read likewise, the
//     top-1 per token equal or a certified near tie; a row whose routing
//     or selection flipped against world 1 (a near tie under the folds'
//     reassociation) is exempt from the hard-element count;
//   * the head geometry: 16 MLA heads slice to 8 and 4 per rank (the
//     split attention kernel's head-group floor), the experts' 256-wide
//     intermediate slices to 128 and 64 (the packed core's group of 64).
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "common/test.hpp"
#include "engine/tp_bus.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/model.hpp"
#include "net/collective_bus.hpp"
#include "glm_dsa_fixture.hpp"

namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;
using dgpp::BusBoundaryReducer;
using dgpp::GlmDsaLayerStream;
using dgpp::GlmDsaModel;
using dgpp::GlmDsaResidency;
using dgpp::GlmDsaTextConfig;
using dgpp::net::BusOptions;
using dgpp::net::CollectiveBus;

namespace {

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

int wait_timeout_ms() {
  const char* v = std::getenv("DGPP_TEST_WAIT_TIMEOUT_MS");
  return v ? std::atoi(v) : 60000;
}

BusOptions loop_options(int rank, int world, uint16_t port) {
  BusOptions o;
  o.world_size = world;
  o.my_rank = rank;
  o.rendezvous_port = port;
  o.rendezvous_host = rank == 0 ? "" : "127.0.0.1";
  o.rendezvous_timeout_ms = 20000;
  o.lat_slots = 8;
  // A 256 KB latency slot: every fold of this test (72 rows x 256 or
  // 1024 bf16) rides the chunked latency path. The bulk RS+AG machine
  // stalls at world 4 when a buffer spans fewer stripes than ranks (the
  // 36 KB attention fold: one stripe, three ranks with no shard, the AG
  // phase waiting on an arrival that never comes — 2026-09-09, recorded
  // in the plan's risks; the fabric's prefill folds are megabytes and its
  // decode folds ride the recorded collectives, so production is not on
  // that path).
  o.lat_slot_bytes = 262144;
  o.bulk_slots = 8;
  o.bulk_slot_bytes = 262144;
  o.qp_depth = 1024;
  o.completion_timeout_ms = [] {
    const char* ms = std::getenv("DGPP_TEST_BUS_TIMEOUT_MS");
    return ms ? std::atoi(ms) : 5000;
  }();
  o.consumer_deadline_s = [] {
    const char* s = std::getenv("DGPP_TEST_CONSUMER_DEADLINE_S");
    return s ? std::atof(s) : 20.0;
  }();
  o.launch_consumers = false;
  return o;
}

std::vector<std::unique_ptr<CollectiveBus>> start_world(int world, uint16_t port) {
  std::vector<std::unique_ptr<CollectiveBus>> out;
  for (int r = 0; r < world; ++r) out.push_back(std::make_unique<CollectiveBus>(loop_options(r, world, port)));
  std::vector<std::string> errors(static_cast<size_t>(world));
  std::thread listener([&] {
    if (!out[0]->start(&errors[0])) DGPP_LOG_ERROR("glm_dsa tp world rank 0: {}", errors[0]);
  });
  std::vector<std::thread> connectors;
  for (int r = 1; r < world; ++r)
    connectors.emplace_back([&, r] {
      if (!out[static_cast<size_t>(r)]->start(&errors[static_cast<size_t>(r)]))
        DGPP_LOG_ERROR("glm_dsa tp world rank {}: {}", r, errors[static_cast<size_t>(r)]);
    });
  listener.join();
  for (auto& t : connectors) t.join();
  for (int r = 0; r < world; ++r)
    if (!errors[static_cast<size_t>(r)].empty()) return {};
  return out;
}

struct ConstructBarrier {
  std::mutex mu;
  std::condition_variable cv;
  int left;
  explicit ConstructBarrier(int world) : left(world) {}
  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mu);
    if (--left == 0) cv.notify_all();
    else cv.wait(lock, [this] { return left == 0; });
  }
};

struct RankOutcome {
  std::string error;
  GlmDsaModel::Outputs out;
};

void rank_work(int rank, int world, const std::string& dir, const GlmDsaTextConfig& cfg,
               const std::vector<int64_t>& tokens, CollectiveBus* bus, ConstructBarrier* barrier,
               RankOutcome* out) {
  bool arrived = false;
  auto arrive_once = [&] {
    if (arrived) return;
    arrived = true;
    barrier->arrive_and_wait();
  };
  try {
    BusBoundaryReducer reducer(*bus, wait_timeout_ms());
    GlmDsaModel model(cfg, dir, static_cast<int>(tokens.size()), 256, GlmDsaResidency::Streaming, &reducer,
                    rank, world);
    arrive_once();
    out->out = model.forward(tokens, true);
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("glm_dsa tp rank {} failed: {}", rank, e.what());
    out->error = "rank " + std::to_string(rank) + ": " + e.what();
    arrive_once();
  }
}

std::vector<int64_t> make_tokens(const GlmDsaTextConfig& cfg, int n) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = 0xA5A5F00DBEEF1234ull;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(cfg.vocab_size));
  }
  if (n > 9) t[9] = cfg.eos_token_ids.empty() ? 0 : cfg.eos_token_ids[0];
  return t;
}

bool bits_equal(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 2) == 0);
}

struct Stats {
  double l2 = 0, max_ulps = 0;
  long hard = 0, total = 0;
};

int bf16_ulps(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) -> int32_t {
    return (v & 0x8000u) ? -static_cast<int32_t>(v & 0x7FFFu) : static_cast<int32_t>(v & 0x7FFFu);
  };
  return std::abs(static_cast<int>(key(a) - key(b)));
}

Stats compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
  Stats s;
  s.total = static_cast<long>(got.size());
  double rms = 0;
  for (uint16_t v : want) rms += std::pow(bf16_bits_to_float(v), 2);
  rms = std::sqrt(rms / std::max<size_t>(want.size(), 1));
  double d2 = 0, o2 = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = bf16_bits_to_float(got[i]), w = bf16_bits_to_float(want[i]);
    int u = bf16_ulps(got[i], want[i]);
    if (std::fabs(g - w) <= 0.02 * rms) u = 0;
    s.max_ulps = std::max(s.max_ulps, static_cast<double>(u));
    if (u > 128) ++s.hard;
    d2 += (g - w) * (g - w);
    o2 += w * w;
  }
  s.l2 = std::sqrt(d2) / std::sqrt(o2 + 1e-30);
  return s;
}

// The merged top-1 over the ranks' vocab slices: the highest logit, ties
// to the lower id.
std::vector<std::pair<int32_t, float>> merged_top1(const std::vector<RankOutcome>& ranks, int T) {
  std::vector<std::pair<int32_t, float>> best(static_cast<size_t>(T), {-1, -INFINITY});
  for (const RankOutcome& r : ranks)
    for (int t = 0; t < T; ++t)
      for (int c = 0; c < r.out.lm_vocab_count; ++c) {
        const float v = r.out.logits[static_cast<size_t>(t) * r.out.lm_vocab_count + c];
        const int32_t id = r.out.lm_vocab_begin + c;
        auto& b = best[static_cast<size_t>(t)];
        if (v > b.second || (v == b.second && id < b.first)) b = {id, v};
      }
  return best;
}

GlmDsaModel::Outputs check_world(int world, uint16_t port, const std::string& dir, const GlmDsaTextConfig& cfg,
                                 const std::vector<int64_t>& tokens, const GlmDsaModel::Outputs& ref) {
  DGPP_LOG_INFO("glm_dsa TP loopback world={} tokens={}", world, tokens.size());
  std::vector<std::unique_ptr<CollectiveBus>> buses = start_world(world, port);
  require(!buses.empty(), "tp bus world failed to start");
  std::vector<RankOutcome> ranks(static_cast<size_t>(world));
  ConstructBarrier barrier(world);
  std::vector<std::thread> workers;
  for (int r = 0; r < world; ++r)
    workers.emplace_back(rank_work, r, world, dir, std::cref(cfg), std::cref(tokens),
                         buses[static_cast<size_t>(r)].get(), &barrier, &ranks[static_cast<size_t>(r)]);
  for (auto& t : workers) t.join();
  for (int r = 0; r < world; ++r) require(ranks[static_cast<size_t>(r)].error.empty(), ranks[static_cast<size_t>(r)].error);
  const int T = static_cast<int>(tokens.size());
  const int L = cfg.num_hidden_layers;
  // ---- cross-rank bitwise ------------------------------------------------------
  for (int r = 1; r < world; ++r) {
    const GlmDsaModel::Outputs& a = ranks[0].out;
    const GlmDsaModel::Outputs& b = ranks[static_cast<size_t>(r)].out;
    require(bits_equal(a.final_hidden_bits, b.final_hidden_bits), "final hidden differs across ranks");
    require(a.layer_states.size() == static_cast<size_t>(L) && b.layer_states.size() == static_cast<size_t>(L),
            "layer captures missing");
    for (int l = 0; l < L; ++l)
      require(bits_equal(a.layer_states[static_cast<size_t>(l)], b.layer_states[static_cast<size_t>(l)]),
              "layer " + std::to_string(l) + " residual differs across ranks");
    require(a.route_ids == b.route_ids, "routing differs across ranks");
    require(a.dsa_selections == b.dsa_selections, "the DSA selections differ across ranks");
  }
  // ---- vs world 1 ---------------------------------------------------------------
  // The folds reassociate (bf16 on the wire), so every element differs at
  // rounding level; a routing flip on a near tie (the replicated router
  // reads a slightly different input) sends that ROW to different experts
  // — a large, legitimate divergence confined to the flipped rows. Rows
  // routed differently at this or an earlier MoE layer are exempt from the
  // hard-element count; every other row must have none.
  const GlmDsaModel::Outputs& o = ranks[0].out;
  const int H = cfg.hidden_size;
  const int K = cfg.num_experts_per_tok;
  require(o.route_ids.size() == static_cast<size_t>(cfg.num_moe_layers()) && ref.route_ids.size() == o.route_ids.size(),
          "route captures");
  // A flipped token's cached rows perturb every later query at the deeper
  // layers (the cache is shared), so the comparison past the first flip is
  // the fixture's chaos, not the fold's: the CLEAN PREFIX — the rows before
  // the first flipped one — is what the deeper layers are held to (the
  // dense-regime rows, at least topk of them, are always clean).
  std::vector<bool> flipped(static_cast<size_t>(T), false);
  long flips = 0, slots = 0, sel_flips = 0;
  require(o.dsa_selections.size() == static_cast<size_t>(cfg.num_indexer_layers()) &&
              ref.dsa_selections.size() == o.dsa_selections.size(),
          "selection captures");
  int index_ordinal = 0;
  for (int l = 0; l < L; ++l) {
    if (cfg.owns_indexer(l)) {
      // A selection that differs from world 1's: the folds' reassociation
      // moved a boundary near tie; the row (and, through the shared
      // layers, its dependants' rows) is exempt from the hard count.
      const std::vector<int32_t>& a = o.dsa_selections[static_cast<size_t>(index_ordinal)];
      const std::vector<int32_t>& b = ref.dsa_selections[static_cast<size_t>(index_ordinal)];
      const size_t ms = a.size() / static_cast<size_t>(T);
      for (int t = 0; t < T; ++t)
        if (std::memcmp(a.data() + static_cast<size_t>(t) * ms, b.data() + static_cast<size_t>(t) * ms, ms * 4) != 0) {
          ++sel_flips;
          if (!flipped[static_cast<size_t>(t)]) std::printf("[ .. ] world %d layer %d: selection flip on row %d\n", world, l, t);
          flipped[static_cast<size_t>(t)] = true;
        }
      ++index_ordinal;
    }
    if (cfg.is_moe_layer(l)) {
      const size_t m = static_cast<size_t>(l - cfg.first_k_dense_replace);
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < K; ++i, ++slots)
          if (o.route_ids[m][static_cast<size_t>(t) * K + i] != ref.route_ids[m][static_cast<size_t>(t) * K + i]) {
            ++flips;
            if (!flipped[static_cast<size_t>(t)]) std::printf("[ .. ] world %d layer %d: routing flip on row %d\n", world, l, t);
            flipped[static_cast<size_t>(t)] = true;
          }
    }
    // The clean prefix's statistics (the world-1 selections at the first
    // indexed layer are bitwise the TP ones — its indexer reads the
    // replicated input — so the prefix is the whole chunk there).
    int first_flip = T;
    for (int t = 0; t < T; ++t)
      if (flipped[static_cast<size_t>(t)]) {
        first_flip = t;
        break;
      }
    std::vector<uint16_t> kept_got, kept_want;
    long hard_clean = 0;
    int kept_rows = 0;
    std::vector<std::pair<double, int>> rows;
    for (int t = 0; t < T; ++t) {
      const std::vector<uint16_t> got(o.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t) * H,
                                      o.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t + 1) * H);
      const std::vector<uint16_t> want(ref.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t) * H,
                                       ref.layer_states[static_cast<size_t>(l)].begin() + static_cast<size_t>(t + 1) * H);
      const Stats r = compare(got, want);
      rows.emplace_back(r.l2, t);
      if (t < first_flip) {
        hard_clean += r.hard;
        ++kept_rows;
        kept_got.insert(kept_got.end(), got.begin(), got.end());
        kept_want.insert(kept_want.end(), want.begin(), want.end());
      }
    }
    const Stats s = compare(kept_got, kept_want);
    std::sort(rows.rbegin(), rows.rend());
    std::printf("[ .. ] world %d layer %d (clean prefix %d of %d rows): l2 %.3g max %g ulps hard %ld of %ld; worst rows",
                world, l, kept_rows, T, s.l2, s.max_ulps, hard_clean, s.total);
    for (int i = 0; i < 4 && i < T; ++i)
      std::printf(" t%d %.3g%s", rows[static_cast<size_t>(i)].second, rows[static_cast<size_t>(i)].first,
                  flipped[static_cast<size_t>(rows[static_cast<size_t>(i)].second)] ? "*" : "");
    std::printf("\n");
    // The clean prefix must span the dense regime — except under the int8
    // activation contract (the Mixed346 checkpoint), whose code flips
    // roughly double a layer's response to the folds' noise (glm_moe_test's
    // sensitivity probe) and cascade the fixture's routing near ties up the
    // context: there the kept rows must exist and hold the numerics budget
    // below, which a wrong slice or fold would still break at layer 1 on
    // every row.
    require(kept_rows >= (cfg.expert_activation_bits == 8 ? 1 : cfg.index_topk),
            "world " + std::to_string(world) + " layer " + std::to_string(l) +
                ": the clean prefix is shorter than the dense regime");
    // The bf16 wire noise of a wider world reaches the odd element of a
    // sharp attention row: one hard element per thousand is the budget.
    require(s.l2 < 0.02 && hard_clean <= s.total / 1000, "world " + std::to_string(world) + " layer " +
                                                             std::to_string(l) + " outside the numerics budget");
  }
  int first_flip = T;
  for (int t = 0; t < T; ++t)
    if (flipped[static_cast<size_t>(t)]) {
      first_flip = t;
      break;
    }
  {
    std::vector<uint16_t> kept_got, kept_want;
    for (int t = 0; t < first_flip; ++t) {
      kept_got.insert(kept_got.end(), o.final_hidden_bits.begin() + static_cast<size_t>(t) * H,
                      o.final_hidden_bits.begin() + static_cast<size_t>(t + 1) * H);
      kept_want.insert(kept_want.end(), ref.final_hidden_bits.begin() + static_cast<size_t>(t) * H,
                       ref.final_hidden_bits.begin() + static_cast<size_t>(t + 1) * H);
    }
    const Stats f = compare(kept_got, kept_want);
    std::printf("[ .. ] world %d final hidden (clean prefix %d rows): l2 %.3g max %g ulps hard %ld of %ld\n", world,
                first_flip, f.l2, f.max_ulps, f.hard, f.total);
    require(f.l2 < 0.02 && f.hard <= f.total / 1000, "world " + std::to_string(world) + " final hidden outside the budget");
  }
  std::printf("[ .. ] world %d routing: %ld of %ld slots differ from world 1; %ld selection flips\n", world, flips,
              slots, sel_flips);
  require(flips <= slots / 20, "too many routing flips against world 1");
  require(sel_flips <= T * cfg.num_indexer_layers() / 2, "too many selection flips against world 1");
  // The merged top-1 vs world 1's: equal, or a near tie in world 1's logits
  // (the flipped rows exempt).
  const auto merged = merged_top1(ranks, T);
  const std::vector<std::pair<int32_t, float>> single = merged_top1({RankOutcome{"", ref}}, T);
  int mism = 0;
  for (int t = 0; t < first_flip; ++t) {
    if (merged[static_cast<size_t>(t)].first == single[static_cast<size_t>(t)].first) continue;
    // world 1's logit for the TP top-1 must be within 2 % of its own top-1.
    const float v_tp = ref.logits[static_cast<size_t>(t) * ref.lm_vocab_count + merged[static_cast<size_t>(t)].first];
    const float v_1 = single[static_cast<size_t>(t)].second;
    if (std::fabs(v_1 - v_tp) > 0.02 * std::fabs(v_1)) ++mism;
  }
  std::printf("[ .. ] world %d top-1 (clean prefix): %d of %d rows beyond a near tie\n", world, mism, first_flip);
  require(mism == 0, "world " + std::to_string(world) + " top-1 differs beyond a near tie");
  return ranks[0].out;
}

}  // namespace

// The three fixtures: the int4/int8 release's shape, the NF4I8 contract's
// (codebook routed experts in the H32-rotated basis, 128-wide down
// slices at world 4) and the Mixed346 contract's (every expert form, the
// int8 activation codes), each on its own ports.
DGPP_TEST(glm_dsa_tp_loopback_worlds_2_and_4_match_world_1) {
  for (const auto& [nf4i8, m346] : {std::pair<bool, bool>{false, false}, {true, false}, {false, true}}) {
  const std::string dir = (fs::current_path() / (m346 ? "glm_dsa_tp_fixture_m346" : nf4i8 ? "glm_dsa_tp_fixture_nf4i8" : "glm_dsa_tp_fixture")).string();
  const GlmDsaTextConfig cfg = glmdsafx::tiny_config(5, 1, true, nf4i8, 8, m346);
  glmdsafx::write_fixture(cfg, dir, nf4i8, 8, m346);
  // 29972-29975: the nf4i8 worlds; 29994-29997: the mixed346 worlds.
  const uint16_t port = m346 ? 29994 : nf4i8 ? 29972 : 29952;
  const std::vector<int64_t> tokens = make_tokens(cfg, 72);
  GlmDsaModel::Outputs ref;
  {
    GlmDsaModel single(cfg, dir, static_cast<int>(tokens.size()), 256);
    ref = single.forward(tokens, true);
  }
  const GlmDsaModel::Outputs rep2 = check_world(2, port, dir, cfg, tokens, ref);
  const GlmDsaModel::Outputs rep4 = check_world(4, port + 1, dir, cfg, tokens, ref);
  // engine.embed_sharding = vocab: each rank gathers its rows of the
  // embedding and one fold sums them — the same worlds must be bitwise the
  // replicated ones (a row plus zeros is exact), selections included.
  GlmDsaLayerStream::set_embed_vocab_sharded(true);
  const GlmDsaModel::Outputs sh2 = check_world(2, (nf4i8 || m346) ? port + 2 : 29958, dir, cfg, tokens, ref);
  const GlmDsaModel::Outputs sh4 = check_world(4, (nf4i8 || m346) ? port + 3 : 29959, dir, cfg, tokens, ref);
  GlmDsaLayerStream::set_embed_vocab_sharded(false);
  for (const auto& [rep, sh, w] : {std::tuple<const GlmDsaModel::Outputs&, const GlmDsaModel::Outputs&, int>{rep2, sh2, 2},
                                   std::tuple<const GlmDsaModel::Outputs&, const GlmDsaModel::Outputs&, int>{rep4, sh4, 4}}) {
    const std::string tag = "world " + std::to_string(w) + " with the vocab-sharded embedding: ";
    require(bits_equal(rep.final_hidden_bits, sh.final_hidden_bits), tag + "final hidden differs from the replicated");
    require(rep.layer_states.size() == sh.layer_states.size(), tag + "layer captures");
    for (size_t l = 0; l < rep.layer_states.size(); ++l)
      require(bits_equal(rep.layer_states[l], sh.layer_states[l]), tag + "layer " + std::to_string(l) + " differs");
    require(rep.route_ids == sh.route_ids && rep.dsa_selections == sh.dsa_selections, tag + "routing or selections differ");
  }
  }
}

int main() { return dgpp::test::run_all(); }
