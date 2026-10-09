// glm_dsa_forward_check: the full GLM-5.3 diagnostic forward over a real
// checkpoint — one prompt of token ids through the streaming resident
// loader, the per-layer residual magnitudes and every position's top-k
// next-token logits; ids come in on the command line. --dump-states
// writes the residuals for tools/glm_dsa_torch_reference.py (transformers'
// own GlmMoeDsa layers on the same weights).
//
//   glm_dsa_forward_check --model ORG/NAME | --checkpoint-dir DIR
//                      --ids 1,2,3,... [--topk K] [--layers N] [--dump-states FILE]
//                      [--world W --rank R --peer HOST --port N] [--resident]
//                      [--image-dir DIR|off] [--lat-slot-bytes N]
//                      [--teacher-file FILE] (scores every prefill position;
//                        scripts/fabric_logprob.py --prefill joins TP slices)
//
// TP (plan D7): one process per node over the fabric bus; every fold of
// the diagnostic forward rides the latency path, so the slot is sized for
// the widest one (a block output [T, hidden]) — the bulk machine stalls at
// world 4 on buffers spanning fewer stripes than ranks (2026-09-09, the
// Qwen plan's risks). Each rank prints its per-layer residual digests
// (bitwise identical across ranks by contract) and the merged top-1 needs
// every rank's vocab slice: each rank prints its own slice's argmax with
// the logit, the script merges.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/log.hpp"
#include "engine/eager_engine.hpp"
#include "engine/graph_engine.hpp"
#include "text/tokenizer.hpp"
#include "engine/tp_bus.hpp"
#include "loaders/architecture.hpp"
#include "net/collective_bus.hpp"
#include "loaders/hf_cache.hpp"
#include "kernels/latent_format.hpp"
#include "models/glm_dsa/config.hpp"
#include "models/glm_dsa/model.hpp"

// --kv-format: the DSA latent cache storage format (bf16 default; the
// teacher-forced KV-dtype A/B passes fp8/fp4 here).
static dgpp::LatentFormat g_kv_format = dgpp::LatentFormat::kBf16;

int main(int argc, char** argv) {
  std::string model_id, ckpt, ids_text, dump_states, peer, image_dir, text, teacher_file;
  int decode_steps = 0;
  int topk = 5, layers = -1, world = 1, rank = 0, port = 29950;
  bool resident = false;
  size_t lat_slot_bytes = 0;
  auto next = [&](int& i) -> std::string {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--kv-format") {
        const auto f = dgpp::latent_format_from_string(next(i));
        if (!f) throw std::runtime_error("--kv-format must be bf16, fp8, fp4, fp8_block or fp4_block");
        g_kv_format = *f;
      }
      else if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--ids") ids_text = next(i);
      else if (a == "--text") text = next(i);              // tokenized with the checkpoint's tokenizer
      else if (a == "--teacher-file")
        teacher_file = next(i);
      else if (a == "--decode-steps") decode_steps = std::stoi(next(i));  // greedy steps after the forward, audited
      else if (a == "--topk") topk = std::stoi(next(i));
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--dump-states") dump_states = next(i);
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--peer") peer = next(i);
      else if (a == "--port") port = std::stoi(next(i));
      else if (a == "--resident") resident = true;
      else if (a == "--image-dir") image_dir = next(i);
      else if (a == "--lat-slot-bytes") lat_slot_bytes = std::stoull(next(i));
      else throw std::runtime_error("unknown argument " + a);
    }
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    if (!teacher_file.empty()) {
      if (!ids_text.empty() || !text.empty() || decode_steps != 0)
        throw std::runtime_error(
            "--teacher-file cannot be combined with --ids, --text or --decode-steps");
      std::ifstream input(teacher_file, std::ios::binary);
      if (!input) throw std::runtime_error("cannot read teacher file: " + teacher_file);
      text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }
    if (ids_text.empty() && text.empty())
      throw std::runtime_error("--ids, --text or a nonempty --teacher-file is required");
    std::vector<int64_t> ids;
    std::unique_ptr<dgpp::text::Tokenizer> tok;
    if (!text.empty()) {
      tok = std::make_unique<dgpp::text::Tokenizer>(
          dgpp::text::Tokenizer::load((std::filesystem::path(ckpt) / "tokenizer.json").string()));
      ids = tok->encode(text);
    } else {
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::stoll(item));
    }
    if (decode_steps > 0 && !tok)
      tok = std::make_unique<dgpp::text::Tokenizer>(
          dgpp::text::Tokenizer::load((std::filesystem::path(ckpt) / "tokenizer.json").string()));
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::GlmMoeDsa)
      throw std::runtime_error("not a GlmMoeDsa checkpoint: " + ckpt);
    dgpp::GlmDsaTextConfig cfg = dgpp::GlmDsaTextConfig::from_json_file(cfg_path);
    if (layers > 0 && layers < cfg.num_hidden_layers) {
      cfg.num_hidden_layers = layers;
      cfg.num_nextn_predict_layers = 0;  // the draft layer follows the full stack only
      cfg.indexer_full.resize(static_cast<size_t>(layers));
      if (cfg.packed_layer_end > layers) cfg.packed_layer_end = layers;
      if (cfg.packed_layer_begin > cfg.packed_layer_end) cfg.packed_layer_begin = cfg.packed_layer_end;
    }
    const int T = static_cast<int>(ids.size());
    if (!teacher_file.empty() && T < 2)
      throw std::runtime_error("teacher file needs at least two tokens");
    const int H = cfg.hidden_size, W = H;
    if (!image_dir.empty()) dgpp::GlmDsaLayerStream::set_resident_image_dir(image_dir == "off" ? "" : image_dir);
    const dgpp::GlmDsaResidency residency = resident ? dgpp::GlmDsaResidency::Resident : dgpp::GlmDsaResidency::Streaming;
    DGPP_LOG_INFO("glm_dsa_forward_check: {} — {} tokens, {} layers, {} world {} rank {}", ckpt, T,
                  cfg.num_hidden_layers, resident ? "resident" : "streaming", world, rank);
    std::unique_ptr<dgpp::net::CollectiveBus> bus;
    std::unique_ptr<dgpp::BusBoundaryReducer> reducer;
    if (world > 1) {
      if (peer.empty() && rank != 0) throw std::runtime_error("--peer is required on ranks > 0");
      // The widest fold: a block output [T, hidden] (bf16).
      if (lat_slot_bytes == 0) lat_slot_bytes = static_cast<size_t>(T) * W * 2 + 4096;
      bus = std::make_unique<dgpp::net::CollectiveBus>(
          dgpp::fabric_bus_options(rank, world, static_cast<uint16_t>(port), peer, 120000, lat_slot_bytes));
      std::string err;
      if (!bus->start(&err)) throw std::runtime_error("rank " + std::to_string(rank) + " bus start: " + err);
      reducer = std::make_unique<dgpp::BusBoundaryReducer>(*bus, 600000);
    }
    const auto t0 = std::chrono::steady_clock::now();
    dgpp::GlmDsaModel model(cfg, ckpt, T + decode_steps, T + decode_steps + 64, residency, reducer.get(), rank, world,
                          /*max_requests=*/1, /*mtp=*/false, /*decode_rows=*/0, g_kv_format);
    const auto t1 = std::chrono::steady_clock::now();
    const dgpp::GlmDsaModel::Outputs out =
        model.forward(ids, teacher_file.empty() || !dump_states.empty());
    const auto t2 = std::chrono::steady_clock::now();
    auto fnv = [](const std::vector<uint16_t>& v) {
      uint64_t h = 1469598103934665603ull;
      for (uint16_t x : v) { h ^= x; h *= 1099511628211ull; }
      return h;
    };
    for (size_t l = 0; l < out.layer_states.size(); ++l) {
      double rms = 0, mx = 0;
      for (uint16_t v : out.layer_states[l]) {
        const float f = dgpp::bf16_bits_to_float(v);
        rms += static_cast<double>(f) * f;
        mx = std::max(mx, static_cast<double>(std::fabs(f)));
      }
      rms = std::sqrt(rms / static_cast<double>(out.layer_states[l].size()));
      std::printf("layer %2zu: h rms %.4g max %.4g digest %016llx\n", l, rms, mx,
                  static_cast<unsigned long long>(fnv(out.layer_states[l])));
    }
    std::printf("final hidden digest %016llx\n", static_cast<unsigned long long>(fnv(out.final_hidden_bits)));
    // This rank's vocab slice: the top-k with GLOBAL ids and the logits.
    const auto top = dgpp::GlmDsaModel::topk(out.logits, T, out.lm_vocab_count, topk);
    if (!teacher_file.empty()) {
      // These are LOCAL argmax values. The prefill logprob reader merges
      // them by logit across ranks before computing global top-1 accuracy.
      // Scoring the full forward exercises the prefill kernels at every
      // position; incremental teacher forcing would primarily test decode.
      for (int t = 0; t + 1 < T; ++t) {
        const float* row = out.logits.data() + static_cast<size_t>(t) * out.lm_vocab_count;
        const auto best = std::max_element(row, row + out.lm_vocab_count);
        const double mx = *best;
        double sum = 0;
        for (int v = 0; v < out.lm_vocab_count; ++v)
          sum += std::exp(static_cast<double>(row[v]) - mx);
        const int64_t target = ids[static_cast<size_t>(t) + 1];
        const int64_t local = target - out.lm_vocab_begin;
        const double logit = local >= 0 && local < out.lm_vocab_count ? row[local] : NAN;
        std::printf(
            "[prefill_tf] rank %d step %d: target %lld argmax %d lmax %.9g lse %.12g target_logit "
            "%.9g\n",
            rank, t, static_cast<long long>(target),
            static_cast<int>(best - row) + out.lm_vocab_begin, mx, mx + std::log(sum), logit);
      }
      std::printf("[prefill_tf_end] rank %d world %d positions %d\n", rank, world, T - 1);
    }
    for (int t = 0; t < T; ++t) {
      std::printf("pos %3d id %7lld ->", t, static_cast<long long>(ids[static_cast<size_t>(t)]));
      for (const auto& [id, v] : top[static_cast<size_t>(t)]) std::printf(" %d(%.3f)", id + out.lm_vocab_begin, v);
      std::printf("\n");
    }
    std::printf("argmax ids:");
    for (int t = 0; t < T; ++t) std::printf("%s%d", t ? "," : " ", top[static_cast<size_t>(t)][0].first + out.lm_vocab_begin);
    std::printf("\n");
    std::printf("argmax logits:");
    for (int t = 0; t < T; ++t) std::printf("%s%.4f", t ? "," : " ", top[static_cast<size_t>(t)][0].second);
    std::printf("\n");
    // Teacher-forced reading (--text): each position's predicted piece
    // beside the text's actual next token — a forward that is right at
    // every length predicts most next tokens of a coherent paragraph.
    if (tok && world == 1) {
      int hits = 0;
      std::string line;
      for (int t = 0; t + 1 < T; ++t) {
        const int32_t pred = top[static_cast<size_t>(t)][0].first + out.lm_vocab_begin;
        const bool hit = pred == ids[static_cast<size_t>(t) + 1];
        hits += hit ? 1 : 0;
        line += (hit ? " [" : " {") + tok->decode(pred) + (hit ? "]" : "}");
      }
      std::printf("teacher-forced: %d of %d next tokens predicted; pieces ([hit] {miss}):%s\n", hits, T - 1,
                  line.c_str());
    }
    // The greedy decode audit (--decode-steps): the eager engine's steps
    // from the prompt (the fabric pick merges the ranks' slices), the
    // transcript decoded, then the re-forward of prompt + transcript —
    // every position's argmax (this rank's slice at world > 1) beside the
    // generated token, so a decode-path divergence shows as the first
    // position whose re-forward argmax is not the token the decode chose.
    if (decode_steps > 0) {
      uint16_t* scratch = nullptr;
      DGPP_CUDA_OK(cudaHostAlloc(reinterpret_cast<void**>(&scratch),
                                 sizeof(uint16_t) * dgpp::kPickScratchElems(std::max(world, 1)), cudaHostAllocDefault));
      dgpp::DecodePick pick = world > 1
          ? dgpp::make_fabric_pick(bus.get(), rank, world, scratch, cfg.vocab_size, 600000)
          : dgpp::make_w1_pick(cfg.vocab_size);
      dgpp::EagerEngineAdapter<dgpp::GlmDsaModel> eng(&model, 1, std::move(pick));
      std::vector<int64_t> gen;
      gen.push_back(eng.prefill(0, ids));
      eng.reserve(0, static_cast<int64_t>(ids.size()) + decode_steps + 1);
      const auto td0 = std::chrono::steady_clock::now();
      for (int s = 0; s < decode_steps; ++s) {
        const std::vector<int32_t> t = eng.step(0);
        for (const int32_t x : t) gen.push_back(x);
      }
      const double step_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td0).count() / decode_steps;
      eng.close(0);
      std::printf("decode: %zu tokens in %d steps (%.1f ms/step eager):", gen.size(), decode_steps, step_ms);
      for (const int64_t g : gen) std::printf(" %lld", static_cast<long long>(g));
      std::printf("\ndecode text: %s\n", tok->decode(gen).c_str());
      std::vector<int64_t> all(ids);
      all.insert(all.end(), gen.begin(), gen.end() - 1);
      const dgpp::GlmDsaModel::Outputs re = model.forward(all, false);
      const int R = static_cast<int>(all.size());
      const auto rtop = dgpp::GlmDsaModel::topk(re.logits, R, re.lm_vocab_count, 1);
      int agree = 0, first_diff = -1;
      std::string line;
      for (size_t i = 0; i < gen.size(); ++i) {
        const int row = static_cast<int>(ids.size()) - 1 + static_cast<int>(i);
        const int32_t am = rtop[static_cast<size_t>(row)][0].first + re.lm_vocab_begin;
        const bool same = am == gen[i];
        if (world == 1) {
          agree += same ? 1 : 0;
          if (!same && first_diff < 0) first_diff = static_cast<int>(i);
          line += (same ? " [" : " {") + tok->decode(am) + (same ? "]" : "}");
        } else {
          char buf[64];
          std::snprintf(buf, sizeof(buf), " %d:%d(%.2f)", row, am, rtop[static_cast<size_t>(row)][0].second);
          line += buf;
        }
      }
      if (world == 1)
        std::printf("re-forward audit: %d of %zu decode rows agree with the re-forward's argmax; first divergence at "
                    "generated token %d (position %zu)\n  pieces:%s\n",
                    agree, gen.size(), first_diff, first_diff < 0 ? 0 : ids.size() + first_diff, line.c_str());
      else
        std::printf("re-forward slice argmax per row (row:id(logit), this rank's vocab slice [%d, +%d)):%s\n",
                    re.lm_vocab_begin, re.lm_vocab_count, line.c_str());
      cudaFreeHost(scratch);
    }
    if (!dump_states.empty()) {
      std::ofstream f(dump_states, std::ios::binary);
      for (const auto& st : out.layer_states) f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
      f.write(reinterpret_cast<const char*>(out.final_hidden_bits.data()), static_cast<std::streamsize>(out.final_hidden_bits.size() * 2));
      std::printf("wrote %s: %zu layers x [%d, %d] + final [%d, %d] bf16\n", dump_states.c_str(), out.layer_states.size(), T, W, T, H);
    }
    const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("boot %.0f ms, forward %.0f ms\n", ms(t0, t1), ms(t1, t2));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "glm_dsa_forward_check: %s\n", e.what());
    return 1;
  }
}
