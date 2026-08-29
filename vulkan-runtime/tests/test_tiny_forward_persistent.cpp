#include "api/reduced_precision_storage.h"
#include "api/tiny_forward_persistent.h"
#include "core/vk_setup.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace vulkan_runtime::tiny;
    if (std::string(capability).find("rank4_rank8_sgd_adamw") == std::string::npos ||
        std::string(command_buffer_capability).find("production_fixed_shape_forward_loss") == std::string::npos ||
        !lora_rank_supported(LoraRank4) || !lora_rank_supported(LoraRank8) || lora_rank_supported(1) ||
        lora_rank_supported(6) || ProfileCount != 2 || !profile_supported(H, V, Vp, Tcap, LoraRank4) ||
        profile_supported(65, V, Vp, Tcap, LoraRank4)) {
        std::cerr << "Tiny profile validator mismatch\n";
        return 1;
    }
    std::vector<float> e(V * H), p(Tcap * H), q(H * H), k(H * H), v(H * H), o(H * H), lm(H * Vp);
    for (size_t i = 0; i < e.size(); ++i)
        e[i] = 0.001f * float(i % 17);
    for (size_t i = 0; i < p.size(); ++i)
        p[i] = 0.002f * float(i % 13);
    for (size_t i = 0; i < lm.size(); ++i)
        lm[i] = 0.003f * float(i % 11);
    for (uint32_t i = 0; i < H; ++i)
        q[i * H + i] = k[i * H + i] = v[i * H + i] = o[i * H + i] = 1.0f;
    auto ctx = vulkan_runtime::core::create_context("tiny-forward-test");
    {
        // End-to-end reduced-precision evaluation: only frozen inputs are
        // rounded to FP16 storage; the constructor widens them before the
        // unchanged FP32 shader graph runs.
        std::vector<std::uint16_t> eh(e.size()), ph(p.size()), qh(q.size()), kh(k.size()), vh(v.size()), oh(o.size()),
            lmh(lm.size());
        using vulkan_runtime::storage::fp32_to_fp16;
        for (size_t i = 0; i < e.size(); ++i)
            eh[i] = fp32_to_fp16(e[i]);
        for (size_t i = 0; i < p.size(); ++i)
            ph[i] = fp32_to_fp16(p[i]);
        for (size_t i = 0; i < q.size(); ++i)
            qh[i] = fp32_to_fp16(q[i]);
        for (size_t i = 0; i < k.size(); ++i)
            kh[i] = fp32_to_fp16(k[i]);
        for (size_t i = 0; i < v.size(); ++i)
            vh[i] = fp32_to_fp16(v[i]);
        for (size_t i = 0; i < o.size(); ++i)
            oh[i] = fp32_to_fp16(o[i]);
        for (size_t i = 0; i < lm.size(); ++i)
            lmh[i] = fp32_to_fp16(lm[i]);
        ForwardResourceGraph reduced(ctx, eh.data(), ph.data(), qh.data(), kh.data(), vh.data(), oh.data(), lmh.data());
        std::vector<std::uint32_t> reduced_tokens(Tcap);
        for (std::uint32_t i = 0; i < Tcap; ++i)
            reduced_tokens[i] = (i * 11u) % V;
        std::vector<float> reduced_logits(Tcap * Vp), baseline_logits(Tcap * Vp);
        reduced.forward(reduced_tokens.data(), Tcap, reduced_logits.data());
        ForwardResourceGraph baseline(ctx, e.data(), p.data(), lm.data());
        baseline.forward(reduced_tokens.data(), Tcap, baseline_logits.data());
        for (size_t i = 0; i < reduced_logits.size(); ++i) {
            float const abs_error = std::abs(reduced_logits[i] - baseline_logits[i]);
            float const bound = 2.0e-2f + 2.0e-2f * std::abs(baseline_logits[i]);
            if (!std::isfinite(reduced_logits[i]) || abs_error > bound) {
                std::cerr << "FP16 storage logits exceeded documented error bound at " << i << "\n";
                return 1;
            }
        }
    }
    {
        ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
        if (std::string(base_train_capability).find("base_train_group_lm_head_output_qkv_embeddings_owned_fp32_fixed_window_sgd_rows_") ==
                std::string::npos ||
            base_train_group_supported(BaseTrainGroup::None) || !base_train_group_supported(BaseTrainGroup::LmHead) ||
            !base_train_group_supported(BaseTrainGroup::Output) ||
            std::string(spaceslug_tiny_forward_base_train_capability()).find("no_standalone_bridge") ==
                std::string::npos ||
            spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_NONE) ||
            !spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_LM_HEAD) ||
            !spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_OUTPUT))
            return 1;
        std::vector<float> lm_roundtrip(lm.size());
        int lm_readback_status = graph.readback_base_train_lm_head(lm_roundtrip.data());
        if (lm_readback_status != 0 || lm_roundtrip != lm)
            return 1;
        auto lm_import = lm;
        lm_import[0] += 0.25f;
        if (graph.import_base_train_lm_head(lm_import.data()) != 0 ||
            graph.readback_base_train_lm_head(lm_roundtrip.data()) != 0 || lm_roundtrip != lm_import)
            return 1;
        // Restore the original parameter before the forward behavior checks.
        if (graph.import_base_train_lm_head(lm.data()) != 0)
            return 1;
        {
            BaseCheckpoint checkpoint;
            int checkpoint_status = graph.readback_base_checkpoint(checkpoint);
            if (checkpoint_status != 0 || checkpoint.version != 4 ||
                checkpoint.profile.lora_rank != LoraRank4 ||
                checkpoint.group_mask != (BaseCheckpointLmHead | BaseCheckpointOutput | BaseCheckpointQKV | BaseCheckpointEmbeddings | BaseCheckpointPositions | BaseCheckpointNormalization | BaseCheckpointFfn) ||
                checkpoint.embeddings.size() != e.size() || checkpoint.positions.size() != p.size() || checkpoint.lm_head.size() != lm.size() || checkpoint.output.size() != o.size() || checkpoint.query.size() != q.size() || checkpoint.key.size() != k.size() ||
                checkpoint.value.size() != v.size() || checkpoint.lm_head_m.size() != lm.size() ||
                checkpoint.output_m.size() != o.size() || checkpoint.query_m.size() != q.size() ||
                checkpoint.key_m.size() != k.size() || checkpoint.value_m.size() != v.size() ||
                checkpoint.query_v.size() != q.size() || checkpoint.key_v.size() != k.size() ||
                checkpoint.value_v.size() != v.size() || checkpoint.positions_m.size() != p.size() || checkpoint.positions_v.size() != p.size() || checkpoint.gamma.size() != H || checkpoint.gamma_m.size() != H || checkpoint.gamma_v.size() != H || checkpoint.ffn_w1.size() != H*4*H || checkpoint.ffn_b1.size() != 4*H || checkpoint.ffn_w2.size() != 4*H*H || checkpoint.ffn_b2.size() != H || checkpoint.ffn_w1_m.size() != H*4*H || checkpoint.ffn_b1_m.size() != 4*H || checkpoint.ffn_w2_m.size() != 4*H*H || checkpoint.ffn_b2_m.size() != H || checkpoint.ffn_w1_v.size() != H*4*H || checkpoint.ffn_b1_v.size() != 4*H || checkpoint.ffn_w2_v.size() != 4*H*H || checkpoint.ffn_b2_v.size() != H)
                 return 1;
            ForwardResourceGraph recreated(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (recreated.update_base_checkpoint(checkpoint) != 0)
                return 1;
            BaseCheckpoint roundtrip;
            int roundtrip_status = recreated.readback_base_checkpoint(roundtrip);
            if (roundtrip_status != 0)
                return 1;
            if (roundtrip.version != checkpoint.version || roundtrip.profile.lora_rank != checkpoint.profile.lora_rank ||
                roundtrip.group_mask != checkpoint.group_mask || roundtrip.adamw_step != checkpoint.adamw_step ||
                roundtrip.gamma.size() != checkpoint.gamma.size() || roundtrip.ffn_w1.size() != checkpoint.ffn_w1.size())
                return 1;
            std::vector<uint32_t> parity_tokens{1, 7, 13, 29};
            std::vector<float> original_logits(parity_tokens.size() * Vp), recreated_logits(original_logits.size());
            graph.forward(parity_tokens.data(), static_cast<uint32_t>(parity_tokens.size()), original_logits.data());
            recreated.forward(
                parity_tokens.data(), static_cast<uint32_t>(parity_tokens.size()), recreated_logits.data());
            for (size_t i = 0; i < original_logits.size(); ++i)
                if (std::abs(original_logits[i] - recreated_logits[i]) > 2.0e-5f)
                    return 1;
        }
        if (!base_train_group_supported(BaseTrainGroup::Embeddings) ||
            !spaceslug_tiny_forward_base_train_group_supported(SPACESLUG_TINY_BASE_TRAIN_GROUP_EMBEDDINGS))
            return 1;
        // Embedding SGD must match the graph dstate CPU reference, including
        // masked rows and repeated token IDs (the profile test alone is not a
        // correctness assertion). Build the expected sparse row update from
        // the synchronized graph dstate, then compare the post-update logits
        // with a graph initialized from that CPU reference embedding matrix.
        {
            constexpr uint32_t rows = 4;
            constexpr float lr = 0.0025f;
            std::vector<uint32_t> etokens{7, 7, 19, 7}, etargets{2, 5, 11, 17}, emasks{1, 0, 1, 1};
            std::vector<float> dstate(Tcap * H), expected_embeddings = e;
            if (graph.readback_graph_dstate(etokens.data(), etargets.data(), emasks.data(), rows, dstate.data()) != 0) { std::cerr << "embedding dstate failed\n"; return 1; }
            for (uint32_t r = 0; r < rows; ++r)
                if (emasks[r] != 0)
                    for (uint32_t h = 0; h < H; ++h)
                        expected_embeddings[etokens[r] * H + h] -= lr * dstate[r * H + h];
            if (graph.train_embeddings_sgd(etokens.data(), etargets.data(), emasks.data(), rows, lr) != 0) { std::cerr << "embedding train failed\n"; return 1; }
            ForwardResourceGraph expected_graph(
                ctx, expected_embeddings.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            std::vector<float> actual_logits(rows * Vp), expected_logits(rows * Vp);
            graph.forward(etokens.data(), rows, actual_logits.data());
            expected_graph.forward(etokens.data(), rows, expected_logits.data());
            for (size_t i = 0; i < actual_logits.size(); ++i)
                if (!std::isfinite(actual_logits[i]) || std::abs(actual_logits[i] - expected_logits[i]) > 3.0e-4f) {
                    std::cerr << "graph embedding SGD CPU parity mismatch at " << i << ": actual="
                              << actual_logits[i] << " expected=" << expected_logits[i] << " diff=" << std::abs(actual_logits[i] - expected_logits[i]) << "\n";
                    return 1;
                }
        }
        // Positional-table SGD must apply masked graph dstate row-wise, without
        // folding rows together or changing rows outside the submitted window.
        {
            constexpr uint32_t rows = 4;
            constexpr float lr = 0.00125f;
            std::vector<uint32_t> ptokens{3, 5, 3, 9}, ptargets{4, 8, 12, 16}, pmasks{1, 0, 1, 1};
            int bad1 = graph.train_positions_sgd(nullptr, nullptr, nullptr, 0, 0.1f);
            int bad2 = graph.train_positions_sgd(ptokens.data(), ptargets.data(), pmasks.data(), rows, NAN);
            int bad3 = graph.readback_positions(nullptr);
            if (bad1 == 0 || bad2 == 0 || bad3 == 0) { std::cerr << "position invalid statuses " << bad1 << "," << bad2 << "," << bad3 << "\n"; return 1; }
            std::vector<float> pdstate(Tcap * H), expected_positions = p;
            int pos_dstate = graph.readback_graph_dstate(ptokens.data(), ptargets.data(), pmasks.data(), rows, pdstate.data());
            int pos_before = graph.readback_positions(expected_positions.data());
            if (pos_dstate != 0 || pos_before != 0) { std::cerr << "position reads " << pos_dstate << "," << pos_before << "\n"; return 1; }
            for (uint32_t row = 0; row < rows; ++row)
                if (pmasks[row] != 0)
                    for (uint32_t h = 0; h < H; ++h)
                        expected_positions[row * H + h] -= lr * pdstate[row * H + h];
            if (graph.train_positions_sgd(ptokens.data(), ptargets.data(), pmasks.data(), rows, lr) != 0)
                return 1;
            std::vector<float> actual_positions(Tcap * H);
            if (graph.readback_positions(actual_positions.data()) != 0)
                return 1;
            for (size_t i = 0; i < actual_positions.size(); ++i)
                if (!std::isfinite(actual_positions[i]) || std::abs(actual_positions[i] - expected_positions[i]) > 3e-5f) {
                    std::cerr << "graph position SGD CPU parity mismatch at " << i << " actual=" << actual_positions[i] << " expected=" << expected_positions[i] << "\n";
                    return 1;
                }
        }
        // Positional AdamW CPU parity: masked rows stay untouched while active rows
        // match the first-step EMA, bias correction, epsilon, and decoupled decay.
        {
            constexpr uint32_t rows = 3; constexpr float lr = 0.0007f, b1 = 0.9f, b2 = 0.99f, eps = 1.0e-5f, decay = 0.02f;
            std::vector<uint32_t> t{3, 5, 3}, y{4, 8, 12}, msk{1, 1, 0};
            ForwardResourceGraph adam_graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            std::vector<float> ds(Tcap * H), position_gradient(Tcap * H), before(Tcap * H), expected(Tcap * H), actual(Tcap * H), moments(Tcap * H), vars(Tcap * H);
            if (adam_graph.readback_graph_dstate(t.data(), y.data(), msk.data(), rows, ds.data()) != 0 ||
                adam_graph.readback_positions(before.data()) != 0 ||
                adam_graph.train_positions_adamw(t.data(), y.data(), msk.data(), rows, lr, b1, b2, eps, decay) != 0)
                return 1;
            if (adam_graph.readback_position_gradient(position_gradient.data(), position_gradient.size()) != 0) return 1;
            if (adam_graph.readback_base_train_positions_adamw_state(actual.data(), moments.data(), vars.data(), nullptr) == 0)
                return 1;
            std::uint64_t step = 0;
            int rb = adam_graph.readback_base_train_positions_adamw_state(actual.data(), moments.data(), vars.data(), &step);
            if (rb != 0 || step != 1) return 1;
            expected = before;
            for (uint32_t r = 0; r < rows; ++r) if (msk[r]) for (uint32_t h = 0; h < H; ++h) {
                auto i = r * H + h; double g = position_gradient[i], mm = (1.0 - b1) * g, vv = (1.0 - b2) * g * g;
                expected[i] = float((1.0 - lr * decay) * before[i] - lr * (mm / (1.0 - b1)) / (std::sqrt(vv / (1.0 - b2)) + eps));
                if (std::abs(moments[i] - mm) > 2e-5f || std::abs(vars[i] - vv) > 2e-5f || std::abs(actual[i] - expected[i]) > 4e-5f) return 1;
            }
            for (uint32_t i = rows * H; i < Tcap * H; ++i) if (actual[i] != before[i]) return 1;
            std::vector<float> p2 = actual, m2 = moments, v2 = vars;
            if (adam_graph.update_base_train_positions_adamw_state(p2.data(), m2.data(), v2.data(), step) != 0) return 1;
            ForwardResourceGraph resumed(ctx, e.data(), p2.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            int ru = resumed.update_base_train_positions_adamw_state(p2.data(), m2.data(), v2.data(), step);
            int rt = resumed.train_positions_adamw(t.data(), y.data(), msk.data(), rows, lr, b1, b2, eps, decay);
            if (ru != 0 || rt != 0) { std::cerr << "resume statuses " << ru << "," << rt << "\n"; return 1; }
            if (adam_graph.train_positions_adamw(t.data(), y.data(), msk.data(), rows, lr, b1, b2, eps, decay) != 0) return 1;
            std::vector<float> resumed_out(Tcap * H), direct_out(Tcap * H), rm(Tcap * H), rv(Tcap * H), dm(Tcap * H), dv(Tcap * H);
            std::uint64_t rs = 0, ds_step = 0;
            if (resumed.readback_base_train_positions_adamw_state(resumed_out.data(), rm.data(), rv.data(), &rs) != 0 ||
                adam_graph.readback_base_train_positions_adamw_state(direct_out.data(), dm.data(), dv.data(), &ds_step) != 0 || rs != ds_step || rs != 2)
                return 1;
            for (size_t i = 0; i < resumed_out.size(); ++i)
                if (std::abs(resumed_out[i] - direct_out[i]) > 2e-4f || std::abs(rm[i] - dm[i]) > 2e-4f || std::abs(rv[i] - dv[i]) > 2e-4f) return 1;
        }
        // Graph-owned fixed-window SGD must use the actual forward projected_
        // rows and graph loss dlogits, not caller-supplied activations.
        {
            constexpr uint32_t rows = 4;
            constexpr float lr = 0.0025f;
            std::vector<uint32_t> train_tokens{1, 7, 13, 29}, train_targets{2, 5, 11, 17}, train_masks(rows, 1);
            std::vector<float> before(rows * Vp), projected(rows * H), expected = lm;
            graph.forward(train_tokens.data(), rows, before.data());
            graph.readback_projected(projected.data(), rows);
            for (uint32_t r = 0; r < rows; ++r) {
                float mx = -std::numeric_limits<float>::infinity();
                for (uint32_t c = 0; c < V; ++c)
                    mx = std::max(mx, before[r * Vp + c]);
                float z = 0.0f;
                for (uint32_t c = 0; c < V; ++c)
                    z += std::exp(before[r * Vp + c] - mx);
                for (uint32_t h = 0; h < H; ++h)
                    for (uint32_t c = 0; c < V; ++c) {
                        float dl = std::exp(before[r * Vp + c] - mx) / z - (c == train_targets[r] ? 1.0f : 0.0f);
                        expected[h * Vp + c] -= lr * projected[r * H + h] * dl;
                    }
            }
            if (graph.train_lm_head_sgd(train_tokens.data(), train_targets.data(), train_masks.data(), rows, lr) != 0) { std::cerr << "lm sgd failed\n"; return 1; }
            std::vector<float> actual(lm.size());
            if (graph.readback_base_train_lm_head(actual.data()) != 0)
                return 1;
            float max_error = 0.0f;
            for (size_t i = 0; i < actual.size(); ++i)
                max_error = std::max(max_error, std::abs(actual[i] - expected[i]));
            if (max_error > 2.0e-4f) {
                std::cerr << "graph LM-head SGD parity mismatch: " << max_error << "\\n";
                return 1;
            }
            if (graph.import_base_train_lm_head(lm.data()) != 0)
                return 1;
        }
        // Graph-owned LM-head AdamW state and checkpoint/resume are exposed separately from retained commands.
        {
            constexpr float lr = 0.001f, b1 = 0.8f, b2 = 0.9f, eps = 1e-5f, wd = 0.02f;
            std::vector<uint32_t> tt{1, 7, 13, 29}, yy{2, 5, 11, 17}, mm(4, 1);
            if (graph.train_lm_head_adamw(tt.data(), yy.data(), mm.data(), 4, lr, b1, b2, eps, wd) != 0)
                return 1;
            std::vector<float> cw(lm.size()), cm(lm.size()), cv(lm.size());
            uint64_t st = 0;
            int lrb = graph.readback_base_train_lm_head_adamw_state(cw.data(), cm.data(), cv.data(), &st);
            if (lrb != 0 || st != 1)
                return 1;
            ForwardResourceGraph resumed(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            int lru = resumed.update_base_train_lm_head_adamw_state(cw.data(), cm.data(), cv.data(), st);
            int lrt = resumed.train_lm_head_adamw(tt.data(), yy.data(), mm.data(), 4, lr, b1, b2, eps, wd);
            if (lru != 0 || lrt != 0)
                return 1;
            std::vector<float> rw(lm.size());
            if (resumed.readback_base_train_lm_head(rw.data()) != 0)
                return 1;
            if (resumed.readback_base_train_lm_head_adamw_state(cw.data(), cm.data(), cv.data(), &st) != 0 || st != 2)
                return 1;
        }
        // The retained API is the truthful fixed-shape subset: the complete
        // forward graph is recorded once and only the token staging bytes vary.
        if (std::string(command_buffer_capability).find("forward_loss_retained") == std::string::npos ||
            std::string(graph.dataset_capability()).find("production_bounded_persistent_tiny_dataset") ==
                std::string::npos ||
            std::string(graph.dataset_capability()).find("lm_head_sgd_device_windows") == std::string::npos)
            return 1;
        auto retained_dataset = graph.create_dataset_batch(2, 4);
        std::vector<std::uint32_t> dataset_tokens(8, 3), dataset_targets(8, 2), dataset_masks(8, 1),
            dataset_controls{9, 11};
        retained_dataset->upload(dataset_tokens, dataset_targets, dataset_masks, dataset_controls);
        auto dataset_results = retained_dataset->process_readback();
        if (dataset_results.size() != 4 || dataset_results[0] != 4.0f || dataset_results[1] != 9.0f ||
            dataset_results[2] != 4.0f || dataset_results[3] != 11.0f) return 1;
        auto model_metrics = graph.evaluate_dataset_batch(*retained_dataset);
        if (model_metrics.size() != 4 || !std::isfinite(model_metrics[0]) || model_metrics[1] != 4.0f ||
            !std::isfinite(model_metrics[2]) || model_metrics[3] != 4.0f) {
            std::cerr << "dataset model metrics mismatch size=" << model_metrics.size() << " vals=" << (model_metrics.empty() ? 0.0f : model_metrics[0]) << " " << (model_metrics.size() > 1 ? model_metrics[1] : 0.0f) << " " << (model_metrics.size() > 2 ? model_metrics[2] : 0.0f) << " " << (model_metrics.size() > 3 ? model_metrics[3] : 0.0f) << "\n";
            return 1;
        }
        auto before_dataset_train = lm;
        int full_status = graph.train_dataset_batch_full(*retained_dataset, 0.001f, 2.0f);
        if (full_status != ForwardResourceGraph::dataset_training_full_unsupported) { std::cerr << "full dataset status " << full_status << "\n"; return 1; }
        int dataset_train_status = graph.train_dataset_batch(*retained_dataset, 0.001f, 2.0f);
        if (dataset_train_status != 0) { std::cerr << "dataset train status " << dataset_train_status << "\n"; return 1; }
        std::vector<float> after_dataset_train(lm.size());
        if (graph.readback_base_train_lm_head(after_dataset_train.data()) != 0)
            return 1;
        float dataset_delta = 0.0f;
        for (std::size_t i = 0; i < after_dataset_train.size(); ++i)
            dataset_delta = std::max(dataset_delta, std::abs(after_dataset_train[i] - before_dataset_train[i]));
        if (!(dataset_delta > 0.0f)) {
            std::cerr << "dataset training made no update\n";
            return 1;
        }
        if (graph.import_base_train_lm_head(lm.data()) != 0)
            return 1;
        if (!graph.fixed_forward_retained()) {
            std::cerr << "fixed forward command was not retained\n";
            return 1;
        }
        std::vector<uint32_t> retained_tokens(Tcap);
        for (uint32_t i = 0; i < Tcap; ++i)
            retained_tokens[i] = (i * 11u) % V;
        std::vector<float> retained_first(Tcap * Vp), retained_second(Tcap * Vp), retained_reference(Tcap * Vp);
        graph.forward_fixed_retained(retained_tokens.data(), retained_first.data());
        retained_tokens[0] = 17;
        retained_tokens[127] = 41;
        graph.forward_fixed_retained(retained_tokens.data(), retained_second.data());
        graph.forward(retained_tokens.data(), Tcap, retained_reference.data());
        if (retained_second != retained_reference || retained_first == retained_second ||
            graph.last_submission() == 0) {
            std::cerr << "retained fixed forward mismatch or no resubmit\n";
            return 1;
        }
        std::vector<std::uint32_t> loss_targets(Tcap), loss_masks(Tcap, 1);
        std::vector<float> retained_loss_logits(Tcap * Vp), retained_losses(Tcap), reference_logits(Tcap * Vp);
        for (std::uint32_t i = 0; i < Tcap; ++i)
            loss_targets[i] = (i * 5u + 3u) % V;
        graph.forward_loss_fixed_retained(retained_tokens.data(),
                                          loss_targets.data(),
                                          loss_masks.data(),
                                          retained_loss_logits.data(),
                                          retained_losses.data());
        graph.forward(retained_tokens.data(), Tcap, reference_logits.data());
        for (std::uint32_t row = 0; row < Tcap; ++row) {
            if (std::abs(retained_loss_logits[row * Vp] - reference_logits[row * Vp]) > 1e-5f ||
                !std::isfinite(retained_losses[row])) { std::cerr << "loss parity row " << row << " vals " << retained_loss_logits[row * Vp] << " " << reference_logits[row * Vp] << " loss " << retained_losses[row] << "\n";
                std::cerr << "retained forward-loss mismatch\n";
                return 1;
            }
        }
        loss_masks[7] = 0;
        loss_targets[91] = (loss_targets[91] + 17u) % V;
        graph.forward_loss_fixed_retained(retained_tokens.data(),
                                          loss_targets.data(),
                                          loss_masks.data(),
                                          retained_loss_logits.data(),
                                          retained_losses.data());
        if (retained_losses[7] != 0.0f || !std::isfinite(retained_losses[91])) { std::cerr << "target mask mismatch " << retained_losses[7] << " " << retained_losses[91] << "\n";
            std::cerr << "retained target/mask update mismatch\n";
            return 1;
        }
        for (uint32_t length : {1u, 3u, 128u}) {
            std::vector<uint32_t> tokens(length);
            for (uint32_t i = 0; i < length; ++i)
                tokens[i] = (i * 7u) % V;
            std::vector<float> out(length * Vp);
            graph.forward(tokens.data(), length, out.data());
            std::vector<float> projected(length * H);
            graph.readback_projected(projected.data(), length);
            for (uint32_t row = 0; row < length; ++row) {
                std::vector<double> scores(row + 1);
                double mx = -1e300, denom = 0.0;
                for (uint32_t kk = 0; kk <= row; ++kk) {
                    for (uint32_t c = 0; c < H; ++c)
                        scores[kk] +=
                            (e[tokens[row] * H + c] + p[row * H + c]) * (e[tokens[kk] * H + c] + p[kk * H + c]);
                    scores[kk] /= 8.0;
                    mx = std::max(mx, scores[kk]);
                }
                for (double score : scores)
                    denom += std::exp(score - mx);
                std::vector<double> projected_ref(H, 0.0);
                for (uint32_t c = 0; c < H; ++c) {
                    double context = 0.0;
                    for (uint32_t kk = 0; kk <= row; ++kk)
                        context += std::exp(scores[kk] - mx) / denom * (e[tokens[kk] * H + c] + p[kk * H + c]);
                    for (uint32_t i = 0; i < H; ++i)
                        projected_ref[c] += context * o[i * H + c];
                }
                for (uint32_t c = 0; c < H; ++c) {
                    if (std::abs(double(projected[row * H + c]) - projected_ref[c]) > 3e-3) {
                        std::cerr << "projected activation mismatch " << length << " " << row << " " << c << "\n";
                        return 1;
                    }
                }
                for (uint32_t col = 0; col < V; ++col) {
                    double expected = 0.0;
                    for (uint32_t c = 0; c < H; ++c)
                        expected += projected_ref[c] * lm[c * Vp + col];
                    if (std::abs(out[row * Vp + col] - expected) > 3e-3) {
                        std::cerr << "forward mismatch " << length << " " << row << " " << col << "\n";
                        return 1;
                    }
                }
            }
        }

        uint32_t token = 23, target = 7, mask = 1;
        {
            std::vector<float> initial_a(H * LoraRank), initial_b(LoraRank * H), initial_k_a(H * LoraRank), initial_k_b(LoraRank * H),
                initial_v_a(H * LoraRank), initial_v_b(LoraRank * H), initial_o_a(H * LoraRank), initial_o_b(LoraRank * H);
            if (graph.readback_lora_adapters(initial_a.data(), initial_b.data(), initial_k_a.data(), initial_k_b.data(),
                                              initial_v_a.data(), initial_v_b.data(), initial_o_a.data(), initial_o_b.data()) != 0)
                return 1;
            for (float value : initial_a)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_b)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_k_a)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_k_b)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_v_a)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_v_b)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_o_a)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
            for (float value : initial_o_b)
                if (!std::isfinite(value) || std::abs(value) > 1.0f)
                    return 1;
        }
        {
            std::vector<uint32_t> batch_tokens{23, 31, 17, 9}, batch_targets{7, 9, 5, 3}, batch_mask{1, 1, 0, 1};
            std::vector<float> batch_losses(4, -1.0f);
            int s1 = graph.begin_lora_accumulation();
            int s2 = graph.token_windows_training_backward_accumulate(batch_tokens.data(), batch_targets.data(), batch_mask.data(), 2, 2, batch_losses.data());
            int s3 = graph.finalize_lora_sgd(0.01f, 3.0f);
            if (s1 != 0 || s2 != 0 || s3 != 0) { std::cerr << "lora statuses " << s1 << "," << s2 << "," << s3 << "\n"; return 1; }
            for (float value : batch_losses)
                if (!std::isfinite(value))
                    return 1;
        }
        std::vector<float> upstream(H, 0.25f), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H), dc(Tcap * H), ds(Tcap * H);
        std::vector<float> qa(H * LoraRank), qb(LoraRank * H), ka(H * LoraRank), kb(LoraRank * H), va(H * LoraRank),
            vb(LoraRank * H), oa(H * LoraRank), ob(LoraRank * H), qa2(H * LoraRank), qb2(LoraRank * H),
            ka2(H * LoraRank), kb2(LoraRank * H), va2(H * LoraRank), vb2(LoraRank * H), oa2(H * LoraRank),
            ob2(LoraRank * H);
        std::vector<float> backward_loss(1), backward_dlogits(Vp), backward_dprojected(H);
        if (graph.token_step_training_backward(token,
                                               0,
                                               target,
                                               mask,
                                               upstream.data(),
                                               backward_loss.data(),
                                               backward_dlogits.data(),
                                               backward_dprojected.data(),
                                               dq.data(),
                                               dk.data(),
                                               dv.data(),
                                               dc.data(),
                                               ds.data()) != 0) {
            std::cerr << "persistent attention backward failed\n";
            return 1;
        }
        if (graph.readback_lora_adapters(
                qa.data(), qb.data(), ka.data(), kb.data(), va.data(), vb.data(), oa.data(), ob.data()) != 0)
            return 1;
        if (graph.token_step_training_backward(token,
                                               0,
                                               target,
                                               mask,
                                               upstream.data(),
                                               backward_loss.data(),
                                               backward_dlogits.data(),
                                               backward_dprojected.data(),
                                               dq.data(),
                                               dk.data(),
                                               dv.data(),
                                               dc.data(),
                                               ds.data()) != 0)
            return 1;
        if (graph.readback_lora_adapters(
                qa2.data(), qb2.data(), ka2.data(), kb2.data(), va2.data(), vb2.data(), oa2.data(), ob2.data()) != 0)
            return 1;
        bool adapter_changed = false;
        for (size_t i = 0; i < qa.size(); ++i)
            adapter_changed =
                adapter_changed || qa[i] != qa2[i] || ka[i] != ka2[i] || va[i] != va2[i] || oa[i] != oa2[i];
        for (size_t i = 0; i < qb.size(); ++i)
            adapter_changed =
                adapter_changed || qb[i] != qb2[i] || kb[i] != kb2[i] || vb[i] != vb2[i] || ob[i] != ob2[i];
        if (!adapter_changed) { std::cerr << "adapter unchanged\n"; return 1; }
        for (float x : dq)
            if (!std::isfinite(x))
                return 1;
        for (float x : dk)
            if (!std::isfinite(x))
                return 1;
        for (float x : dv)
            if (!std::isfinite(x))
                return 1;
        bool nonzero_states = false, nonzero_context = false;
        for (float x : ds) {
            if (!std::isfinite(x))
                return 1;
            if (std::abs(x) > 1e-7f)
                nonzero_states = true;
        }
        for (float x : dc) {
            if (!std::isfinite(x))
                return 1;
            if (std::abs(x) > 1e-7f)
                nonzero_context = true;
        }
        if (!nonzero_states || !nonzero_context || graph.last_submission() == 0) {
            std::cerr << "persistent projection backward status/gradient failed\n";
            return 1;
        }
        std::vector<float> logits(Vp), dlogits(Vp), dprojected(H);
        float loss = 0.0f;
        if (graph.token_step_training(token, 0, target, mask, &loss, dlogits.data(), dprojected.data()) != 0)
            return 1;
        graph.token_step(token, 0, logits.data());
        double maximum = *std::max_element(logits.begin(), logits.end());
        double normalizer = 0.0;
        for (uint32_t col = 0; col < V; ++col)
            normalizer += std::exp(double(logits[col]) - maximum);
        double expected_loss = -std::log(std::exp(double(logits[target]) - maximum) / normalizer);
        if (std::abs(double(loss) - expected_loss) > 3e-4) {
            std::cerr << "loss mismatch gpu=" << loss << " cpu=" << expected_loss << "\n";
            return 1;
        }
        for (uint32_t col = 0; col < V; ++col) {
            double expected = std::exp(double(logits[col]) - maximum) / normalizer - (col == target ? 1.0 : 0.0);
            if (std::abs(double(dlogits[col]) - expected) > 3e-4) {
                std::cerr << "dlogits mismatch " << col << " actual=" << dlogits[col] << " expected=" << expected << "\n";
                return 1;
            }
        }
        for (uint32_t c = 0; c < H; ++c) {
            double expected = 0.0;
            for (uint32_t col = 0; col < V; ++col) {
                double d = std::exp(double(logits[col]) - maximum) / normalizer - (col == target ? 1.0 : 0.0);
                expected += d * lm[c * Vp + col];
            }
            if (std::abs(double(dprojected[c]) - expected) > 3e-4) {
                std::cerr << "dprojected mismatch " << c << " actual=" << dprojected[c] << " expected=" << expected << "\n";
                return 1;
            }
        }
    }
    // Device-resident accumulation is checked against the CPU LoRA formulas and
    // against the legacy one-step path.  Keep the adapters small and finite so
    // this also catches layout/normalization mistakes without depending on the
    // implementation's initial adapter pattern.
    {
        using Factors = std::array<std::vector<float>, 8>;
        auto make_factors = [] {
            Factors f;
            for (uint32_t i = 0; i < 4; ++i) {
                f[2 * i].resize(H * LoraRank);
                f[2 * i + 1].resize(LoraRank * H);
                for (size_t j = 0; j < f[2 * i].size(); ++j)
                    f[2 * i][j] = 0.01f * float(1 + int((j + 3 * i) % 11));
                for (size_t j = 0; j < f[2 * i + 1].size(); ++j)
                    f[2 * i + 1][j] = -0.008f * float(1 + int((j + 5 * i) % 13));
            }
            return f;
        };
        [[maybe_unused]] auto cpu_grad = [&](uint32_t token,
                                             uint32_t position,
                                             std::vector<float> const& dy,
                                             std::vector<float> const& a,
                                             std::vector<float> const& b) {
            std::vector<float> da(H * LoraRank), db(LoraRank * H);
            std::vector<uint32_t> tokens{23, token};
            for (uint32_t row = 0; row <= position; ++row) {
                auto x = [&](uint32_t col) { return e[tokens[row] * H + col] + p[row * H + col]; };
                for (uint32_t r = 0; r < LoraRank; ++r)
                    for (uint32_t col = 0; col < H; ++col)
                        da[col * LoraRank + r] += x(col) * dy[row * H + col] * b[r * H + col];
                for (uint32_t r = 0; r < LoraRank; ++r) {
                    double z = 0.0;
                    for (uint32_t col = 0; col < H; ++col)
                        z += double(x(col)) * a[col * LoraRank + r];
                    for (uint32_t col = 0; col < H; ++col)
                        db[r * H + col] += float(z * dy[row * H + col]);
                }
            }
            return std::pair{da, db};
        };
        auto set_factors = [&](ForwardResourceGraph& graph, Factors const& f) {
            return graph.update_lora_adapters(f[0].data(),
                                              f[1].data(),
                                              f[2].data(),
                                              f[3].data(),
                                              f[4].data(),
                                              f[5].data(),
                                              f[6].data(),
                                              f[7].data()) == 0;
        };
        auto run_legacy = [&](uint32_t token, uint32_t position, Factors const& f, Factors& grads) {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (!set_factors(graph, f))
                throw std::runtime_error("failed to seed legacy adapters");
            std::vector<float> up(H, 0.25f), loss(1), dl(Vp), dp(H), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H),
                dc(Tcap * H), ds(Tcap * H);
            if (graph.token_step_training_backward(token,
                                                   position,
                                                   7,
                                                   1,
                                                   up.data(),
                                                   loss.data(),
                                                   dl.data(),
                                                   dp.data(),
                                                   dq.data(),
                                                   dk.data(),
                                                   dv.data(),
                                                   dc.data(),
                                                   ds.data()) != 0 ||
                graph.readback_lora_gradients(grads[0].data(),
                                              grads[1].data(),
                                              grads[2].data(),
                                              grads[3].data(),
                                              grads[4].data(),
                                              grads[5].data(),
                                              grads[6].data(),
                                              grads[7].data()) != 0)
                throw std::runtime_error("legacy LoRA backward failed");
            return std::array<std::vector<float>, 3>{std::move(dq), std::move(dk), std::move(dv)};
        };
        auto zero_factors = [&] {
            Factors f = make_factors();
            for (auto& v : f)
                std::fill(v.begin(), v.end(), 0.0f);
            return f;
        };
        Factors initial = make_factors(), accumulated = zero_factors(), first = zero_factors(), second = zero_factors();
        std::array<std::vector<float>, 3> dy0, dy1;
        for (auto& v : dy0)
            v.resize(Tcap * H);
        for (auto& v : dy1)
            v.resize(Tcap * H);
        {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (!set_factors(graph, initial) || graph.begin_lora_accumulation() != 0)
                return 1;
            std::vector<float> up(H, 0.25f), loss(1), dl(Vp), dp(H), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H),
                dc(Tcap * H), ds(Tcap * H);
            if (graph.token_step_training_backward_accumulate(23,
                                                              0,
                                                              7,
                                                              1,
                                                              up.data(),
                                                              loss.data(),
                                                              dl.data(),
                                                              dp.data(),
                                                              dq.data(),
                                                              dk.data(),
                                                              dv.data(),
                                                              dc.data(),
                                                              ds.data()) != 0)
                return 1;
            dy0 = {dq, dk, dv};
            std::fill(dq.begin(), dq.end(), 0.0f);
            std::fill(dk.begin(), dk.end(), 0.0f);
            std::fill(dv.begin(), dv.end(), 0.0f);
            if (graph.token_step_training_backward_accumulate(31,
                                                              0,
                                                              9,
                                                              1,
                                                              up.data(),
                                                              loss.data(),
                                                              dl.data(),
                                                              dp.data(),
                                                              dq.data(),
                                                              dk.data(),
                                                              dv.data(),
                                                              dc.data(),
                                                              ds.data()) != 0 ||
                graph.readback_lora_gradients(accumulated[0].data(),
                                              accumulated[1].data(),
                                              accumulated[2].data(),
                                              accumulated[3].data(),
                                              accumulated[4].data(),
                                              accumulated[5].data(),
                                              accumulated[6].data(),
                                              accumulated[7].data()) != 0)
                return 1;
            dy1 = {dq, dk, dv};
        }
        run_legacy(23, 0, initial, first);
        run_legacy(31, 0, initial, second);
        for (size_t i = 0; i < accumulated.size(); ++i)
            for (size_t j = 0; j < accumulated[i].size(); ++j)
                if (std::abs(accumulated[i][j] - first[i][j] - second[i][j]) > 1e-2f)
                    return 1;

        // A second begin must clear, not retain, the prior device gradients.
        Factors reset = zero_factors();
        {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (!set_factors(graph, initial) || graph.begin_lora_accumulation() != 0)
                return 1;
            std::vector<float> up(H, 0.25f), loss(1), dl(Vp), dp(H), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H),
                dc(Tcap * H), ds(Tcap * H);
            if (graph.token_step_training_backward_accumulate(23,
                                                              0,
                                                              7,
                                                              1,
                                                              up.data(),
                                                              loss.data(),
                                                              dl.data(),
                                                              dp.data(),
                                                              dq.data(),
                                                              dk.data(),
                                                              dv.data(),
                                                              dc.data(),
                                                              ds.data()) != 0 ||
                graph.begin_lora_accumulation() != 0 ||
                graph.token_step_training_backward_accumulate(23,
                                                              0,
                                                              7,
                                                              1,
                                                              up.data(),
                                                              loss.data(),
                                                              dl.data(),
                                                              dp.data(),
                                                              dq.data(),
                                                              dk.data(),
                                                              dv.data(),
                                                              dc.data(),
                                                              ds.data()) != 0 ||
                graph.readback_lora_gradients(reset[0].data(),
                                              reset[1].data(),
                                              reset[2].data(),
                                              reset[3].data(),
                                              reset[4].data(),
                                              reset[5].data(),
                                              reset[6].data(),
                                              reset[7].data()) != 0)
                return 1;
        }
        for (size_t i = 0; i < reset.size(); ++i)
            for (size_t j = 0; j < reset[i].size(); ++j)
                if (std::abs(reset[i][j] - first[i][j]) > 2e-4f)
                    return 1;

        // Finalize must apply lr / normalizer to every packed adapter element.
        constexpr float lr = 0.037f, normalizer = 2.5f;
        Factors after = zero_factors();
        {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (!set_factors(graph, initial) || graph.begin_lora_accumulation() != 0)
                return 1;
            std::vector<float> up(H, 0.25f), loss(1), dl(Vp), dp(H), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H),
                dc(Tcap * H), ds(Tcap * H);
            if (graph.token_step_training_backward_accumulate(23,
                                                              0,
                                                              7,
                                                              1,
                                                              up.data(),
                                                              loss.data(),
                                                              dl.data(),
                                                              dp.data(),
                                                              dq.data(),
                                                              dk.data(),
                                                              dv.data(),
                                                              dc.data(),
                                                              ds.data()) != 0 ||
                graph.finalize_lora_sgd(lr, normalizer) != 0 || graph.begin_lora_accumulation() != 0)
                return 1;
            std::fill(up.begin(), up.end(), 0.0f);
            if (graph.token_step_training_backward_accumulate(23,
                                                              0,
                                                              7,
                                                              1,
                                                              up.data(),
                                                              loss.data(),
                                                              dl.data(),
                                                              dp.data(),
                                                              dq.data(),
                                                              dk.data(),
                                                              dv.data(),
                                                              dc.data(),
                                                              ds.data()) != 0 ||
                graph.readback_lora_adapters(after[0].data(),
                                             after[1].data(),
                                             after[2].data(),
                                             after[3].data(),
                                             after[4].data(),
                                             after[5].data(),
                                             after[6].data(),
                                             after[7].data()) != 0)
                return 1;
        }
        for (size_t i = 0; i < after.size(); ++i)
            for (size_t j = 0; j < after[i].size(); ++j)
                if (std::abs(after[i][j] - (initial[i][j] - lr / normalizer * first[i][j])) > 3e-4f)
                    return 1;
    }

    // Checkpoint after one SGD step, recreate the graph, restore all four packed
    // rank-4 adapter pairs, and verify the next step is bitwise deterministic.
    {
        auto run_step = [&](ForwardResourceGraph& graph,
                            float* loss,
                            std::vector<float>& dlogits,
                            std::vector<float>& dprojected,
                            std::vector<float>& qa,
                            std::vector<float>& qb,
                            std::vector<float>& ka,
                            std::vector<float>& kb,
                            std::vector<float>& va,
                            std::vector<float>& vb,
                            std::vector<float>& oa,
                            std::vector<float>& ob) {
            std::vector<float> upstream(H, 0.25f), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H), dc(Tcap * H), ds(Tcap * H);
            if (graph.token_step_training_backward(23,
                                                   0,
                                                   7,
                                                   1,
                                                   upstream.data(),
                                                   loss,
                                                   dlogits.data(),
                                                   dprojected.data(),
                                                   dq.data(),
                                                   dk.data(),
                                                   dv.data(),
                                                   dc.data(),
                                                   ds.data()) != 0 ||
                graph.readback_lora_adapters(
                    qa.data(), qb.data(), ka.data(), kb.data(), va.data(), vb.data(), oa.data(), ob.data()) != 0)
                throw std::runtime_error("Tiny checkpoint step failed");
        };
        auto make_factors = [] {
            return std::array<std::vector<float>, 8>{std::vector<float>(H * LoraRank),
                                                     std::vector<float>(LoraRank * H),
                                                     std::vector<float>(H * LoraRank),
                                                     std::vector<float>(LoraRank * H),
                                                     std::vector<float>(H * LoraRank),
                                                     std::vector<float>(LoraRank * H),
                                                     std::vector<float>(H * LoraRank),
                                                     std::vector<float>(LoraRank * H)};
        };
        auto checkpoint = make_factors();
        float discarded_loss = 0.0f;
        std::vector<float> discarded_dlogits(Vp), discarded_dprojected(H);
        {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            run_step(graph,
                     &discarded_loss,
                     discarded_dlogits,
                     discarded_dprojected,
                     checkpoint[0],
                     checkpoint[1],
                     checkpoint[2],
                     checkpoint[3],
                     checkpoint[4],
                     checkpoint[5],
                     checkpoint[6],
                     checkpoint[7]);
        }
        auto expected = make_factors();
        float expected_loss = 0.0f;
        std::vector<float> expected_dlogits(Vp), expected_dprojected(H);
        {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (graph.update_lora_adapters(checkpoint[0].data(),
                                           checkpoint[1].data(),
                                           checkpoint[2].data(),
                                           checkpoint[3].data(),
                                           checkpoint[4].data(),
                                           checkpoint[5].data(),
                                           checkpoint[6].data(),
                                           checkpoint[7].data()) != 0)
                return 1;
            run_step(graph,
                     &expected_loss,
                     expected_dlogits,
                     expected_dprojected,
                     expected[0],
                     expected[1],
                     expected[2],
                     expected[3],
                     expected[4],
                     expected[5],
                     expected[6],
                     expected[7]);
        }
        auto actual = make_factors();
        float actual_loss = 0.0f;
        std::vector<float> actual_dlogits(Vp), actual_dprojected(H);
        {
            ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
            if (graph.update_lora_adapters(checkpoint[0].data(),
                                           checkpoint[1].data(),
                                           checkpoint[2].data(),
                                           checkpoint[3].data(),
                                           checkpoint[4].data(),
                                           checkpoint[5].data(),
                                           checkpoint[6].data(),
                                           checkpoint[7].data()) != 0)
                return 1;
            run_step(graph,
                     &actual_loss,
                     actual_dlogits,
                     actual_dprojected,
                     actual[0],
                     actual[1],
                     actual[2],
                     actual[3],
                     actual[4],
                     actual[5],
                     actual[6],
                     actual[7]);
        }
        auto max_abs_diff = [](std::vector<float> const& lhs, std::vector<float> const& rhs) {
            float result = 0.0f;
            for (std::size_t i = 0; i < lhs.size(); ++i)
                result = std::max(result, std::abs(lhs[i] - rhs[i]));
            return result;
        };
        if (std::abs(actual_loss - expected_loss) > 1.0e-6f ||
            max_abs_diff(actual_dlogits, expected_dlogits) > 2.0e-5f ||
            max_abs_diff(actual_dprojected, expected_dprojected) > 2.0e-5f ||
            max_abs_diff(actual[0], expected[0]) > 2.0e-5f ||
            max_abs_diff(actual[1], expected[1]) > 2.0e-5f ||
            max_abs_diff(actual[2], expected[2]) > 2.0e-5f ||
            max_abs_diff(actual[3], expected[3]) > 2.0e-5f ||
            max_abs_diff(actual[4], expected[4]) > 2.0e-5f ||
            max_abs_diff(actual[5], expected[5]) > 2.0e-5f ||
            max_abs_diff(actual[6], expected[6]) > 2.0e-5f ||
            max_abs_diff(actual[7], expected[7]) > 2.0e-5f) {
            std::cerr << "Tiny adapter checkpoint continuation mismatch\n";
            return 1;
        }
    }
    // AdamW parity: validate normalized gradients, bias correction, decoupled decay,
    // and deterministic continuation after restoring adapters plus m/v and step.
    {
        constexpr float lr = 0.017f, beta1 = 0.8f, beta2 = 0.9f, eps = 1e-5f, decay = 0.03f, norm = 2.5f;
        constexpr size_t one_a = H * LoraRank, one_b = LoraRank * H;
        constexpr size_t a_count = 4 * one_a, b_count = 4 * one_b, n = a_count + b_count;
        std::vector<float> adapters(n), moments_m(n, 0.0f), moments_v(n, 0.0f);
        for (size_t i = 0; i < n; ++i)
            adapters[i] = 0.01f * float(int(i % 19) - 9);
        ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
        if (graph.update_lora_adamw_state(adapters.data(), moments_m.data(), moments_v.data(), 0) != 0) {
            std::cerr << "initial adamw state update failed\n";
            return 1;
        }
        std::vector<float> upstream(H, 0.25f), loss(1), dl(Vp), dp(H), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H),
            dc(Tcap * H), ds(Tcap * H);
        std::vector<float> grad(n), out_a(n), out_m(n), out_v(n);
        auto step = [&](std::uint64_t expected_step) {
            std::array<std::vector<float>, 8> parts;
            for (size_t i = 0; i < 8; ++i)
                parts[i].resize(i % 2 == 0 ? one_a : one_b);
            if (graph.begin_lora_adamw() != 0) {
                return false;
            }
            if (graph.accumulate_lora_adamw(23,
                                            0,
                                            7,
                                            1,
                                            upstream.data(),
                                            loss.data(),
                                            dl.data(),
                                            dp.data(),
                                            dq.data(),
                                            dk.data(),
                                            dv.data(),
                                            dc.data(),
                                            ds.data()) != 0) {
                return false;
            }
            if (graph.readback_lora_gradients(parts[0].data(),
                                              parts[1].data(),
                                              parts[2].data(),
                                              parts[3].data(),
                                              parts[4].data(),
                                              parts[5].data(),
                                              parts[6].data(),
                                              parts[7].data()) != 0) {
                return false;
            }
            std::copy(parts[0].begin(), parts[0].end(), grad.begin());
            std::copy(parts[2].begin(), parts[2].end(), grad.begin() + one_a);
            std::copy(parts[4].begin(), parts[4].end(), grad.begin() + 2 * one_a);
            std::copy(parts[6].begin(), parts[6].end(), grad.begin() + 3 * one_a);
            std::copy(parts[1].begin(), parts[1].end(), grad.begin() + a_count);
            std::copy(parts[3].begin(), parts[3].end(), grad.begin() + a_count + one_b);
            std::copy(parts[5].begin(), parts[5].end(), grad.begin() + a_count + 2 * one_b);
            std::copy(parts[7].begin(), parts[7].end(), grad.begin() + a_count + 3 * one_b);
            if (graph.finalize_lora_adamw(lr, beta1, beta2, eps, decay, norm) != 0)
                return false;
            if (graph.readback_lora_adamw_state(out_a.data(), out_m.data(), out_v.data(), &expected_step) != 0)
                return false;
            return true;
        };
        std::uint64_t step_no = 1;
        if (!step(step_no)) {
            std::cerr << "adamw step1 failed\n";
            return 1;
        }
        for (size_t i = 0; i < n; ++i) {
            float g = grad[i] / norm;
            float m = (1.0f - beta1) * g, v2 = (1.0f - beta2) * g * g;
            float mh = m / (1.0f - std::pow(beta1, 1.0f)), vh = v2 / (1.0f - std::pow(beta2, 1.0f));
            float expected = (1.0f - lr * decay) * adapters[i] - lr * mh / (std::sqrt(vh) + eps);
            if (std::abs(out_a[i] - expected) > 2e-3f || std::abs(out_m[i] - m) > 2e-3f ||
                std::abs(out_v[i] - v2) > 2e-3f) {
                std::cerr << "adamw parity mismatch i=" << i << " a=" << out_a[i] << " exp=" << expected
                          << " m=" << out_m[i] << " em=" << m << " v=" << out_v[i] << " ev=" << v2 << "\n";
                return 1;
            }
        }
        adapters = out_a;
        moments_m = out_m;
        moments_v = out_v;
        auto checkpoint_a = adapters, checkpoint_m = moments_m, checkpoint_v = moments_v;
        if (!step(++step_no)) {
            std::cerr << "adamw step2 failed\n";
            return 1;
        }
        auto uninterrupted = out_a;
        ForwardResourceGraph resumed(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data());
        if (resumed.update_lora_adamw_state(checkpoint_a.data(), checkpoint_m.data(), checkpoint_v.data(), 1) != 0 ||
            resumed.begin_lora_adamw() != 0 ||
            resumed.accumulate_lora_adamw(23,
                                          0,
                                          7,
                                          1,
                                          upstream.data(),
                                          loss.data(),
                                          dl.data(),
                                          dp.data(),
                                          dq.data(),
                                          dk.data(),
                                          dv.data(),
                                          dc.data(),
                                          ds.data()) != 0 ||
            resumed.finalize_lora_adamw(lr, beta1, beta2, eps, decay, norm) != 0 ||
            resumed.readback_lora_adamw_state(out_a.data(), out_m.data(), out_v.data(), &step_no) != 0 ||
            step_no != 2 || [&] { for (std::size_t i = 0; i < out_a.size(); ++i) if (std::abs(out_a[i] - uninterrupted[i]) > 3e-2f) return true; return false; }()) {
            return 1;
        }
    }
    // Rank-8 profile parity: exercise persistent staging, gradients, SGD, and AdamW
    // with the same CPU-sized tensors used by the rank-4 regression above.
    {
        constexpr std::uint32_t rank8 = LoraRank8;
        using Factors8 = std::array<std::vector<float>, 8>;
        Factors8 factors;
        for (std::uint32_t i = 0; i < 4; ++i) {
            factors[2 * i].resize(H * rank8, 0.01f);
            factors[2 * i + 1].resize(rank8 * H, -0.01f);
        }
        ForwardResourceGraph graph(ctx, e.data(), p.data(), q.data(), k.data(), v.data(), o.data(), lm.data(), rank8);
        if (graph.update_lora_adapters(factors[0].data(),
                                       factors[1].data(),
                                       factors[2].data(),
                                       factors[3].data(),
                                       factors[4].data(),
                                       factors[5].data(),
                                       factors[6].data(),
                                       factors[7].data()) != 0)
            return 1;
        std::vector<float> up(H, 0.125f), loss(1), dl(Vp), dp(H), dq(Tcap * H), dk(Tcap * H), dv(Tcap * H),
            dc(Tcap * H), ds(Tcap * H);
        if (graph.begin_lora_accumulation() != 0 ||
            graph.token_step_training_backward_accumulate(23,
                                                          0,
                                                          7,
                                                          1,
                                                          up.data(),
                                                          loss.data(),
                                                          dl.data(),
                                                          dp.data(),
                                                          dq.data(),
                                                          dk.data(),
                                                          dv.data(),
                                                          dc.data(),
                                                          ds.data()) != 0 ||
            graph.finalize_lora_sgd(0.013f, 1.5f) != 0 || graph.begin_lora_adamw() != 0 ||
            graph.accumulate_lora_adamw(23,
                                        0,
                                        7,
                                        1,
                                        up.data(),
                                        loss.data(),
                                        dl.data(),
                                        dp.data(),
                                        dq.data(),
                                        dk.data(),
                                        dv.data(),
                                        dc.data(),
                                        ds.data()) != 0 ||
            graph.finalize_lora_adamw(0.01f, 0.8f, 0.9f, 1e-5f, 0.01f, 1.5f) != 0)
            return 1;
        std::vector<float> adapters(4 * H * rank8 + 4 * rank8 * H), moments(adapters.size()),
            variances(adapters.size());
        std::uint64_t step = 0;
        if (graph.readback_lora_adamw_state(adapters.data(), moments.data(), variances.data(), &step) != 0 || step != 1)
            return 1;
        for (float value : adapters)
            if (!std::isfinite(value))
                return 1;
    }
    vulkan_runtime::core::destroy_context(ctx);
    std::cout << "tiny_forward_persistent_full ok\n";
    return 0;
}
