#pragma once

/// @file ops.hpp
/// @brief TransformerRunner — flexible Llama/GPT-J/Phi-style forward pass.
/// @ingroup tensorbit-run
///
/// Auto-detects model naming conventions (Llama, GPT-J, Phi) by probing
/// tensor names.  Handles sparse N:M and dense linear paths with GQA support.

#include "tensorbit/run/backend.hpp"
#include "tensorbit/run/common.hpp"
#include "tensorbit/run/model.hpp"
#include "tensorbit/run/tensor.hpp"

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace tensorbit {
namespace run {

struct RunState {
    Tensor x;
    Tensor residual;
    int    seq_pos = 0;
    std::vector<Tensor> k_cache;
    std::vector<Tensor> v_cache;
};

class TransformerRunner {
public:
    explicit TransformerRunner(const Model& model) : model_(model), state_() {}

    void init_state() {
        auto cfg = model_.config();
        int  hidden = cfg.hidden_size > 0 ? cfg.hidden_size : 512;
        int  n_layers = cfg.num_layers > 0 ? cfg.num_layers : 1;

        state_.x = Tensor({(size_t)std::max(hidden, cfg.vocab_size)}, Dtype::kF32);
        state_.residual = Tensor({(size_t)hidden}, Dtype::kF32);
        state_.seq_pos = 0;

        state_.k_cache.resize(n_layers);
        state_.v_cache.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            int n_kv = cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.num_heads;
            size_t hd = (cfg.head_dim > 0) ? (size_t)cfg.head_dim :
                         (size_t)hidden / (cfg.num_heads > 0 ? cfg.num_heads : 1);
            size_t kv_dim = (size_t)n_kv * hd;
            state_.k_cache[i] = Tensor({(size_t)cfg.max_seq_len, kv_dim}, Dtype::kF32);
            state_.v_cache[i] = Tensor({(size_t)cfg.max_seq_len, kv_dim}, Dtype::kF32);
        }
    }

    void reset() {
        state_.seq_pos = 0;
    }

    int vocab_size() const { return model_.config().vocab_size; }
    const float* logits() const { return state_.x.f32(); }

    /// Flexible layer name search: tries multiple naming conventions.
    static std::string find_weight_name(const Model& m, int layer, const char* component) {
        // Llama/Mistral: model.layers.{layer}.{component}.weight
        std::string llama = "model.layers." + std::to_string(layer) + "." + component + ".weight";
        if (m.find_layer(llama) >= 0) return llama;

        // GPT-J: transformer.h.{layer}.{component}.weight
        std::string gptj = "transformer.h." + std::to_string(layer) + "." + component + ".weight";
        if (m.find_layer(gptj) >= 0) return gptj;

        // Phi: transformer.h.{layer}.{component}.weight (same as GPT-J)
        // Phi also uses model.layers.{layer}.{component}.weight for some releases
        // Try bare component (no layer prefix — some optimised configs):
        std::string bare = component + std::string(".weight");
        if (m.find_layer(bare) >= 0) return bare;

        return "";  // not found
    }

    /// Find a named weight layer in the model, 0..num_layers-1, or -1.
    static int find_weight_idx(const Model& m, const std::string& name) {
        if (name.empty()) return -1;
        return m.find_layer(name);
    }

    /// Single token forward pass.
    const float* forward(int token_id) {
        auto cfg = model_.config();
        int  hidden = cfg.hidden_size > 0 ? cfg.hidden_size : 512;
        int  n_layers = cfg.num_layers > 0 ? cfg.num_layers : 1;
        int  n_heads = cfg.num_heads > 0 ? cfg.num_heads : 8;
        int  n_kv_heads = cfg.num_kv_heads > 0 ? cfg.num_kv_heads : n_heads;
        int  head_dim = cfg.head_dim > 0 ? cfg.head_dim : (hidden / n_heads);

        if (head_dim <= 0 || hidden <= 0) return nullptr;

        // Embedding
        int emb_idx = -1;
        for (auto& name : {"model.embed_tokens.weight", "transformer.wte.weight", "transformer.embd.wte.weight"}) {
            emb_idx = model_.find_layer(name);
            if (emb_idx >= 0) break;
        }
        if (emb_idx < 0) return nullptr;
        Tensor emb = weight_fp32(emb_idx);
        if (!emb.data()) return nullptr;
        embedding(state_.x, &token_id, 1, emb);

        // Layer loop
        for (int l = 0; l < n_layers; l++) {
            auto Qw = find_weight_name(model_, l, "self_attn.q_proj");
            auto Kw = find_weight_name(model_, l, "self_attn.k_proj");
            auto Vw = find_weight_name(model_, l, "self_attn.v_proj");
            auto Ow = find_weight_name(model_, l, "self_attn.o_proj");
            auto Gw = find_weight_name(model_, l, "mlp.gate_proj");
            auto Uw = find_weight_name(model_, l, "mlp.up_proj");
            auto Dw = find_weight_name(model_, l, "mlp.down_proj");
            auto In = find_weight_name(model_, l, "input_layernorm");
            auto Pn = find_weight_name(model_, l, "post_attention_layernorm");

            int wq_i = find_weight_idx(model_, Qw);
            int wk_i = find_weight_idx(model_, Kw);
            int wv_i = find_weight_idx(model_, Vw);
            if (wq_i < 0 || wk_i < 0 || wv_i < 0) continue;  // no attention — skip

            int wo_i = find_weight_idx(model_, Ow);
            int wg_i = find_weight_idx(model_, Gw);
            int wu_i = find_weight_idx(model_, Uw);
            int wd_i = find_weight_idx(model_, Dw);
            int in_i = find_weight_idx(model_, In);
            int pn_i = find_weight_idx(model_, Pn);

            // --- RMSNorm before attention ---
            Tensor rms_w = in_i >= 0 ? weight_fp32(in_i) : Tensor();
            rms_norm(state_.x, state_.x, rms_w, cfg.norm_eps);

            // --- QKV projections ---
            int q_dim = n_heads * head_dim;
            int kv_dim_total = n_kv_heads * head_dim;

            Tensor q = Tensor({(size_t)n_heads, (size_t)1, (size_t)head_dim}, Dtype::kF32);
            Tensor k = Tensor({(size_t)n_kv_heads, (size_t)1, (size_t)head_dim}, Dtype::kF32);
            Tensor v = Tensor({(size_t)n_kv_heads, (size_t)1, (size_t)head_dim}, Dtype::kF32);

            Tensor Wq = weight_fp32(wq_i);
            Tensor Wk = weight_fp32(wk_i);
            Tensor Wv = weight_fp32(wv_i);
            if (!Wq.data() || !Wk.data() || !Wv.data()) continue;

            Tensor Mq = model_.get_mask(wq_i);
            Tensor q_flat = Tensor({(size_t)q_dim}, Dtype::kF32);
            Tensor k_flat = Tensor({(size_t)kv_dim_total}, Dtype::kF32);
            Tensor v_flat = Tensor({(size_t)kv_dim_total}, Dtype::kF32);

            if (Mq.data()) {
                sparse_linear(q_flat, state_.x, Wq, Mq, Tensor());
                sparse_linear(k_flat, state_.x, Wk, model_.get_mask(wk_i), Tensor());
                sparse_linear(v_flat, state_.x, Wv, model_.get_mask(wv_i), Tensor());
            } else {
                dense_linear(q_flat, state_.x, Wq, Tensor());
                dense_linear(k_flat, state_.x, Wk, Tensor());
                dense_linear(v_flat, state_.x, Wv, Tensor());
            }
            // Reshape to [heads, 1, head_dim]
            for (int h = 0; h < n_heads; h++)
                std::memcpy(q.f32() + h * head_dim, q_flat.f32() + h * head_dim, head_dim * sizeof(float));
            for (int h = 0; h < n_kv_heads; h++) {
                std::memcpy(k.f32() + h * head_dim, k_flat.f32() + h * head_dim, head_dim * sizeof(float));
                std::memcpy(v.f32() + h * head_dim, v_flat.f32() + h * head_dim, head_dim * sizeof(float));
            }

            // --- RoPE ---
            rope(q, k, state_.seq_pos, cfg.rope_theta, head_dim);

            // --- KV cache: write at current position ---
            int pos = state_.seq_pos;
            size_t kv_row = kv_dim_total;
            {
                float* kc = state_.k_cache[l].f32() + pos * kv_row;
                float* vc = state_.v_cache[l].f32() + pos * kv_row;
                std::memcpy(kc, k.f32(), kv_row * sizeof(float));
                std::memcpy(vc, v.f32(), kv_row * sizeof(float));
            }

            // --- Attention ---
            int attn_out_size = q_dim;  // prefill: 1 token
            if (pos > 0) attn_out_size *= (pos + 1);  // decode: cur_seq tokens
            Tensor attn_out = Tensor({(size_t)attn_out_size}, Dtype::kF32);

            if (pos == 0) {
                float scale = 1.f / sqrtf((float)head_dim);
                scaled_dot_product(attn_out, q, k, v, nullptr, scale, false);
            } else {
                int cur_seq = pos + 1;
                float scale = 1.f / sqrtf((float)head_dim);

                // GQA: expand from n_kv_heads to n_heads
                int n_rep = n_heads / n_kv_heads;
                Tensor k_exp = Tensor({(size_t)n_heads, (size_t)cur_seq, (size_t)head_dim}, Dtype::kF32);
                Tensor v_exp = Tensor({(size_t)n_heads, (size_t)cur_seq, (size_t)head_dim}, Dtype::kF32);
                Tensor q_exp = Tensor({(size_t)n_heads, (size_t)cur_seq, (size_t)head_dim}, Dtype::kF32);

                for (int g = 0; g < n_kv_heads; g++) {
                    for (int r = 0; r < n_rep; r++) {
                        int h_dst = g * n_rep + r;
                        for (int s = 0; s < cur_seq; s++) {
                            // Correct cache indexing: cache[s * kv_row + g * head_dim + d]
                            std::memcpy(k_exp.f32() + h_dst * cur_seq * head_dim + s * head_dim,
                                        state_.k_cache[l].f32() + s * kv_row + g * head_dim,
                                        head_dim * sizeof(float));
                            std::memcpy(v_exp.f32() + h_dst * cur_seq * head_dim + s * head_dim,
                                        state_.v_cache[l].f32() + s * kv_row + g * head_dim,
                                        head_dim * sizeof(float));
                        }
                    }
                }
                // Pad Q: repeat single token across positions
                for (int h = 0; h < n_heads; h++)
                    for (int s = 0; s < cur_seq; s++)
                        std::memcpy(q_exp.f32() + h * cur_seq * head_dim + s * head_dim,
                                    q.f32() + h * head_dim, head_dim * sizeof(float));

                scaled_dot_product(attn_out, q_exp, k_exp, v_exp, nullptr, scale, true);
            }

            // --- Output projection ---
            if (wo_i >= 0) {
                Tensor Wo = weight_fp32(wo_i);
                Tensor Mo = model_.get_mask(wo_i);
                Tensor attn_proj = Tensor({(size_t)hidden}, Dtype::kF32);
                if (Wo.data()) {
                    if (Mo.data())
                        sparse_linear(attn_proj, attn_out, Wo, Mo, Tensor());
                    else
                        dense_linear(attn_proj, attn_out, Wo, Tensor());
                    residual_add(state_.x, attn_proj, state_.x);
                }
            }

            // --- RMSNorm before MLP ---
            Tensor rms_w2 = pn_i >= 0 ? weight_fp32(pn_i) : Tensor();
            Tensor x_mlp = Tensor({(size_t)hidden}, Dtype::kF32);
            rms_norm(x_mlp, state_.x, rms_w2, cfg.norm_eps);

            // --- MLP: Gate/Up → SiLU → Down ---
            if (wg_i >= 0 && wu_i >= 0 && wd_i >= 0) {
                Tensor Wg = weight_fp32(wg_i);
                Tensor Wu = weight_fp32(wu_i);
                Tensor Wd = weight_fp32(wd_i);
                Tensor Mg = model_.get_mask(wg_i);
                Tensor Mu = model_.get_mask(wu_i);
                Tensor Md = model_.get_mask(wd_i);
                if (Wg.data() && Wu.data()) {
                    Tensor gate = Tensor({(size_t)cfg.intermediate_size}, Dtype::kF32);
                    Tensor up   = Tensor({(size_t)cfg.intermediate_size}, Dtype::kF32);
                    if (Mg.data()) {
                        sparse_linear(gate, x_mlp, Wg, Mg, Tensor());
                    } else {
                        dense_linear(gate, x_mlp, Wg, Tensor());
                    }
                    if (Mu.data()) {
                        sparse_linear(up, x_mlp, Wu, Mu, Tensor());
                    } else {
                        dense_linear(up, x_mlp, Wu, Tensor());
                    }
                    silu(gate, gate);
                    for (size_t i = 0; i < (size_t)cfg.intermediate_size; i++)
                        gate.f32()[i] *= up.f32()[i];
                    if (Wd.data()) {
                        Tensor mlp_out = Tensor({(size_t)hidden}, Dtype::kF32);
                        if (Md.data()) {
                            sparse_linear(mlp_out, gate, Wd, Md, Tensor());
                        } else {
                            dense_linear(mlp_out, gate, Wd, Tensor());
                        }
                        residual_add(state_.x, mlp_out, state_.x);
                    }
                }
            }
        }

        // --- Final RMSNorm ---
        int fn_i = -1;
        for (auto& name : {"model.norm.weight", "transformer.ln_f.weight", "transformer.norm.weight"}) {
            fn_i = model_.find_layer(name);
            if (fn_i >= 0) break;
        }
        Tensor fn_w = fn_i >= 0 ? weight_fp32(fn_i) : Tensor();
        rms_norm(state_.x, state_.x, fn_w, cfg.norm_eps);

        // --- LM head ---
        int lm_i = -1;
        for (auto& name : {"lm_head.weight", "model.embed_tokens.weight", "transformer.wte.weight"}) {
            lm_i = model_.find_layer(name);
            if (lm_i >= 0) break;
        }
        if (lm_i >= 0) {
            Tensor Wlm = weight_fp32(lm_i);
            if (Wlm.data()) {
                Tensor logits = Tensor({(size_t)cfg.vocab_size}, Dtype::kF32);
                dense_linear(logits, state_.x, Wlm, Tensor());
                std::memcpy(state_.x.f32(), logits.f32(), cfg.vocab_size * sizeof(float));
            }
        }

        state_.seq_pos++;
        return state_.x.f32();
    }

private:
    /* Return FP32 weights for a tensor, dequantizing INT4/INT8 on the fly.
       The temp FP32 buffer lives only for the duration of the current op
       (RAII frees it when the Tensor goes out of scope). */
    Tensor weight_fp32(int idx) const {
        Tensor w = weight_fp32(idx);
        if (!w.data()) return w;
        Dtype wdt = w.dtype();
        if (wdt != Dtype::kINT4 && wdt != Dtype::kINT8) return w;

        Tensor sc = model_.get_scales(idx);
        if (!sc.data()) return w;

        size_t n = w.shape()[0] * w.shape()[1];
        Tensor f32w({w.shape()[0], w.shape()[1]}, Dtype::kF32);
        float* dst = f32w.f32();
        const float* scl = sc.f32();
        uint32_t gs = model_.layer_info(idx).group_size;
        if (gs == 0) gs = (uint32_t)n;
        size_t nsc = sc.shape()[0];

        if (wdt == Dtype::kINT4) {
            const uint8_t* src = (const uint8_t*)w.data();
            for (size_t i = 0; i < n; ++i) {
                size_t g = i / gs;
                float scale = (g < nsc) ? scl[g] : 1.0f;
                size_t pi = i / 2;
                uint8_t b = src[pi];
                int nib = (i & 1) ? (b >> 4) : (b & 0xF);
                dst[i] = (nib & 8) ? (float)(nib - 16) * scale : (float)nib * scale;
            }
        } else {
            const int8_t* src = (const int8_t*)w.data();
            for (size_t i = 0; i < n; ++i) {
                size_t g = i / gs;
                float scale = (g < nsc) ? scl[g] : 1.0f;
                dst[i] = (float)src[i] * scale;
            }
        }
        return f32w;
    }
    std::string fmt_layer_name(int layer, const char* component) {
        return "model.layers." + std::to_string(layer) + "." + component;
    }

    const Model& model_;
    RunState     state_;
};

}  // namespace run
}  // namespace tensorbit
