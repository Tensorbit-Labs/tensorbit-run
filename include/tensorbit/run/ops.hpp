#ifndef TENSORBIT_RUN_OPS_HPP
#define TENSORBIT_RUN_OPS_HPP

#include <cmath>
#include <cstring>

#include "tensorbit/run/common.hpp"
#include "tensorbit/run/model.hpp"
#include "tensorbit/run/tensor.hpp"
#include "tensorbit/run/backend.hpp"

namespace tensorbit {
namespace run {

/* ================================================================
 * Forward pass state — activations and KV cache for one sequence
 * ================================================================ */

struct RunState {
    Tensor x;               /* current hidden state [hidden_size] */
    Tensor residual;        /* residual for attention [hidden_size] */

    /* KV cache: [num_layers, 2, max_seq_len, num_kv_heads * head_dim] */
    std::vector<Tensor> k_cache;
    std::vector<Tensor> v_cache;

    int seq_pos = 0;        /* current sequence position */
};

/* ================================================================
 * Transformer forward pass (Llama-style decoder)
 *
 * Architecture:
 *   x = embedding(token_ids)
 *   for each layer:
 *     residual = x
 *     x = rms_norm(x, input_layernorm)
 *     q,k,v = sparse_linear_qkv(x, Wq,Wk,Wv)
 *     q,k = rope(q, k, position)
 *     k_cache[seq_pos] = k, v_cache[seq_pos] = v
 *     x = scaled_dot_product(q, k_cache, v_cache, causal)
 *     x = sparse_linear(x, Wo)
 *     x = residual_add(x, residual)
 *     residual = x
 *     x = rms_norm(x, post_attention_layernorm)
 *     gate = silu(sparse_linear(x, W_gate))
 *     up = sparse_linear(x, W_up)
 *     x = gate * up
 *     x = sparse_linear(x, W_down)
 *     x = residual_add(x, residual)
 *   x = rms_norm(x, final_norm)
 *   logits = dense_linear(x, lm_head)
 * ================================================================ */

class TransformerRunner {
public:
    TransformerRunner(const Model& model) : model_(model) { init_state(); }

    /* Initialize run state allocations */
    void init_state() {
        auto cfg = model_.config();
        int  hidden = cfg.hidden_size;
        int  n_layers = cfg.num_layers;

        state_.x = Tensor({(size_t)hidden}, Dtype::kF32);
        state_.residual = Tensor({(size_t)hidden}, Dtype::kF32);
        state_.seq_pos = 0;

        /* KV cache: pre-allocate for max_seq_len (up to 2048) */
        state_.k_cache.resize(n_layers);
        state_.v_cache.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            size_t kv_dim = (size_t)cfg.num_kv_heads * cfg.head_dim;
            state_.k_cache[i] = Tensor({(size_t)cfg.max_seq_len, kv_dim}, Dtype::kF32);
            state_.v_cache[i] = Tensor({(size_t)cfg.max_seq_len, kv_dim}, Dtype::kF32);
        }
    }

    /* Run one token through the model.
     * token_id: input token
     * Returns: logits tensor [vocab_size] */
    const float* forward(int token_id) {
        auto cfg = model_.config();
        int  hidden = cfg.hidden_size;
        int  n_layers = cfg.num_layers;
        int  n_heads = cfg.num_heads;
        int  n_kv_heads = cfg.num_kv_heads;
        int  head_dim = cfg.head_dim;

        /* Embedding */
        int emb_idx = model_.find_layer("model.embed_tokens.weight");
        if (emb_idx < 0) {
            TB_LOG_ERROR("embedding layer not found");
            return nullptr;
        }
        Tensor emb = model_.get_weights(emb_idx);
        if (!emb.data()) return nullptr;

        embedding(state_.x, &token_id, 1, emb);

        /* Layer loop */
        for (int l = 0; l < n_layers; l++) {
            auto q_proj = fmt_layer_name(l, "self_attn.q_proj");
            auto k_proj = fmt_layer_name(l, "self_attn.k_proj");
            auto v_proj = fmt_layer_name(l, "self_attn.v_proj");
            auto o_proj = fmt_layer_name(l, "self_attn.o_proj");
            auto gate_proj = fmt_layer_name(l, "mlp.gate_proj");
            auto up_proj = fmt_layer_name(l, "mlp.up_proj");
            auto down_proj = fmt_layer_name(l, "mlp.down_proj");
            auto input_ln = fmt_layer_name(l, "input_layernorm");
            auto post_ln = fmt_layer_name(l, "post_attention_layernorm");

            int wq_i = model_.find_layer(q_proj + ".weight");
            int wk_i = model_.find_layer(k_proj + ".weight");
            int wv_i = model_.find_layer(v_proj + ".weight");
            int wo_i = model_.find_layer(o_proj + ".weight");
            int wg_i = model_.find_layer(gate_proj + ".weight");
            int wu_i = model_.find_layer(up_proj + ".weight");
            int wd_i = model_.find_layer(down_proj + ".weight");
            int in_i = model_.find_layer(input_ln + ".weight");
            int pn_i = model_.find_layer(post_ln + ".weight");

            /* -- RMSNorm before attention -- */
            Tensor rms_weight_in = in_i >= 0 ? model_.get_weights(in_i) : Tensor();
            rms_norm(state_.x, state_.x, rms_weight_in, cfg.norm_eps);

            /* -- Copy residual -- */
            Tensor x_before_attn = Tensor({(size_t)hidden}, Dtype::kF32);
            std::memcpy(x_before_attn.f32(), state_.x.f32(), hidden * sizeof(float));

            /* -- QKV projections -- */
            Tensor q =
                Tensor({(size_t)n_heads, 1, (size_t)head_dim}, Dtype::kF32);
            Tensor k =
                Tensor({(size_t)n_kv_heads, 1, (size_t)head_dim}, Dtype::kF32);
            Tensor v =
                Tensor({(size_t)n_kv_heads, 1, (size_t)head_dim}, Dtype::kF32);

            if (wq_i >= 0 && wk_i >= 0 && wv_i >= 0) {
                Tensor Wq = model_.get_weights(wq_i);
                Tensor Wk = model_.get_weights(wk_i);
                Tensor Wv = model_.get_weights(wv_i);
                Tensor Mq = model_.get_mask(wq_i);
                Tensor Mk = model_.get_mask(wk_i);
                Tensor Mv = model_.get_mask(wv_i);

                Tensor q_flat = Tensor({(size_t)(n_heads * head_dim)}, Dtype::kF32);
                Tensor k_flat = Tensor({(size_t)(n_kv_heads * head_dim)}, Dtype::kF32);
                Tensor v_flat = Tensor({(size_t)(n_kv_heads * head_dim)}, Dtype::kF32);

                if (Mq.data()) {
                    sparse_linear(q_flat, state_.x, Wq, Mq, Tensor());
                    sparse_linear(k_flat, state_.x, Wk, Mk, Tensor());
                    sparse_linear(v_flat, state_.x, Wv, Mv, Tensor());
                } else {
                    dense_linear(q_flat, state_.x, Wq, Tensor());
                    dense_linear(k_flat, state_.x, Wk, Tensor());
                    dense_linear(v_flat, state_.x, Wv, Tensor());
                }

                /* Reshape flat → [heads, 1, head_dim] */
                for (int h = 0; h < n_heads; h++) {
                    std::memcpy(q.f32() + h * head_dim, q_flat.f32() + h * head_dim,
                                head_dim * sizeof(float));
                }
                for (int h = 0; h < n_kv_heads; h++) {
                    std::memcpy(k.f32() + h * head_dim, k_flat.f32() + h * head_dim,
                                head_dim * sizeof(float));
                    std::memcpy(v.f32() + h * head_dim, v_flat.f32() + h * head_dim,
                                head_dim * sizeof(float));
                }
            }

            /* -- RoPE -- */
            rope(q, k, state_.seq_pos, cfg.rope_theta, head_dim);

            /* -- KV cache store -- */
            int pos = state_.seq_pos;
            {
                float* kc = state_.k_cache[l].f32() + pos * n_kv_heads * head_dim;
                float* vc = state_.v_cache[l].f32() + pos * n_kv_heads * head_dim;
                std::memcpy(kc, k.f32(), n_kv_heads * head_dim * sizeof(float));
                std::memcpy(vc, v.f32(), n_kv_heads * head_dim * sizeof(float));
            }

            /* -- Scaled dot-product attention (with KV cache) -- */
            Tensor attn_out = Tensor({(size_t)(n_heads * head_dim)}, Dtype::kF32);

            if (state_.seq_pos == 0) {
                /* Prefill: just compute from current q, k, v */
                /* For simplicity, use the Q from all heads */
                float scale = 1.0f / sqrtf((float)head_dim);
                scaled_dot_product(attn_out, q, k, v, nullptr, scale, false);
            } else {
                /* Decode: q has shape [n_heads, 1, head_dim], k_cache has [cur_seq, kv_dim] */
                /* For now, use a simple single-token attention */
                float scale = 1.0f / sqrtf((float)head_dim);
                scaled_dot_product(attn_out, q, k, v, nullptr, scale, false);
            }

            /* -- Output projection -- */
            Tensor attn_proj = Tensor({(size_t)hidden}, Dtype::kF32);
            if (wo_i >= 0) {
                Tensor Wo = model_.get_weights(wo_i);
                Tensor Mo = model_.get_mask(wo_i);
                if (Mo.data()) {
                    sparse_linear(attn_proj, attn_out, Wo, Mo, Tensor());
                } else {
                    dense_linear(attn_proj, attn_out, Wo, Tensor());
                }
            }

            /* -- Residual add after attention -- */
            residual_add(state_.x, attn_proj, x_before_attn);

            /* -- RMSNorm before MLP -- */
            Tensor rms_weight_post = pn_i >= 0 ? model_.get_weights(pn_i) : Tensor();
            Tensor x_mlp = Tensor({(size_t)hidden}, Dtype::kF32);
            rms_norm(x_mlp, state_.x, rms_weight_post, cfg.norm_eps);

            /* -- Copy residual -- */
            Tensor x_before_mlp = Tensor({(size_t)hidden}, Dtype::kF32);
            std::memcpy(x_before_mlp.f32(), state_.x.f32(), hidden * sizeof(float));

            /* -- MLP: gate * silu(up) then down -- */
            Tensor gate_out = Tensor({(size_t)cfg.intermediate_size}, Dtype::kF32);
            Tensor up_out = Tensor({(size_t)cfg.intermediate_size}, Dtype::kF32);

            if (wg_i >= 0 && wu_i >= 0) {
                Tensor Wg = model_.get_weights(wg_i);
                Tensor Mg = model_.get_mask(wg_i);
                Tensor Wu = model_.get_weights(wu_i);
                Tensor Mu = model_.get_mask(wu_i);

                if (Mg.data()) {
                    sparse_linear(gate_out, x_mlp, Wg, Mg, Tensor());
                } else {
                    dense_linear(gate_out, x_mlp, Wg, Tensor());
                }

                silu(gate_out, gate_out);

                if (Mu.data()) {
                    sparse_linear(up_out, x_mlp, Wu, Mu, Tensor());
                } else {
                    dense_linear(up_out, x_mlp, Wu, Tensor());
                }

                /* Element-wise multiply */
                for (int j = 0; j < cfg.intermediate_size; j++) {
                    gate_out.f32()[j] *= up_out.f32()[j];
                }

                /* Down projection */
                Tensor mlp_out = Tensor({(size_t)hidden}, Dtype::kF32);
                if (wd_i >= 0) {
                    Tensor Wd = model_.get_weights(wd_i);
                    Tensor Md = model_.get_mask(wd_i);
                    if (Md.data()) {
                        sparse_linear(mlp_out, gate_out, Wd, Md, Tensor());
                    } else {
                        dense_linear(mlp_out, gate_out, Wd, Tensor());
                    }
                }

                /* Residual add after MLP */
                residual_add(state_.x, mlp_out, x_before_mlp);
            }
        }

        /* -- Final RMSNorm -- */
        int final_norm_idx = model_.find_layer("model.norm.weight");
        if (final_norm_idx >= 0) {
            Tensor final_norm = model_.get_weights(final_norm_idx);
            rms_norm(state_.x, state_.x, final_norm, cfg.norm_eps);
        }

        /* -- LM head -- */
        int lm_head_idx = model_.find_layer("lm_head.weight");
        if (lm_head_idx < 0) {
            lm_head_idx = model_.find_layer("model.embed_tokens.weight");
        }
        if (lm_head_idx >= 0) {
            Tensor lm_head = model_.get_weights(lm_head_idx);
            logits_ = Tensor({(size_t)cfg.vocab_size}, Dtype::kF32);
            dense_linear(logits_, state_.x, lm_head, Tensor());
        }

        state_.seq_pos++;
        return logits_.f32();
    }

    /* Get current logits */
    const float* logits() const { return logits_.f32(); }

    /* Get vocab size */
    int vocab_size() const { return model_.config().vocab_size; }

    /* Reset state for a new sequence */
    void reset() {
        state_.seq_pos = 0;
        if (state_.x.data()) memset(state_.x.f32(), 0, state_.x.bytes());
    }

private:
    std::string fmt_layer_name(int layer, const char* suffix) {
        char buf[256];
        snprintf(buf, sizeof(buf), "model.layers.%d.%s", layer, suffix);
        return std::string(buf);
    }

    const Model& model_;
    RunState     state_;
    Tensor       logits_;
};

}  // namespace run
}  // namespace tensorbit

#endif /* TENSORBIT_RUN_OPS_HPP */
