#ifndef TENSORBIT_RUN_OPS_H
#define TENSORBIT_RUN_OPS_H

#include "tensorbit/run/common.h"
#include "tensorbit/run/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Sparse N:M Linear
 * y = x * W^T + bias
 * W is stored densely with pruned weights zeroed out.
 * mask indicates which weights are kept (N:M sparsity).
 *
 * x: [in_features]
 * W: [out_features, in_features]
 * mask: [out_features, in_features / M]  (packed bytes, N bits per byte set)
 * y: [out_features]
 * ================================================================ */

typedef struct {
    int nm_n;
    int nm_m;
} TbLinearParams;

int tb_cpu_sparse_linear_f32(size_t out_features, size_t in_features,
                              const float* TB_RESTRICT x,
                              const float* TB_RESTRICT w,
                              const uint8_t* TB_RESTRICT mask,
                              const float* TB_RESTRICT bias,
                              float* TB_RESTRICT y,
                              int nm_n, int nm_m);

/* ================================================================
 * Dense Linear
 * y = x * W^T + bias
 * ================================================================ */

int tb_cpu_dense_linear_f32(size_t out_features, size_t in_features,
                             const float* TB_RESTRICT x,
                             const float* TB_RESTRICT w,
                             const float* TB_RESTRICT bias,
                             float* TB_RESTRICT y);

/* ================================================================
 * RMS Normalization
 * y = x / rms(x) * weight
 * rms(x) = sqrt(mean(x^2) + eps)
 * ================================================================ */

int tb_cpu_rms_norm_f32(size_t n, const float* TB_RESTRICT x,
                         const float* TB_RESTRICT weight,
                         float* TB_RESTRICT y, float eps);

/* ================================================================
 * Layer Normalization
 * y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
 * ================================================================ */

int tb_cpu_layer_norm_f32(size_t n, const float* TB_RESTRICT x,
                           const float* TB_RESTRICT weight,
                           const float* TB_RESTRICT bias,
                           float* TB_RESTRICT y, float eps);

/* ================================================================
 * GELU activation (tanh approximation)
 * y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * ================================================================ */

int tb_cpu_gelu_f32(size_t n, const float* TB_RESTRICT x, float* TB_RESTRICT y);

/* ================================================================
 * SiLU activation
 * y = x * sigmoid(x) = x / (1 + exp(-x))
 * ================================================================ */

int tb_cpu_silu_f32(size_t n, const float* TB_RESTRICT x, float* TB_RESTRICT y);

/* ================================================================
 * Softmax (numerically stable)
 * y_i = exp(x_i - x_max) / sum_j exp(x_j - x_max)
 * Optional causal mask: positions j > i are set to -inf before softmax.
 * mask can be NULL for no masking.
 * ================================================================ */

int tb_cpu_softmax_f32(size_t n, const float* TB_RESTRICT x,
                        const float* TB_RESTRICT mask,
                        float* TB_RESTRICT y);

/* ================================================================
 * Scaled Dot-Product Attention
 * scores = q * k^T / sqrt(head_dim)
 * if mask: scores += mask
 * if causal: scores[i][j] where j > i are set to -inf
 * attn = softmax(scores)
 * y = attn * v
 *
 * q, k, v: [n_heads, seq_len, head_dim]  (concatenated)
 * y:       [n_heads, seq_len, head_dim]
 * ================================================================ */

int tb_cpu_scaled_dot_product_f32(int n_heads, int seq_len, int head_dim,
                                   const float* TB_RESTRICT q,
                                   const float* TB_RESTRICT k,
                                   const float* TB_RESTRICT v,
                                   const float* TB_RESTRICT mask,
                                   float* TB_RESTRICT y,
                                   bool causal, float scale);

/* ================================================================
 * Rotary Position Embedding (RoPE)
 * For each head, for each position m, for each pair (2i, 2i+1):
 *   theta_i = 1 / base^(2i/head_dim)
 *   cos_mi = cos(m * theta_i)
 *   sin_mi = sin(m * theta_i)
 *   q[2i]'   = q[2i]*cos_mi   - q[2i+1]*sin_mi
 *   q[2i+1]' = q[2i+1]*cos_mi + q[2i]*sin_mi
 * (same for k)
 *
 * q, k: [n_heads, seq_len, head_dim]  concatenated
 * position: absolute position (or relative for grouped-query)
 * ================================================================ */

int tb_cpu_rope_f32(int n_heads, int seq_len, int head_dim,
                     float* TB_RESTRICT q,
                     float* TB_RESTRICT k,
                     int position, float theta_base);

/* GQA-aware RoPE: q and k may have different head counts */
int tb_cpu_rope_gqa_f32(int n_q_heads, int n_kv_heads, int head_dim,
                         float* TB_RESTRICT q,
                         float* TB_RESTRICT k,
                         int position, float theta_base);

/* ================================================================
 * Residual Add
 * y = x + residual (element-wise)
 * ================================================================ */

int tb_cpu_residual_add_f32(size_t n, const float* TB_RESTRICT x,
                              const float* TB_RESTRICT residual,
                              float* TB_RESTRICT y);

/* ================================================================
 * Embedding Lookup
 * y[i] = embedding_table[token_ids[i]]  (copies row)
 * ================================================================ */

int tb_cpu_embedding_f32(const int* TB_RESTRICT token_ids, int n_tokens,
                          int embedding_dim,
                          const float* TB_RESTRICT table,
                          float* TB_RESTRICT y);

/* ================================================================
 * Conv2D (basic, no groups, NCHW format)
 * y = conv2d(x, kernel, stride, padding) + bias
 * Performed as im2col + GEMM for simplicity.
 * ================================================================ */

int tb_cpu_conv2d_f32(int batch, int in_ch, int in_h, int in_w,
                       int out_ch, int kh, int kw,
                       int stride, int padding,
                       const float* TB_RESTRICT x,
                       const float* TB_RESTRICT kernel,
                       const float* TB_RESTRICT bias,
                       float* TB_RESTRICT y);

#ifdef __cplusplus
}
#endif

#endif /* TENSORBIT_RUN_OPS_H */
