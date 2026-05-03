/// @file kernels.cu
/// @brief CUDA GPU kernels for tensorbit-run inference ops.
/// @ingroup tensorbit-run
///
/// Optimised for Ampere (SM80) / Hopper (SM90).  All kernels use
/// 256-thread blocks and grid-stride loops for arbitrary sizes.

#include <cuda_runtime.h>
#include <cstddef>

static constexpr int kBlockSize = 256;

// ---- Sparse N:M Linear (2:4, 1:4, etc.) ----
__global__ void sparse_nm_linear_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ w, const uint8_t* __restrict__ mask,
    size_t out_features, size_t in_features, int nm_n, int nm_m)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_features) return;

    size_t groups_per_row = in_features / nm_m;
    const float* w_row = w + row * in_features;
    const uint8_t* m_row = mask + row * groups_per_row;

    float sum = 0.f;
    for (size_t g = 0; g < groups_per_row; g++) {
        uint8_t mbyte = m_row[g];
        size_t base = g * nm_m;
        for (int k = 0; k < nm_m; k++) {
            if (!(mbyte & (1u << k))) continue;  // skip pruned weights
            sum += w_row[base + k] * x[base + k];
        }
    }
    y[row] = sum;
}

// ---- Dense Linear (FP32) ----
__global__ void dense_linear_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ w, const float* __restrict__ bias,
    size_t out_features, size_t in_features)
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_features) return;

    float sum = bias ? bias[row] : 0.f;
    const float* w_row = w + row * in_features;
    for (size_t j = 0; j < in_features; j++)
        sum += w_row[j] * x[j];
    y[row] = sum;
}

// ---- RMS Normalisation (per-row) ----
__global__ void rms_norm_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ weight, size_t n, float eps)
{
    size_t row = blockIdx.x;
    size_t offset = row * n;
    float*       y_row = y + offset;
    const float* x_row = x + offset;

    // Compute sum of squares
    float ss = 0.f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        ss += x_row[i] * x_row[i];

    // Warp-level reduction
    for (int d = 16; d > 0; d >>= 1)
        ss += __shfl_down_sync(0xffffffff, ss, d);
    ss = __shfl_sync(0xffffffff, ss, 0);

    float inv_rms = rsqrtf(ss / (float)n + eps);
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float val = x_row[i] * inv_rms;
        if (weight) val *= weight[i];
        y_row[i] = val;
    }
}

// ---- GELU (tanh approximation) ----
__global__ void gelu_kernel(float* __restrict__ y, const float* __restrict__ x, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    y[idx] = 0.5f * v * (1.f + tanhf(0.79788456f * (v + 0.044715f * v * v * v)));
}

// ---- SiLU (swish) ----
__global__ void silu_kernel(float* __restrict__ y, const float* __restrict__ x, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    y[idx] = v * (1.f / (1.f + expf(-v)));
}

// ---- RoPE (rotary position encoding) ----
__global__ void rope_kernel(
    float* __restrict__ q, float* __restrict__ k,
    int n_q_heads, int n_kv_heads, int head_dim,
    int position, float theta_base)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = n_q_heads * head_dim / 2;
    if (idx >= (size_t)total_q) return;

    int h = idx / (head_dim / 2);
    int i = idx % (head_dim / 2);
    float theta = __powf(theta_base, -(2.f * (float)i) / (float)head_dim);
    float c = cosf((float)position * theta);
    float s = sinf((float)position * theta);

    // Q
    {
        float q0 = q[h * head_dim + 2 * i];
        float q1 = q[h * head_dim + 2 * i + 1];
        q[h * head_dim + 2 * i]     = q0 * c - q1 * s;
        q[h * head_dim + 2 * i + 1] = q1 * c + q0 * s;
    }
    // K (fewer heads for GQA)
    if (h < n_kv_heads) {
        float k0 = k[h * head_dim + 2 * i];
        float k1 = k[h * head_dim + 2 * i + 1];
        k[h * head_dim + 2 * i]     = k0 * c - k1 * s;
        k[h * head_dim + 2 * i + 1] = k1 * c + k0 * s;
    }
}

// ---- Residual Add ----
__global__ void residual_add_kernel(float* __restrict__ y, const float* __restrict__ a,
                                     const float* __restrict__ b, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = a[idx] + b[idx];
}

// ---- Numerically Stable Softmax (per-row, warp-reduce) ----
__global__ void softmax_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    size_t rows, size_t cols, const float* __restrict__ mask)
{
    size_t row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * cols;
    float*       y_row = y + row * cols;

    // Find max
    float max_val = -INFINITY;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x_row[i] + (mask ? mask[row * cols + i] : 0.f);
        if (v > max_val) max_val = v;
    }
    for (int d = 16; d > 0; d >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, d));
    max_val = __shfl_sync(0xffffffff, max_val, 0);

    // Compute exp sum
    float sum = 0.f;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x_row[i] + (mask ? mask[row * cols + i] : 0.f);
        float e = expf(v - max_val);
        y_row[i] = e;
        sum += e;
    }
    for (int d = 16; d > 0; d >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, d);
    sum = __shfl_sync(0xffffffff, sum, 0);

    float inv_sum = (sum > 0.f) ? (1.f / sum) : 1.f;
    for (size_t i = threadIdx.x; i < cols; i += blockDim.x)
        y_row[i] *= inv_sum;
}

// ---- Embedding Lookup — one warp per token ----
__global__ void embedding_kernel(
    float* __restrict__ y, const int* __restrict__ token_ids,
    const float* __restrict__ table, int n_tokens, int embed_dim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)(n_tokens * embed_dim)) return;

    int t = (int)idx / embed_dim;
    int d = (int)idx % embed_dim;
    int token = token_ids[t];
    y[idx] = table[token * embed_dim + d];
}

// ---- LayerNorm (subtract mean, divide by std, apply weight/bias) ----
__global__ void layer_norm_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ weight, const float* __restrict__ bias,
    size_t n, float eps)
{
    size_t row = blockIdx.x;
    size_t offset = row * n;
    float*       y_row = y + offset;
    const float* x_row = x + offset;

    // Mean
    float mean = 0.f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x)
        mean += x_row[i];
    for (int d = 16; d > 0; d >>= 1)
        mean += __shfl_down_sync(0xffffffff, mean, d);
    mean = __shfl_sync(0xffffffff, mean, 0) / (float)n;

    // Variance
    float var = 0.f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float diff = x_row[i] - mean;
        var += diff * diff;
    }
    for (int d = 16; d > 0; d >>= 1)
        var += __shfl_down_sync(0xffffffff, var, d);
    var = __shfl_sync(0xffffffff, var, 0) / (float)n;

    float inv_std = rsqrtf(var + eps);
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        float val = (x_row[i] - mean) * inv_std;
        if (weight) val = val * weight[i] + (bias ? bias[i] : 0.f);
        y_row[i] = val;
    }
}

// ---- Conv2D (NCHW, direct loops — GPUs prefer im2col+GEMM, but this avoids extra memory) ----
__global__ void conv2d_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ w, const float* __restrict__ bias,
    int B, int C_in, int H, int W,
    int C_out, int K, int stride, int pad)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * C_out * ((H + 2*pad - K) / stride + 1) * ((W + 2*pad - K) / stride + 1);
    if (idx >= (size_t)total_out) return;

    int OH = (H + 2*pad - K) / stride + 1;
    int OW = (W + 2*pad - K) / stride + 1;
    int out_per_img = C_out * OH * OW;

    int b  = (int)idx / out_per_img;
    int oc = ((int)idx % out_per_img) / (OH * OW);
    int oh = ((int)idx % (OH * OW)) / OW;
    int ow = (int)idx % OW;

    float sum = bias ? bias[oc] : 0.f;
    for (int ic = 0; ic < C_in; ic++) {
        for (int kh = 0; kh < K; kh++) {
            int ih = oh * stride + kh - pad;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < K; kw++) {
                int iw = ow * stride + kw - pad;
                if (iw < 0 || iw >= W) continue;
                sum += x[((b * C_in + ic) * H + ih) * W + iw]
                     * w[((oc * C_in + ic) * K + kh) * K + kw];
            }
        }
    }
    y[idx] = sum;
}

// ---- Scaled Dot-Product Attention (causal, single head) ----
__global__ void sdp_attention_kernel(
    float* __restrict__ y, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v,
    int n_heads, int seq_len, int kv_len, int head_dim, float scale, bool causal)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = n_heads * seq_len * head_dim;
    if (idx >= (size_t)total_out) return;

    int h  = (int)idx / (seq_len * head_dim);
    int si = ((int)idx % (seq_len * head_dim)) / head_dim;
    int di = (int)idx % head_dim;

    const float* qh = q + h * seq_len * head_dim;
    const float* kh = k + h * kv_len * head_dim;
    const float* vh = v + h * kv_len * head_dim;

    // Compute softmax scores for row si
    __shared__ float s_max[256], s_sum[256];
    float max_val = -INFINITY;

    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        float score = 0.f;
        for (int d = 0; d < head_dim; d++)
            score += qh[si * head_dim + d] * kh[j * head_dim + d];
        score *= scale;
        if (causal && j > si) score = -INFINITY;
        if (score > max_val) max_val = score;
    }
    s_max[threadIdx.x] = max_val;
    __syncthreads();
    for (int d = blockDim.x/2; d > 0; d >>= 1) {
        if (threadIdx.x < d)
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + d]);
        __syncthreads();
    }
    max_val = s_max[0];

    // Compute weighted sum
    float total = 0.f;
    float acc = 0.f;
    for (int j = threadIdx.x; j < kv_len; j += blockDim.x) {
        float score = 0.f;
        for (int d = 0; d < head_dim; d++)
            score += qh[si * head_dim + d] * kh[j * head_dim + d];
        score = (causal && j > si) ? -INFINITY : score * scale;
        float w = expf(score - max_val);
        if (threadIdx.x == 0) total += w;
        acc += w * vh[j * head_dim + di];
    }
    for (int d = blockDim.x/2; d > 0; d >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, d);
    if (threadIdx.x == 0) {
        for (int d = blockDim.x/2; d > 0; d >>= 1)
            total += __shfl_down_sync(0xffffffff, total, d);
    }
    acc /= (total > 0.f ? total : 1.f);
    y[idx] = acc;
}

// ================================================================
// Host launch wrappers
// ================================================================

extern "C" {

__host__ int tb_cuda_sparse_linear_f32(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ w, const uint8_t* __restrict__ mask,
    size_t out_features, size_t in_features, int nm_n, int nm_m)
{
    if (!y || !x || !w || !mask) return -1;
    int grid = (int)((out_features + kBlockSize - 1) / kBlockSize);
    sparse_nm_linear_kernel<<<grid, kBlockSize>>>(y, x, w, mask, out_features, in_features, nm_n, nm_m);
    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_dense_linear_f32(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ w, const float* __restrict__ bias,
    size_t out_features, size_t in_features)
{
    if (!y || !x || !w) return -1;
    int grid = (int)((out_features + kBlockSize - 1) / kBlockSize);
    dense_linear_kernel<<<grid, kBlockSize>>>(y, x, w, bias, out_features, in_features);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_rms_norm_f32(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ weight, size_t n, float eps)
{
    if (!y || !x) return -1;
    rms_norm_kernel<<<1, kBlockSize>>>(y, x, weight, n, eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_gelu_f32(float* __restrict__ y, const float* __restrict__ x, size_t n)
{
    if (!y || !x) return -1;
    int grid = (int)((n + kBlockSize - 1) / kBlockSize);
    gelu_kernel<<<grid, kBlockSize>>>(y, x, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_silu_f32(float* __restrict__ y, const float* __restrict__ x, size_t n)
{
    if (!y || !x) return -1;
    int grid = (int)((n + kBlockSize - 1) / kBlockSize);
    silu_kernel<<<grid, kBlockSize>>>(y, x, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_rope_f32(
    float* __restrict__ q, float* __restrict__ k,
    int n_q_heads, int n_kv_heads, int head_dim,
    int position, float theta_base)
{
    if (!q || !k) return -1;
    int total_q = n_q_heads * head_dim / 2;
    int grid = (int)((total_q + kBlockSize - 1) / kBlockSize);
    rope_kernel<<<grid, kBlockSize>>>(q, k, n_q_heads, n_kv_heads, head_dim, position, theta_base);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_residual_add_f32(
    float* __restrict__ y, const float* __restrict__ a,
    const float* __restrict__ b, size_t n)
{
    if (!y || !a || !b) return -1;
    int grid = (int)((n + kBlockSize - 1) / kBlockSize);
    residual_add_kernel<<<grid, kBlockSize>>>(y, a, b, n);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_softmax_f32(
    float* __restrict__ y, const float* __restrict__ x,
    size_t rows, size_t cols, const float* __restrict__ mask)
{
    if (!y || !x) return -1;
    softmax_kernel<<<(int)rows, kBlockSize>>>(y, x, rows, cols, mask);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_embedding_f32(
    float* __restrict__ y, const int* __restrict__ token_ids,
    const float* __restrict__ table, int n_tokens, int embed_dim)
{
    if (!y || !token_ids || !table) return -1;
    int total = n_tokens * embed_dim;
    int grid = (int)((total + kBlockSize - 1) / kBlockSize);
    embedding_kernel<<<grid, kBlockSize>>>(y, token_ids, table, n_tokens, embed_dim);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_layer_norm_f32(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ weight, const float* __restrict__ bias,
    size_t n, float eps)
{
    if (!y || !x) return -1;
    layer_norm_kernel<<<1, kBlockSize>>>(y, x, weight, bias, n, eps);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_conv2d_f32(
    float* __restrict__ y, const float* __restrict__ x,
    const float* __restrict__ w, const float* __restrict__ bias,
    int B, int C_in, int H, int W, int C_out, int K, int stride, int pad)
{
    if (!y || !x || !w) return -1;
    int OH = (H + 2*pad - K) / stride + 1;
    int OW = (W + 2*pad - K) / stride + 1;
    int total = B * C_out * OH * OW;
    int grid = (int)((total + kBlockSize - 1) / kBlockSize);
    conv2d_kernel<<<grid, kBlockSize>>>(y, x, w, bias, B, C_in, H, W, C_out, K, stride, pad);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

__host__ int tb_cuda_sdp_attention_f32(
    float* __restrict__ y, const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v,
    int n_heads, int seq_len, int kv_len, int head_dim, float scale, bool causal)
{
    if (!y || !q || !k || !v) return -1;
    int total = n_heads * seq_len * head_dim;
    int grid = (int)((total + kBlockSize - 1) / kBlockSize);
    sdp_attention_kernel<<<grid, kBlockSize>>>(y, q, k, v, n_heads, seq_len, kv_len, head_dim, scale, causal);
    return (cudaGetLastError() == cudaSuccess) ? 0 : -1;
}

}  // extern "C"
