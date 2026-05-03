#include "tensorbit/run/ops.h"
#include <math.h>

int tb_cpu_rope_f32(int n_heads, int seq_len, int head_dim, float* TB_RESTRICT q, float* TB_RESTRICT k,
                     int position, float theta_base) {
    if (!q || !k) return TB_ERR_INVALID_ARG;

    size_t head_stride = (size_t)seq_len * head_dim;

    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_stride;
        float* kh = k + h * head_stride;

        for (int s = 0; s < seq_len; s++) {
            int    pos = position + s;
            float* qs = qh + s * head_dim;
            float* ks = kh + s * head_dim;

            for (int i = 0; i < head_dim / 2; i++) {
                float theta = 1.0f / powf(theta_base, (2.0f * i) / (float)head_dim);
                float cos_val = cosf((float)pos * theta);
                float sin_val = sinf((float)pos * theta);

                int idx0 = 2 * i;
                int idx1 = 2 * i + 1;

                float q0 = qs[idx0];
                float q1 = qs[idx1];
                qs[idx0] = q0 * cos_val - q1 * sin_val;
                qs[idx1] = q1 * cos_val + q0 * sin_val;

                float k0 = ks[idx0];
                float k1 = ks[idx1];
                ks[idx0] = k0 * cos_val - k1 * sin_val;
                ks[idx1] = k1 * cos_val + k0 * sin_val;
            }
        }
    }

    return TB_OK;
}
