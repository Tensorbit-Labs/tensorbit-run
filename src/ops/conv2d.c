#include "tensorbit/run/ops.h"
#include <string.h>

int tb_cpu_conv2d_f32(int batch, int in_ch, int in_h, int in_w, int out_ch, int kh, int kw,
                       int stride, int padding, const float* TB_RESTRICT x,
                       const float* TB_RESTRICT kernel, const float* TB_RESTRICT bias,
                       float* TB_RESTRICT y) {
    if (!x || !kernel || !y) return TB_ERR_INVALID_ARG;

    int out_h = (in_h + 2 * padding - kh) / stride + 1;
    int out_w = (in_w + 2 * padding - kw) / stride + 1;

    memset(y, 0, sizeof(float) * batch * out_ch * out_h * out_w);

    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float acc = bias ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < in_ch; ic++) {
                        for (int r = 0; r < kh; r++) {
                            for (int c = 0; c < kw; c++) {
                                int ih = oh * stride + r - padding;
                                int iw = ow * stride + c - padding;
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    size_t x_idx = ((b * in_ch + ic) * in_h + ih) * in_w + iw;
                                    size_t k_idx = ((oc * in_ch + ic) * kh + r) * kw + c;
                                    acc += x[x_idx] * kernel[k_idx];
                                }
                            }
                        }
                    }

                    size_t y_idx = ((b * out_ch + oc) * out_h + oh) * out_w + ow;
                    y[y_idx] = acc;
                }
            }
        }
    }

    return TB_OK;
}
