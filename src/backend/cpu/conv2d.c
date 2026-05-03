#include "tensorbit/run/backend.h"
#include "tensorbit/run/ops.h"

int tb_cpu_conv2d_dispatch(void* output, const void** inputs, int n_inputs, const void* params) {
    const TbConv2dParams* cp = (const TbConv2dParams*)params;
    int stride = cp ? cp->stride : 1;
    int padding = cp ? cp->padding : 0;

    TbTensor*       y = (TbTensor*)output;
    const TbTensor* x = (const TbTensor*)inputs[0];
    const TbTensor* kernel = (const TbTensor*)inputs[1];
    const TbTensor* bias = (n_inputs >= 3) ? (const TbTensor*)inputs[2] : NULL;

    int batch = (int)x->shape[0];
    int in_ch = (int)x->shape[1];
    int in_h = (int)x->shape[2];
    int in_w = (int)x->shape[3];
    int out_ch = (int)kernel->shape[0];
    int kh = (int)kernel->shape[2];
    int kw = (int)kernel->shape[3];

    return tb_cpu_conv2d_f32(batch, in_ch, in_h, in_w, out_ch, kh, kw, stride, padding,
                              tb_tensor_cf32(x), tb_tensor_cf32(kernel),
                              bias ? tb_tensor_cf32(bias) : NULL, tb_tensor_f32(y));
}
