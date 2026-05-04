#ifndef TENSORBIT_RUN_BACKEND_HPP
#define TENSORBIT_RUN_BACKEND_HPP

#include <functional>
#include <string>
#include <vector>

#include "tensorbit/run/backend.h"
#include "tensorbit/run/common.hpp"
#include "tensorbit/run/tensor.hpp"

namespace tensorbit {
namespace run {

/* ================================================================
 * C++ backend registry — wraps the C backend system
 * ================================================================ */

class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry reg;
        return reg;
    }

    /* Initialize all available backends */
    void init() {
        if (initialized_) return;
        initialized_ = true;

        tb_backend_cpu_init();
#ifdef TB_HAS_CUDA
        tb_backend_cuda_init();
#endif
    }

    /* Check if a backend is available */
    bool has_backend(const std::string& name) const {
        int n = tb_backend_count();
        for (int i = 0; i < n; i++) {
            const TbBackend* be = tb_backend_get(i);
            if (be && be->name == name) return true;
        }
        return false;
    }

    /* Get count of registered backends */
    int count() const { return tb_backend_count(); }

private:
    BackendRegistry() = default;
    bool initialized_ = false;
};

/* ================================================================
 * High-level op functions (C++ convenience wrappers)
 * These dispatch through the C backend system.
 * ================================================================ */

inline int sparse_linear(Tensor& y, const Tensor& x, const Tensor& w, const Tensor& mask,
                          const Tensor& bias) {
    return tb_dispatch_sparse_linear(y.c_tensor(), x.c_tensor(), w.c_tensor(), mask.c_tensor(),
                                     bias.data() ? bias.c_tensor() : nullptr);
}

inline int dense_linear(Tensor& y, const Tensor& x, const Tensor& w, const Tensor& bias) {
    return tb_dispatch_dense_linear(y.c_tensor(), x.c_tensor(), w.c_tensor(),
                                     bias.data() ? bias.c_tensor() : nullptr);
}

inline int rms_norm(Tensor& y, const Tensor& x, const Tensor& weight, float eps) {
    return tb_dispatch_rms_norm(y.c_tensor(), x.c_tensor(),
                                 weight.data() ? weight.c_tensor() : nullptr, eps);
}

inline int layer_norm(Tensor& y, const Tensor& x, const Tensor& weight, const Tensor& bias,
                       float eps) {
    return tb_dispatch_layer_norm(y.c_tensor(), x.c_tensor(),
                                   weight.data() ? weight.c_tensor() : nullptr,
                                   bias.data() ? bias.c_tensor() : nullptr, eps);
}

inline int gelu(Tensor& y, const Tensor& x) { return tb_dispatch_gelu(y.c_tensor(), x.c_tensor()); }

inline int silu(Tensor& y, const Tensor& x) { return tb_dispatch_silu(y.c_tensor(), x.c_tensor()); }

inline int softmax(Tensor& y, const Tensor& x, const Tensor* mask = nullptr) {
    return tb_dispatch_softmax(y.c_tensor(), x.c_tensor(), mask ? mask->c_tensor() : nullptr);
}

inline int attention_qkv(Tensor& q, Tensor& k, Tensor& v, const Tensor& x, const Tensor& wq,
                          const Tensor& wk, const Tensor& wv, const Tensor& mq, const Tensor& mk,
                          const Tensor& mv, int num_heads, int num_kv_heads, int head_dim) {
    return tb_dispatch_attention_qkv(q.c_tensor(), k.c_tensor(), v.c_tensor(), x.c_tensor(),
                                      wq.c_tensor(), wk.c_tensor(), wv.c_tensor(), mq.c_tensor(),
                                      mk.c_tensor(), mv.c_tensor(), num_heads, num_kv_heads,
                                      head_dim);
}

inline int scaled_dot_product(Tensor& y, const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor* mask, float scale, bool causal) {
    return tb_dispatch_scaled_dot_product(y.c_tensor(), q.c_tensor(), k.c_tensor(), v.c_tensor(),
                                           mask ? mask->c_tensor() : nullptr, scale, causal);
}

inline int rope(Tensor& q, Tensor& k, int position, float theta, int head_dim) {
    return tb_dispatch_rope(q.c_tensor(), k.c_tensor(), position, theta, head_dim);
}

inline int residual_add(Tensor& y, const Tensor& x, const Tensor& residual) {
    return tb_dispatch_residual_add(y.c_tensor(), x.c_tensor(), residual.c_tensor());
}

inline int embedding(Tensor& y, const int* token_ids, int n_tokens, const Tensor& table) {
    return tb_dispatch_embedding(y.c_tensor(), token_ids, n_tokens, table.c_tensor());
}

inline int conv2d(Tensor& y, const Tensor& x, const Tensor& kernel, const Tensor& bias, int stride,
                   int padding) {
    return tb_dispatch_conv2d(y.c_tensor(), x.c_tensor(), kernel.c_tensor(),
                               bias.data() ? bias.c_tensor() : nullptr, stride, padding);
}

}  // namespace run
}  // namespace tensorbit

#endif /* TENSORBIT_RUN_BACKEND_HPP */
