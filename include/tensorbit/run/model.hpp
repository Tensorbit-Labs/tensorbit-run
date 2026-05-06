#ifndef TENSORBIT_RUN_MODEL_HPP
#define TENSORBIT_RUN_MODEL_HPP

#include <map>
#include <string>
#include <vector>

#include "tensorbit/run/common.hpp"
#include "tensorbit/run/model.h"
#include "tensorbit/run/tensor.hpp"

namespace tensorbit {
namespace run {

/* ================================================================
 * Error codes for C++ layer
 * ================================================================ */

enum class ModelError {
    kOk = 0,
    kFileOpen,
    kFileRead,
    kBadMagic,
    kVersion,
    kOom,
    kInvalidArg,
    kLayerNotFound,
    kJsonParse,
    kBackendNotFound,
};

/* ================================================================
 * ModelConfig
 * ================================================================ */

struct ModelConfig {
    std::string architecture;
    int         hidden_size = 4096;
    int         num_heads = 32;
    int         num_kv_heads = 8;
    int         intermediate_size = 14336;
    int         vocab_size = 32000;
    int         max_seq_len = 2048;
    int         num_layers = 0;
    float       norm_eps = 1e-5f;
    float       rope_theta = 10000.0f;
    int         head_dim = 128;
};

/* ================================================================
 * LayerInfo — metadata about one layer
 * ================================================================ */

struct LayerInfo {
    std::string name;
    size_t      shape[2] = {0, 0};
    int         nm_n = 0;
    int         nm_m = 0;
    Dtype       dtype = Dtype::kF32;
    int         scale_count = 0;
    int         group_size = 0;
};

/* ================================================================
 * Model — C++ RAII wrapper for model loading and layer access
 * Zero-copy: weights and masks are views into mmap'd file data.
 * ================================================================ */

class Model {
public:
    Model() = default;

    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Model(Model&& other) noexcept : model_(other.model_) {
        other.model_ = {};
    }

    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            close();
            model_ = other.model_;
            other.model_ = {};
        }
        return *this;
    }

    ~Model() { close(); }

    /* Load from .tbm file */
    static Result<Model, ModelError> load_tbm(const std::string& path) {
        Model m;
        int   ret = tb_model_load_tbm(&m.model_, path.c_str());
        if (ret != TB_OK) {
            auto ec = static_cast<ModelError>(ret);
            return unexpected(ec);
        }
        return m;
    }

    /* Load from directory with manifest.json */
    static Result<Model, ModelError> load_dir(const std::string& dir_path) {
        Model m;
        int   ret = tb_model_load_dir(&m.model_, dir_path.c_str());
        if (ret != TB_OK) {
            auto ec = static_cast<ModelError>(ret);
            return unexpected(ec);
        }
        return m;
    }

    /* Number of layers (tensors) */
    int num_layers() const { return model_.num_layers; }

    /* Model config */
    ModelConfig config() const {
        ModelConfig cfg;
        cfg.architecture = model_.config.architecture;
        cfg.hidden_size = model_.config.hidden_size;
        cfg.num_heads = model_.config.num_heads;
        cfg.num_kv_heads = model_.config.num_kv_heads;
        cfg.intermediate_size = model_.config.intermediate_size;
        cfg.vocab_size = model_.config.vocab_size;
        cfg.max_seq_len = model_.config.max_seq_len;
        cfg.num_layers = model_.config.num_layers;
        cfg.norm_eps = model_.config.norm_eps;
        cfg.rope_theta = model_.config.rope_theta;
        cfg.head_dim = model_.config.head_dim;
        return cfg;
    }

    /* Find layer index by name. Returns -1 if not found. */
    int find_layer(const std::string& name) const { return tb_model_find_layer(&model_, name.c_str()); }

    /* Get layer info */
    LayerInfo layer_info(int idx) const {
        LayerInfo info;
        if (idx >= 0 && idx < model_.num_layers) {
            const TbLayerDesc& l = model_.layers[idx];
            info.name = l.name;
            info.shape[0] = l.shape[0];
            info.shape[1] = l.shape[1];
            info.nm_n = (int)l.nm_n;
            info.nm_m = (int)l.nm_m;
            info.dtype = static_cast<Dtype>(l.dtype);
            info.scale_count = (int)l.scale_count;
            info.group_size  = (int)l.group_size;
        }
        return info;
    }

    /* Get weight tensor (zero-copy view). Returns empty tensor on error. */
    Tensor get_weights(int idx) const {
        TbTensor t;
        int      ret = tb_model_get_weight_tensor(&model_, idx, &t);
        if (ret != TB_OK) return Tensor();
        return Tensor::wrap(t.data, {t.shape[0], t.shape[1]},
                             static_cast<Dtype>(t.dtype), DeviceLocation::kHost);
    }

    /* Get mask tensor (zero-copy view). Returns empty tensor on error. */
    Tensor get_mask(int idx) const {
        TbTensor t;
        int      ret = tb_model_get_mask_tensor(&model_, idx, &t);
        if (ret != TB_OK) return Tensor();
        return Tensor::wrap(t.data, {t.shape[0]}, Dtype::kU8, DeviceLocation::kHost);
    }

    /* Get scale tensor (zero-copy view). Returns empty tensor on error. */
    Tensor get_scales(int idx) const {
        TbTensor t;
        int      ret = tb_model_get_scale_tensor(&model_, idx, &t);
        if (ret != TB_OK) return Tensor();
        return Tensor::wrap(t.data, {t.shape[0]}, Dtype::kF32, DeviceLocation::kHost);
    }

    /* Get weight tensor, dequantizing INT4/INT8 to FP32 automatically.
       FP32 weights are returned zero-copy.  INT4/INT8 allocate a temp
       FP32 buffer that is freed when the tensor is destroyed. */
    Tensor get_weight_fp32(int idx) const {
        TbTensor t;
        int      ret = tb_model_get_weight_tensor(&model_, idx, &t);
        if (ret != TB_OK) return Tensor();

        TbDtype wdt = t.dtype;
        if (wdt != TB_DTYPE_INT4 && wdt != TB_DTYPE_INT8) {
            return Tensor::wrap(t.data, {t.shape[0], t.shape[1]},
                                 static_cast<Dtype>(wdt), DeviceLocation::kHost);
        }

        /* ---- INT4 / INT8 dequant ---- */
        TbTensor sc;
        ret = tb_model_get_scale_tensor(&model_, idx, &sc);
        if (ret != TB_OK) return Tensor();

        const TbLayerDesc& layer = model_.layers[idx];
        size_t n = layer.num_weights;
        uint32_t gs = layer.group_size > 0 ? layer.group_size : (uint32_t)n;

        Tensor f32w({layer.shape[0], layer.shape[1]}, Dtype::kF32);
        float* dst = f32w.f32();
        const float* scl = (const float*)sc.data;

        if (wdt == TB_DTYPE_INT4) {
            const uint8_t* src = (const uint8_t*)t.data;
            for (size_t i = 0; i < n; ++i) {
                size_t g = i / gs;
                float scale = (g < sc.shape[0]) ? scl[g] : 1.0f;
                size_t pi = i / 2;
                uint8_t b = src[pi];
                int nib = (i & 1) ? (b >> 4) : (b & 0xF);
                dst[i] = (nib & 8) ? (float)(nib - 16) * scale : (float)nib * scale;
            }
        } else { /* INT8 */
            const int8_t* src = (const int8_t*)t.data;
            for (size_t i = 0; i < n; ++i) {
                size_t g = i / gs;
                float scale = (g < sc.shape[0]) ? scl[g] : 1.0f;
                dst[i] = (float)src[i] * scale;
            }
        }

        return f32w;
    }

    /* Raw C handle */
    TbModel*       c_model() { return &model_; }
    const TbModel* c_model() const { return &model_; }

private:
    void close() {
        tb_model_free(&model_);
        model_ = {};
    }

    TbModel model_ = {};
};

}  // namespace run
}  // namespace tensorbit

#endif /* TENSORBIT_RUN_MODEL_HPP */
