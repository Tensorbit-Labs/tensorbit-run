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
    int         hidden_size = 0;
    int         num_heads = 0;
    int         num_kv_heads = 0;
    int         intermediate_size = 0;
    int         vocab_size = 0;
    int         max_seq_len = 0;
    int         num_layers = 0;
    float       norm_eps = 1e-6f;
    float       rope_theta = 10000.0f;
    int         head_dim = 0;
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
