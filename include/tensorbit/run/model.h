#ifndef TENSORBIT_RUN_MODEL_H
#define TENSORBIT_RUN_MODEL_H

#include "tensorbit/run/common.h"
#include "tensorbit/run/tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Layer descriptor — describes one weight tensor in the model
 * ================================================================ */

typedef struct {
    char     name[TB_MAX_TENSOR_NAME];
    size_t   shape[2];
    uint32_t nm_n;
    uint32_t nm_m;
    TbDtype  dtype;
    size_t   num_weights;
    size_t   num_mask_bytes;
    size_t   tbm_offset;
} TbLayerDesc;

/* ================================================================
 * Model config — architecture hyperparameters
 * ================================================================ */

#define TB_MAX_MODEL_NAME 64

typedef struct {
    char   architecture[TB_MAX_MODEL_NAME];
    int    hidden_size;
    int    num_heads;
    int    num_kv_heads;
    int    intermediate_size;
    int    vocab_size;
    int    max_seq_len;
    int    num_layers;
    float  norm_eps;
    float  rope_theta;
    int    head_dim;
} TbModelConfig;

/* ================================================================
 * Model handle
 * ================================================================ */

typedef struct {
    TbLayerDesc*  layers;
    int           num_layers;
    TbModelConfig config;

    /* mmap / buffer backing */
    void*         mapped_data;
    size_t        mapped_size;
    bool          owns_mapping;

    /* file handle for reading */
    void*         file_handle;
} TbModel;

/* ================================================================
 * .tbm container format constants
 * ================================================================ */

#define TBM_TAIL_SIZE 4

/* ================================================================
 * Model loading
 * ================================================================ */

/* Load from a .tbm container file */
int tb_model_load_tbm(TbModel* model, const char* path);

/* Load from a directory with manifest.json */
int tb_model_load_dir(TbModel* model, const char* dir_path);

/* Free the model and all resources */
void tb_model_free(TbModel* model);

/* Find a layer by name, returns index or -1 */
int tb_model_find_layer(const TbModel* model, const char* name);

/* Get tensor for a layer's weights (zero-copy window into mmap'd data) */
int tb_model_get_weight_tensor(const TbModel* model, int layer_idx, TbTensor* out);

/* Get tensor for a layer's mask (zero-copy window into mmap'd data) */
int tb_model_get_mask_tensor(const TbModel* model, int layer_idx, TbTensor* out);

/* ================================================================
 * .tbm builder (for tests / conversion scripts)
 * ================================================================ */

typedef struct {
    char   name[TB_MAX_TENSOR_NAME];
    size_t shape[2];
    int    nm_n;
    int    nm_m;
    TbDtype dtype;
    const void* weight_data;
    size_t weight_byte_size;
    const uint8_t* mask_data;
    size_t mask_byte_size;
} TbLayerInput;

int tb_tbm_create(const char* path, const TbModelConfig* config,
                  const TbLayerInput* layers, int num_layers);

#ifdef __cplusplus
}
#endif

#endif /* TENSORBIT_RUN_MODEL_H */
