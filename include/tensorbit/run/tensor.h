#ifndef TENSORBIT_RUN_TENSOR_H
#define TENSORBIT_RUN_TENSOR_H

#include "tensorbit/run/common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * TBHeader — matches tensorbit-core serialization format
 * Must be exactly 4096 bytes (packed, no padding between fields).
 * ================================================================ */

#define TB_HEADER_SIZE    4096
#define TB_MAGIC          0x31304254
#define TB_VERSION        1
#define TB_MAX_RANK       8
#define TB_MAX_TENSOR_NAME 128

#pragma pack(push, 1)
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t nm_n;
    uint32_t nm_m;
    uint64_t num_weights;
    uint64_t num_mask_bytes;
    uint64_t weights_offset;
    uint64_t masks_offset;
    uint8_t  precision;
    uint8_t  reserved[4047];
} TBHeader;
#pragma pack(pop)

/* compile-time size check */
#ifdef __cplusplus
static_assert(sizeof(TBHeader) == 4096, "TBHeader must be exactly 4096 bytes");
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(TBHeader) == 4096, "TBHeader must be exactly 4096 bytes");
#endif

/* ================================================================
 * Tensor
 * ================================================================ */

typedef struct {
    void*    data;
    size_t   shape[TB_MAX_RANK];
    uint8_t  rank;
    TbDtype  dtype;
    TbDevice device;
    bool     owns_data;
} TbTensor;

/* Create an empty tensor (no allocation) */
TB_INLINE TbTensor tb_tensor_empty(void) {
    TbTensor t;
    t.data = NULL;
    memset(t.shape, 0, sizeof(t.shape));
    t.rank = 0;
    t.dtype = TB_DTYPE_F32;
    t.device = TB_DEVICE_CPU;
    t.owns_data = false;
    return t;
}

/* Create a tensor with allocated memory */
TB_INLINE TbTensor tb_tensor_create(size_t shape[], uint8_t rank, TbDtype dtype, TbDevice device) {
    TbTensor t;
    t.data = NULL;
    t.rank = 0;
    t.dtype = dtype;
    t.device = device;
    t.owns_data = false;
    memset(t.shape, 0, sizeof(t.shape));

    if (rank > TB_MAX_RANK) {
        TB_LOG_ERROR("tensor rank %u exceeds max %u", rank, (unsigned)TB_MAX_RANK);
        return t;
    }

    size_t nelem = 1;
    for (uint8_t i = 0; i < rank; i++) {
        nelem *= shape[i];
        t.shape[i] = shape[i];
    }
    t.rank = rank;

    if (nelem > 0) {
        size_t byte_size = nelem * tb_dtype_size(dtype);
        t.data = tb_aligned_alloc(byte_size, 64);
        if (t.data) {
            memset(t.data, 0, byte_size);
            t.owns_data = true;
        }
    }

    return t;
}

/* Create a tensor wrapping external data (no ownership) */
TB_INLINE TbTensor tb_tensor_wrap(void* data, size_t shape[], uint8_t rank, TbDtype dtype, TbDevice device) {
    TbTensor t;
    t.data = data;
    t.dtype = dtype;
    t.device = device;
    t.owns_data = false;
    t.rank = 0;
    memset(t.shape, 0, sizeof(t.shape));
    if (rank <= TB_MAX_RANK) {
        for (uint8_t i = 0; i < rank; i++) t.shape[i] = shape[i];
        t.rank = rank;
    }
    return t;
}

/* Total number of elements */
TB_INLINE size_t tb_tensor_nelem(const TbTensor* t) {
    size_t n = 1;
    for (uint8_t i = 0; i < t->rank; i++) n *= t->shape[i];
    return n;
}

/* Byte size */
TB_INLINE size_t tb_tensor_bytes(const TbTensor* t) {
    return tb_tensor_nelem(t) * tb_dtype_size(t->dtype);
}

/* Free owned data */
TB_INLINE void tb_tensor_free(TbTensor* t) {
    if (t->owns_data && t->data) {
        tb_aligned_free(t->data);
    }
    t->data = NULL;
    t->rank = 0;
    t->owns_data = false;
}

/* Float pointer accessor */
TB_INLINE float* tb_tensor_f32(TbTensor* t) { return (float*)t->data; }
TB_INLINE const float* tb_tensor_cf32(const TbTensor* t) { return (const float*)t->data; }

/* float* from offset */
TB_INLINE float* tb_tensor_f32_at(TbTensor* t, size_t offset) { return ((float*)t->data) + offset; }

#ifdef __cplusplus
}
#endif

#endif /* TENSORBIT_RUN_TENSOR_H */
