#ifndef TENSORBIT_RUN_COMMON_H
#define TENSORBIT_RUN_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================
 * Platform detection
 * ================================================================ */

#if defined(_WIN32) || defined(_WIN64)
#define TB_PLATFORM_WINDOWS 1
#elif defined(__linux__)
#define TB_PLATFORM_LINUX 1
#elif defined(__APPLE__)
#define TB_PLATFORM_APPLE 1
#elif defined(ESP_PLATFORM)
#define TB_PLATFORM_ESP32 1
#endif

/* ================================================================
 * Compiler hints
 * ================================================================ */

#if defined(__GNUC__) || defined(__clang__)
#define TB_INLINE   static inline __attribute__((always_inline))
#define TB_RESTRICT __restrict__
#define TB_LIKELY(x)   __builtin_expect(!!(x), 1)
#define TB_UNLIKELY(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#define TB_INLINE   static __forceinline
#define TB_RESTRICT __restrict
#define TB_LIKELY(x)   (x)
#define TB_UNLIKELY(x) (x)
#else
#define TB_INLINE   static inline
#define TB_RESTRICT
#define TB_LIKELY(x)   (x)
#define TB_UNLIKELY(x) (x)
#endif

/* ================================================================
 * Error codes
 * ================================================================ */

typedef enum {
    TB_OK = 0,
    TB_ERR_FILE_OPEN = 1,
    TB_ERR_FILE_READ = 2,
    TB_ERR_FILE_WRITE = 3,
    TB_ERR_FILE_SEEK = 4,
    TB_ERR_BAD_MAGIC = 5,
    TB_ERR_VERSION = 6,
    TB_ERR_OOM = 7,
    TB_ERR_INVALID_ARG = 8,
    TB_ERR_INVALID_SHAPE = 9,
    TB_ERR_UNSUPPORTED_DTYPE = 10,
    TB_ERR_TRUNCATED = 11,
    TB_ERR_LAYER_NOT_FOUND = 12,
    TB_ERR_JSON_PARSE = 13,
    TB_ERR_BACKEND_NOT_FOUND = 14,
    TB_ERR_MATH_ERROR = 15,
} TbError;

/* ================================================================
 * Device types
 * ================================================================ */

typedef enum {
    TB_DEVICE_CPU = 0,
    TB_DEVICE_CUDA = 1,
    TB_DEVICE_METAL = 2,
    TB_DEVICE_VULKAN = 3,
    TB_DEVICE_NPU = 4,
} TbDevice;

/* ================================================================
 * Data type
 * ================================================================ */

typedef enum {
    TB_DTYPE_F32  = 0,
    TB_DTYPE_F16  = 1,
    TB_DTYPE_BF16 = 2,
    TB_DTYPE_F64  = 3,
    TB_DTYPE_I32  = 4,
    TB_DTYPE_U8   = 5,
    TB_DTYPE_U32  = 6,
    TB_DTYPE_INT8 = 7,
    TB_DTYPE_INT4 = 8,
} TbDtype;

TB_INLINE size_t tb_dtype_size(TbDtype dtype) {
    switch (dtype) {
        case TB_DTYPE_F32:
            return 4;
        case TB_DTYPE_F16:
            return 2;
        case TB_DTYPE_BF16:
            return 2;
        case TB_DTYPE_F64:
            return 8;
        case TB_DTYPE_I32:
            return 4;
        case TB_DTYPE_U8:
            return 1;
        case TB_DTYPE_U32:
            return 4;
        case TB_DTYPE_INT8:
            return 1;
        case TB_DTYPE_INT4:
            return 0;  /* variable: ceil(count/2) bytes for N logical elements */
        default:
            return 0;
    }
}

TB_INLINE const char* tb_error_string(int err) {
    switch (err) {
        case TB_OK:
            return "ok";
        case TB_ERR_FILE_OPEN:
            return "file open failed";
        case TB_ERR_FILE_READ:
            return "file read failed";
        case TB_ERR_FILE_WRITE:
            return "file write failed";
        case TB_ERR_FILE_SEEK:
            return "file seek failed";
        case TB_ERR_BAD_MAGIC:
            return "bad magic number";
        case TB_ERR_VERSION:
            return "unsupported version";
        case TB_ERR_OOM:
            return "out of memory";
        case TB_ERR_INVALID_ARG:
            return "invalid argument";
        case TB_ERR_INVALID_SHAPE:
            return "invalid shape";
        case TB_ERR_UNSUPPORTED_DTYPE:
            return "unsupported dtype";
        case TB_ERR_TRUNCATED:
            return "file truncated";
        case TB_ERR_LAYER_NOT_FOUND:
            return "layer not found";
        case TB_ERR_JSON_PARSE:
            return "JSON parse error";
        case TB_ERR_BACKEND_NOT_FOUND:
            return "backend not found";
        case TB_ERR_MATH_ERROR:
            return "math error";
        default:
            return "unknown error";
    }
}

/* ================================================================
 * Memory allocation
 * ================================================================ */

TB_INLINE void* tb_malloc(size_t size) { return malloc(size); }
TB_INLINE void  tb_free(void* ptr) { free(ptr); }

TB_INLINE void* tb_aligned_alloc(size_t size, size_t alignment) {
#if defined(_WIN32) || defined(_WIN64)
    return _aligned_malloc(size, alignment);
#elif defined(ESP_PLATFORM)
    return heap_caps_aligned_alloc(alignment, size, MALLOC_CAP_SPIRAM);
#else
    void* ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#endif
}

TB_INLINE void tb_aligned_free(void* ptr) {
#if defined(_WIN32) || defined(_WIN64)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/* ================================================================
 * Logging
 * ================================================================ */

#ifndef TB_LOG_DISABLE
#define TB_LOG_ERROR(fmt, ...) fprintf(stderr, "[tb-run][ERROR] " fmt "\n", ##__VA_ARGS__)
#define TB_LOG_WARN(fmt, ...)  fprintf(stderr, "[tb-run][WARN]  " fmt "\n", ##__VA_ARGS__)
#define TB_LOG_INFO(fmt, ...)  fprintf(stdout, "[tb-run][INFO]  " fmt "\n", ##__VA_ARGS__)
#ifdef TB_DEBUG
#define TB_LOG_DEBUG(fmt, ...) fprintf(stdout, "[tb-run][DEBUG] " fmt "\n", ##__VA_ARGS__)
#else
#define TB_LOG_DEBUG(fmt, ...) ((void)0)
#endif
#else
#define TB_LOG_ERROR(fmt, ...) ((void)0)
#define TB_LOG_WARN(fmt, ...)  ((void)0)
#define TB_LOG_INFO(fmt, ...)  ((void)0)
#define TB_LOG_DEBUG(fmt, ...) ((void)0)
#endif

/* ================================================================
 * Math constants
 * ================================================================ */

#define TB_PI        3.14159265358979323846f
#define TB_SQRT2     1.41421356237309504880f
#define TB_SQRT_2_PI 0.79788456080286535588f

/* ================================================================
 * Utility: min/max
 * ================================================================ */

TB_INLINE int    tb_min_i32(int a, int b) { return a < b ? a : b; }
TB_INLINE size_t tb_min_sz(size_t a, size_t b) { return a < b ? a : b; }
TB_INLINE size_t tb_max_sz(size_t a, size_t b) { return a > b ? a : b; }
TB_INLINE float  tb_max_f32(float a, float b) { return a > b ? a : b; }

#ifdef __cplusplus
}
#endif

#endif /* TENSORBIT_RUN_COMMON_H */
