#include "tensorbit/run/model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* ================================================================
 * Platform file I/O
 * ================================================================ */

typedef struct {
#ifdef _WIN32
    HANDLE handle;
    HANDLE mapping;
#else
    int    fd;
#endif
    size_t size;
    void*  data;
    int    owns_data;  // 1 = data allocated via malloc (mmap fallback)
} TbFileMapping;

static int tb_file_open_read(const char* path, TbFileMapping* fm) {
    memset(fm, 0, sizeof(*fm));
#ifdef _WIN32
    fm->handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (fm->handle == INVALID_HANDLE_VALUE) return TB_ERR_FILE_OPEN;
    LARGE_INTEGER li;
    GetFileSizeEx(fm->handle, &li);
    fm->size = (size_t)li.QuadPart;
    fm->mapping = CreateFileMappingA(fm->handle, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!fm->mapping) { CloseHandle(fm->handle); return TB_ERR_FILE_OPEN; }
    fm->data = MapViewOfFile(fm->mapping, FILE_MAP_READ, 0, 0, 0);
    if (!fm->data) { CloseHandle(fm->mapping); CloseHandle(fm->handle); return TB_ERR_FILE_OPEN; }
#else
    fm->fd = open(path, O_RDONLY);
    if (fm->fd < 0) return TB_ERR_FILE_OPEN;
    struct stat st;
    if (fstat(fm->fd, &st) != 0) { close(fm->fd); return TB_ERR_FILE_READ; }
    fm->size = (size_t)st.st_size;
    fm->data = mmap(NULL, fm->size, PROT_READ, MAP_PRIVATE, fm->fd, 0);
    if (fm->data == MAP_FAILED) {
        // mmap failed — fall back to malloc + read for environments
        // where mmap is unreliable (e.g. WSL2 DrvFs / 9p for large files)
        fm->data = tb_malloc(fm->size);
        if (!fm->data) { close(fm->fd); return TB_ERR_OOM; }
        size_t total = 0;
        while (total < fm->size) {
            ssize_t n = read(fm->fd, (char*)fm->data + total,
                             fm->size - total);
            if (n <= 0) { tb_free(fm->data); fm->data = NULL;
                          close(fm->fd); return TB_ERR_FILE_READ; }
            total += (size_t)n;
        }
        fm->owns_data = 1;
        close(fm->fd);
        fm->fd = -1;
    }
#endif
    return TB_OK;
}

static void tb_file_close(TbFileMapping* fm) {
    if (!fm->data) return;
#ifdef _WIN32
    UnmapViewOfFile(fm->data);
    CloseHandle(fm->mapping);
    CloseHandle(fm->handle);
#else
    if (fm->owns_data) {
        tb_free(fm->data);
    } else {
        munmap(fm->data, fm->size);
    }
    if (fm->fd >= 0) close(fm->fd);
#endif
    fm->data = NULL;
}

/* ================================================================
 * Minimal JSON parser
 * Handles only: {}, [], "", numbers, true/false/null
 * No unicode escapes, no floating point in numbers.
 * ================================================================ */

typedef struct {
    const char* cur;
    const char* end;
    int         err;
} TbJsonParser;

static void tb_json_skip_ws(TbJsonParser* p) {
    while (p->cur < p->end) {
        char c = *p->cur;
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            p->cur++;
        else
            break;
    }
}

static int tb_json_peek(TbJsonParser* p) {
    tb_json_skip_ws(p);
    return (p->cur < p->end) ? (unsigned char)*p->cur : -1;
}

static int tb_json_next(TbJsonParser* p) {
    tb_json_skip_ws(p);
    if (p->cur >= p->end) return -1;
    return (unsigned char)*(p->cur++);
}

static int tb_json_expect(TbJsonParser* p, char expected) {
    int c = tb_json_next(p);
    if (c != expected) {
        p->err = TB_ERR_JSON_PARSE;
        return -1;
    }
    return c;
}

static int tb_json_parse_string(TbJsonParser* p, char* out, size_t out_size) {
    int c = tb_json_next(p);
    if (c != '"') { p->err = TB_ERR_JSON_PARSE; return -1; }

    size_t i = 0;
    while (p->cur < p->end) {
        c = (unsigned char)*p->cur;
        if (c == '"') {
            p->cur++;
            if (i < out_size) out[i] = '\0';
            else if (out_size > 0) out[out_size - 1] = '\0';
            return 0;
        }
        if (c == '\\') {
            p->cur++;
            if (p->cur < p->end) {
                char esc = *p->cur;
                if (esc == '"' || esc == '\\' || esc == '/')
                    c = esc;
                else if (esc == 'n')
                    c = '\n';
                else if (esc == 't')
                    c = '\t';
                p->cur++;
            }
        } else {
            p->cur++;
        }
        if (i < out_size) out[i++] = (char)c;
    }
    p->err = TB_ERR_JSON_PARSE;
    return -1;
}

static int tb_json_parse_int(TbJsonParser* p, int64_t* out) {
    tb_json_skip_ws(p);
    if (p->cur >= p->end) { p->err = TB_ERR_JSON_PARSE; return -1; }

    int sign = 1;
    if (*p->cur == '-') { sign = -1; p->cur++; }

    int64_t val = 0;
    int     digits = 0;
    while (p->cur < p->end && *p->cur >= '0' && *p->cur <= '9') {
        val = val * 10 + (*p->cur - '0');
        p->cur++;
        digits++;
    }
    if (digits == 0) { p->err = TB_ERR_JSON_PARSE; return -1; }
    *out = sign * val;
    return 0;
}

static int tb_json_parse_float(TbJsonParser* p, double* out) {
    tb_json_skip_ws(p);
    if (p->cur >= p->end) { p->err = TB_ERR_JSON_PARSE; return -1; }

    int sign = 1;
    if (*p->cur == '-') { sign = -1; p->cur++; }

    double val = 0.0;
    int    digits = 0;
    while (p->cur < p->end && *p->cur >= '0' && *p->cur <= '9') {
        val = val * 10.0 + (*p->cur - '0');
        p->cur++;
        digits++;
    }
    if (*p->cur == '.') {
        p->cur++;
        double frac = 0.1;
        while (p->cur < p->end && *p->cur >= '0' && *p->cur <= '9') {
            val += (*p->cur - '0') * frac;
            frac *= 0.1;
            p->cur++;
            digits++;
        }
    }
    if (*p->cur == 'e' || *p->cur == 'E') {
        p->cur++;
        int esign = 1;
        if (*p->cur == '-') { esign = -1; p->cur++; }
        else if (*p->cur == '+') { p->cur++; }
        int64_t exp_val = 0;
        tb_json_parse_int(p, &exp_val);
        val *= pow(10.0, (double)(esign * exp_val));
    }
    if (digits == 0) { p->err = TB_ERR_JSON_PARSE; return -1; }
    *out = sign * val;
    return 0;
}

static int tb_json_skip_value(TbJsonParser* p) {
    int c = tb_json_peek(p);
    if (c < 0) { p->err = TB_ERR_JSON_PARSE; return -1; }

    if (c == '{') {
        tb_json_next(p);
        if (tb_json_peek(p) != '}') {
            while (1) {
                char buf[256];
                tb_json_parse_string(p, buf, sizeof(buf));
                tb_json_expect(p, ':');
                tb_json_skip_value(p);
                c = tb_json_next(p);
                if (c == '}') break;
                if (c != ',') { p->err = TB_ERR_JSON_PARSE; return -1; }
            }
        } else {
            tb_json_next(p);
        }
    } else if (c == '[') {
        tb_json_next(p);
        if (tb_json_peek(p) != ']') {
            while (1) {
                tb_json_skip_value(p);
                c = tb_json_next(p);
                if (c == ']') break;
                if (c != ',') { p->err = TB_ERR_JSON_PARSE; return -1; }
            }
        } else {
            tb_json_next(p);
        }
    } else if (c == '"') {
        char buf[4096];
        tb_json_parse_string(p, buf, sizeof(buf));
    } else if (c == 't' || c == 'f' || c == 'n') {
        while (p->cur < p->end && *p->cur >= 'a' && *p->cur <= 'z') p->cur++;
    } else {
        int64_t dummy;
        double  d_dummy;
        (void)d_dummy;
        tb_json_parse_int(p, &dummy);
    }
    return p->err;
}

/* ================================================================
 * Parse model config from JSON
 * ================================================================ */

static int tb_json_parse_config(TbJsonParser* p, TbModelConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->norm_eps = 1e-6f;
    cfg->rope_theta = 10000.0f;

    if (tb_json_peek(p) != '{') { return TB_ERR_JSON_PARSE; }
    tb_json_next(p);

    while (tb_json_peek(p) != '}') {
        char key[128];
        tb_json_parse_string(p, key, sizeof(key));
        tb_json_expect(p, ':');

        if (strcmp(key, "hidden_size") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->hidden_size = (int)v;
        } else if (strcmp(key, "num_heads") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->num_heads = (int)v;
        } else if (strcmp(key, "num_kv_heads") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->num_kv_heads = (int)v;
        } else if (strcmp(key, "intermediate_size") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->intermediate_size = (int)v;
        } else if (strcmp(key, "vocab_size") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->vocab_size = (int)v;
        } else if (strcmp(key, "num_layers") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->num_layers = (int)v;
        } else if (strcmp(key, "max_seq_len") == 0) {
            int64_t v;
            tb_json_parse_int(p, &v);
            cfg->max_seq_len = (int)v;
        } else if (strcmp(key, "norm_eps") == 0) {
            double v;
            tb_json_parse_float(p, &v);
            cfg->norm_eps = (float)v;
        } else if (strcmp(key, "rope_theta") == 0) {
            double v;
            tb_json_parse_float(p, &v);
            cfg->rope_theta = (float)v;
        } else {
            tb_json_skip_value(p);
        }

        int c = tb_json_next(p);
        if (c == '}') break;
        if (c != ',') { return TB_ERR_JSON_PARSE; }
    }

    if (cfg->num_kv_heads == 0) cfg->num_kv_heads = cfg->num_heads;
    cfg->head_dim = cfg->hidden_size / cfg->num_heads;
    return p->err ? TB_ERR_JSON_PARSE : TB_OK;
}

/* ================================================================
 * Parse tensor array from JSON
 * ================================================================ */

static int tb_json_parse_tensors(TbJsonParser* p, TbLayerDesc* layers, int max_layers, int* out_count) {
    *out_count = 0;

    if (tb_json_peek(p) != '[') { return TB_ERR_JSON_PARSE; }
    tb_json_next(p);

    while (tb_json_peek(p) != ']' && *out_count < max_layers) {
        if (tb_json_peek(p) != '{') { return TB_ERR_JSON_PARSE; }
        tb_json_next(p);

        TbLayerDesc* layer = &layers[*out_count];
        memset(layer, 0, sizeof(*layer));

        while (tb_json_peek(p) != '}') {
            char key[128];
            tb_json_parse_string(p, key, sizeof(key));
            tb_json_expect(p, ':');

            if (strcmp(key, "name") == 0) {
                tb_json_parse_string(p, layer->name, sizeof(layer->name));
            } else if (strcmp(key, "offset") == 0) {
                int64_t v;
                tb_json_parse_int(p, &v);
                layer->tbm_offset = (size_t)v;
            } else if (strcmp(key, "shape") == 0) {
                if (tb_json_peek(p) != '[') { return TB_ERR_JSON_PARSE; }
                tb_json_next(p);
                for (int s = 0; s < 2 && tb_json_peek(p) != ']'; s++) {
                    int64_t v;
                    tb_json_parse_int(p, &v);
                    layer->shape[s] = (size_t)v;
                    int c = tb_json_next(p);
                    if (c == ']') break;
                    if (c != ',') { return TB_ERR_JSON_PARSE; }
                }
            } else if (strcmp(key, "nm_n") == 0) {
                int64_t v;
                tb_json_parse_int(p, &v);
                layer->nm_n = (uint32_t)v;
            } else if (strcmp(key, "nm_m") == 0) {
                int64_t v;
                tb_json_parse_int(p, &v);
                layer->nm_m = (uint32_t)v;
            } else if (strcmp(key, "dtype") == 0) {
                char dtype_str[32];
                tb_json_parse_string(p, dtype_str, sizeof(dtype_str));
                if (strcmp(dtype_str, "fp32") == 0) layer->dtype = TB_DTYPE_F32;
                else if (strcmp(dtype_str, "fp16") == 0) layer->dtype = TB_DTYPE_F16;
                else if (strcmp(dtype_str, "bf16") == 0) layer->dtype = TB_DTYPE_BF16;
                else if (strcmp(dtype_str, "fp64") == 0) layer->dtype = TB_DTYPE_F64;
            } else if (strcmp(key, "num_weights") == 0) {
                int64_t v;
                tb_json_parse_int(p, &v);
                layer->num_weights = (size_t)v;
            } else if (strcmp(key, "num_mask_bytes") == 0) {
                int64_t v;
                tb_json_parse_int(p, &v);
                layer->num_mask_bytes = (size_t)v;
            } else {
                tb_json_skip_value(p);
            }

            int c = tb_json_next(p);
            if (c == '}') break;
            if (c != ',') { return TB_ERR_JSON_PARSE; }
        }

        (*out_count)++;

        int c = tb_json_next(p);
        if (c == ']') break;
        if (c != ',') { return TB_ERR_JSON_PARSE; }
    }

    if (p->err) return TB_ERR_JSON_PARSE;
    return TB_OK;
}

/* ================================================================
 * Parse .tbm JSON index
 * ================================================================ */

static int tb_parse_tbm_index(const char* json, size_t json_len,
                               TbModelConfig* config, TbLayerDesc* layers,
                               int max_layers, int* out_count) {
    TbJsonParser p;
    p.cur = json;
    p.end = json + json_len;
    p.err = TB_OK;

    if (tb_json_peek(&p) != '{') return TB_ERR_JSON_PARSE;
    tb_json_next(&p);

    while (tb_json_peek(&p) != '}') {
        char key[128];
        tb_json_parse_string(&p, key, sizeof(key));
        tb_json_expect(&p, ':');

        if (strcmp(key, "architecture") == 0) {
            char arch[TB_MAX_MODEL_NAME];
            tb_json_parse_string(&p, arch, sizeof(arch));
            memcpy(config->architecture, arch, sizeof(config->architecture));
            config->architecture[sizeof(config->architecture) - 1] = '\0';
        } else if (strcmp(key, "config") == 0) {
            tb_json_parse_config(&p, config);
        } else if (strcmp(key, "tensors") == 0) {
            tb_json_parse_tensors(&p, layers, max_layers, out_count);
        } else {
            tb_json_skip_value(&p);
        }

        int c = tb_json_next(&p);
        if (c == '}') break;
        if (c != ',') return TB_ERR_JSON_PARSE;
    }

    if (p.err) return TB_ERR_JSON_PARSE;
    if (config->num_layers == 0 && *out_count > 0) config->num_layers = 1;
    return TB_OK;
}

/* ================================================================
 * Load .tbm container
 * ================================================================ */

int tb_model_load_tbm(TbModel* model, const char* path) {
    memset(model, 0, sizeof(*model));

    TbFileMapping fm;
    int           ret = tb_file_open_read(path, &fm);
    if (ret != TB_OK) return ret;

    if (fm.size < TBM_TAIL_SIZE + 16) {
        tb_file_close(&fm);
        return TB_ERR_TRUNCATED;
    }

    /* read 4-byte index length from end */
    const uint8_t* base = (const uint8_t*)fm.data;
    uint32_t       index_len;
    memcpy(&index_len, base + fm.size - TBM_TAIL_SIZE, sizeof(index_len));

    if (index_len == 0 || (size_t)index_len > fm.size - TBM_TAIL_SIZE) {
        tb_file_close(&fm);
        return TB_ERR_TRUNCATED;
    }

    size_t index_offset = fm.size - TBM_TAIL_SIZE - index_len;
    char*  json_buf = (char*)tb_malloc(index_len + 1);
    if (!json_buf) {
        tb_file_close(&fm);
        return TB_ERR_OOM;
    }

    memcpy(json_buf, base + index_offset, index_len);
    json_buf[index_len] = '\0';

    int           max_layers = 512;
    TbLayerDesc*  layers = (TbLayerDesc*)tb_malloc(sizeof(TbLayerDesc) * max_layers);
    if (!layers) {
        tb_free(json_buf);
        tb_file_close(&fm);
        return TB_ERR_OOM;
    }

    int num_layers = 0;
    ret = tb_parse_tbm_index(json_buf, index_len, &model->config, layers, max_layers, &num_layers);
    tb_free(json_buf);

    if (ret != TB_OK) {
        tb_free(layers);
        tb_file_close(&fm);
        return ret;
    }

    /* validate each layer's .tb header */
    for (int i = 0; i < num_layers; i++) {
        size_t offset = layers[i].tbm_offset;
        if (offset + TB_HEADER_SIZE > fm.size) {
            tb_free(layers);
            tb_file_close(&fm);
            return TB_ERR_TRUNCATED;
        }

        const TBHeader* hdr = (const TBHeader*)(base + offset);
        if (hdr->magic != TB_MAGIC || hdr->version != TB_VERSION) {
            TB_LOG_WARN("layer %d '%s': bad magic or version", i, layers[i].name);
        }
    }

    model->layers = layers;
    model->num_layers = num_layers;
    model->mapped_data = fm.data;
    model->mapped_size = fm.size;
    model->owns_mapping = true;
    /* keep fm.data alive — don't close */

    /* stash platform handles for cleanup */
#ifdef _WIN32
    model->file_handle = (void*)(uintptr_t)fm.handle;
    /* We need mapping and handle for cleanup. Store in a small struct. */
    {
        struct {
            HANDLE handle;
            HANDLE mapping;
        }* fh;
        fh = (void*)tb_malloc(sizeof(*fh));
        fh->handle = fm.handle;
        fh->mapping = fm.mapping;
        model->file_handle = fh;
    }
#else
    {
        int* fdp = (int*)tb_malloc(sizeof(int));
        *fdp = fm.fd;
        model->file_handle = fdp;
    }
#endif

    TB_LOG_INFO("loaded %s: %d layers, arch=%s, hidden=%d, layers=%d",
                path, num_layers, model->config.architecture,
                model->config.hidden_size, model->config.num_layers);
    return TB_OK;
}

/* ================================================================
 * Load from directory with manifest.json
 * ================================================================ */

int tb_model_load_dir(TbModel* model, const char* dir_path) {
    memset(model, 0, sizeof(*model));

    char   manifest_path[1024];
    if ((size_t)snprintf(manifest_path, sizeof(manifest_path), "%s/manifest.json", dir_path) >= sizeof(manifest_path)) {
        return TB_ERR_INVALID_ARG;
    }

    FILE* f = fopen(manifest_path, "rb");
    if (!f) return TB_ERR_FILE_OPEN;

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* json_buf = (char*)tb_malloc(fsize + 1);
    if (!json_buf) { fclose(f); return TB_ERR_OOM; }

    fread(json_buf, 1, fsize, f);
    json_buf[fsize] = '\0';
    fclose(f);

    int          max_layers = 512;
    TbLayerDesc* layers = (TbLayerDesc*)tb_malloc(sizeof(TbLayerDesc) * max_layers);
    int          num_layers = 0;

    int ret = tb_parse_tbm_index(json_buf, fsize, &model->config, layers, max_layers, &num_layers);
    tb_free(json_buf);

    if (ret != TB_OK) {
        tb_free(layers);
        return ret;
    }

    model->layers = layers;
    model->num_layers = num_layers;
    model->owns_mapping = false;
    model->mapped_data = NULL;

    TB_LOG_INFO("loaded directory %s: %d layers", dir_path, num_layers);
    return TB_OK;
}

/* ================================================================
 * Free model
 * ================================================================ */

void tb_model_free(TbModel* model) {
    if (model->layers) {
        tb_free(model->layers);
        model->layers = NULL;
    }

    if (model->owns_mapping && model->mapped_data && model->file_handle) {
#ifdef _WIN32
        struct {
            HANDLE handle;
            HANDLE mapping;
        }* fh = (void*)model->file_handle;
        UnmapViewOfFile(model->mapped_data);
        CloseHandle(fh->mapping);
        CloseHandle(fh->handle);
        tb_free(fh);
#else
        int* fdp = (int*)model->file_handle;
        munmap(model->mapped_data, model->mapped_size);
        close(*fdp);
        tb_free(fdp);
#endif
    }

    if (model->file_handle && !model->owns_mapping) {
        tb_free(model->file_handle);
    }

    model->mapped_data = NULL;
    model->file_handle = NULL;
    model->num_layers = 0;
}

/* ================================================================
 * Find layer by name
 * ================================================================ */

int tb_model_find_layer(const TbModel* model, const char* name) {
    for (int i = 0; i < model->num_layers; i++) {
        if (strcmp(model->layers[i].name, name) == 0) return i;
    }
    return -1;
}

/* ================================================================
 * Get weight/mask tensor for a layer
 * ================================================================ */

int tb_model_get_weight_tensor(const TbModel* model, int layer_idx, TbTensor* out) {
    if (layer_idx < 0 || layer_idx >= model->num_layers) return TB_ERR_LAYER_NOT_FOUND;
    if (!model->mapped_data) return TB_ERR_INVALID_ARG;

    const TbLayerDesc* layer = &model->layers[layer_idx];
    const uint8_t*     base = (const uint8_t*)model->mapped_data;
    size_t             offset = layer->tbm_offset;

    const TBHeader* hdr = (const TBHeader*)(base + offset);
    if (hdr->magic != TB_MAGIC) return TB_ERR_BAD_MAGIC;

    size_t shape[2] = {layer->shape[0], layer->shape[1]};
    *out = tb_tensor_wrap((void*)(base + offset + hdr->weights_offset), shape, 2, layer->dtype,
                           TB_DEVICE_CPU);
    return TB_OK;
}

int tb_model_get_mask_tensor(const TbModel* model, int layer_idx, TbTensor* out) {
    if (layer_idx < 0 || layer_idx >= model->num_layers) return TB_ERR_LAYER_NOT_FOUND;
    if (!model->mapped_data) return TB_ERR_INVALID_ARG;

    const TbLayerDesc* layer = &model->layers[layer_idx];
    const uint8_t*     base = (const uint8_t*)model->mapped_data;
    size_t             offset = layer->tbm_offset;

    const TBHeader* hdr = (const TBHeader*)(base + offset);
    if (hdr->magic != TB_MAGIC) return TB_ERR_BAD_MAGIC;

    /* mask is 1D: one byte per M-sized group */
    size_t shape[1] = {layer->num_mask_bytes};
    *out = tb_tensor_wrap((void*)(base + offset + hdr->masks_offset), shape, 1, TB_DTYPE_U8,
                           TB_DEVICE_CPU);
    return TB_OK;
}

/* ================================================================
 * .tbm builder (for tests)
 * ================================================================ */

#include <stdio.h>

int tb_tbm_create(const char* path, const TbModelConfig* config, const TbLayerInput* layers,
                  int num_layers) {
    FILE* f = fopen(path, "wb");
    if (!f) return TB_ERR_FILE_OPEN;

    size_t* offsets = (size_t*)tb_malloc(sizeof(size_t) * num_layers);
    if (!offsets) { fclose(f); return TB_ERR_OOM; }

    /* write each .tb blob */
    for (int i = 0; i < num_layers; i++) {
        offsets[i] = (size_t)ftell(f);

        TBHeader hdr;
        memset(&hdr, 0, sizeof(hdr));
        hdr.magic = TB_MAGIC;
        hdr.version = TB_VERSION;
        hdr.nm_n = layers[i].nm_n;
        hdr.nm_m = layers[i].nm_m;
        hdr.num_weights = layers[i].weight_byte_size / tb_dtype_size(layers[i].dtype);
        hdr.num_mask_bytes = layers[i].mask_byte_size;
        hdr.weights_offset = TB_HEADER_SIZE;
        hdr.masks_offset = TB_HEADER_SIZE + layers[i].weight_byte_size;
        switch (layers[i].dtype) {
            case TB_DTYPE_F32:
                hdr.precision = 0;
                break;
            case TB_DTYPE_F16:
                hdr.precision = 1;
                break;
            case TB_DTYPE_BF16:
                hdr.precision = 2;
                break;
            case TB_DTYPE_F64:
                hdr.precision = 3;
                break;
            default:
                hdr.precision = 0;
                break;
        }

        fwrite(&hdr, sizeof(hdr), 1, f);
        if (layers[i].weight_data && layers[i].weight_byte_size > 0) {
            fwrite(layers[i].weight_data, 1, layers[i].weight_byte_size, f);
        }
        if (layers[i].mask_data && layers[i].mask_byte_size > 0) {
            fwrite(layers[i].mask_data, 1, layers[i].mask_byte_size, f);
        }
    }

    /* build JSON index */
    char* json = (char*)tb_malloc(1024 * 1024);
    if (!json) {
        tb_free(offsets);
        fclose(f);
        return TB_ERR_OOM;
    }

    int pos = snprintf(json, 1024 * 1024, "{\"architecture\":\"%s\",", config->architecture);
    pos += snprintf(json + pos, 1024 * 1024 - pos,
                    "\"config\":{"
                    "\"hidden_size\":%d,"
                    "\"num_heads\":%d,"
                    "\"num_kv_heads\":%d,"
                    "\"intermediate_size\":%d,"
                    "\"vocab_size\":%d,"
                    "\"num_layers\":%d,"
                    "\"max_seq_len\":%d,"
                    "\"norm_eps\":%g,"
                    "\"rope_theta\":%g"
                    "},",
                    config->hidden_size, config->num_heads, config->num_kv_heads,
                    config->intermediate_size, config->vocab_size, config->num_layers,
                    config->max_seq_len, (double)config->norm_eps, (double)config->rope_theta);

    pos += snprintf(json + pos, 1024 * 1024 - pos, "\"tensors\":[");
    for (int i = 0; i < num_layers; i++) {
        const char* dtype_str = "fp32";
        switch (layers[i].dtype) {
            case TB_DTYPE_F16:
                dtype_str = "fp16";
                break;
            case TB_DTYPE_BF16:
                dtype_str = "bf16";
                break;
            case TB_DTYPE_F64:
                dtype_str = "fp64";
                break;
            default:
                break;
        }
        size_t nw = layers[i].weight_byte_size / tb_dtype_size(layers[i].dtype);
        pos += snprintf(json + pos, 1024 * 1024 - pos,
                        "%s{\"name\":\"%s\",\"offset\":%zu,\"shape\":[%zu,%zu],"
                        "\"nm_n\":%d,\"nm_m\":%d,\"dtype\":\"%s\","
                        "\"num_weights\":%zu,\"num_mask_bytes\":%zu}",
                        i > 0 ? "," : "", layers[i].name, offsets[i], layers[i].shape[0],
                        layers[i].shape[1], layers[i].nm_n, layers[i].nm_m, dtype_str, nw,
                        layers[i].mask_byte_size);
    }
    pos += snprintf(json + pos, 1024 * 1024 - pos, "]}");

    /* write JSON */
    size_t json_len = strlen(json);
    fwrite(json, 1, json_len, f);

    /* write 4-byte index length */
    uint32_t index_len = (uint32_t)json_len;
    fwrite(&index_len, sizeof(index_len), 1, f);

    tb_free(json);
    tb_free(offsets);
    fclose(f);
    return TB_OK;
}
