/// @file model_esp32.c
/// @brief ESP32 bare-metal .tbm loader — template for ESP-IDF builds.
/// @ingroup tensorbit-run
///
/// ## Implementation Status
/// This file is a complete scaffold for ESP32 flash-based model loading.
/// When building with ESP-IDF (`ESP_PLATFORM` defined), replace the mmap-based
/// loader (model.c) with this flash-based implementation:
///   1. Add this file to CMakeLists.txt ESP32 target sources
///   2. Include the JSON parser from model.c (currently inline in `tb_model_load_tbm`)
///   3. Set the model partition label to "tbm" in your partition table
///
/// The `TbFlashFile` abstraction provides fopen/fread/fseek-like API over
/// ESP32 SPI flash via `esp_partition_read()`.  For now, desktop builds
/// use the mmap-based loader; this file serves as the ESP32 migration guide.
///
/// @see model.c for the full JSON parser reference implementation.

#ifdef ESP_PLATFORM

#include "tensorbit/run/model.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "esp_partition.h"
#include "esp_spi_flash.h"
#include "esp_system.h"

/* ================================================================
 * Flash-backed file handle
 * ================================================================ */

typedef struct {
    const esp_partition_t* part;
    size_t                 offset;
    size_t                 size;
    size_t                 cursor;  // current read position
} TbFlashFile;

/* Find the .tbm data partition by label ("tbm" or "model") */
static const esp_partition_t* find_tbm_partition(void) {
    const esp_partition_t* p = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "tbm");
    if (!p)
        p = esp_partition_find_first(
            ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    return p;
}

static int tb_flash_open(TbFlashFile* ff) {
    memset(ff, 0, sizeof(*ff));
    ff->part = find_tbm_partition();
    if (!ff->part) return TB_ERR_FILE_OPEN;
    ff->size   = (size_t)ff->part->size;
    ff->offset = (size_t)ff->part->address;
    ff->cursor = 0;
    return TB_OK;
}

static int tb_flash_read(TbFlashFile* ff, void* buf, size_t len) {
    if (ff->cursor + len > ff->size) return TB_ERR_FILE_READ;
    esp_err_t err = esp_partition_read(ff->part, ff->cursor, buf, len);
    if (err != ESP_OK) return TB_ERR_FILE_READ;
    ff->cursor += len;
    return TB_OK;
}

static int tb_flash_seek(TbFlashFile* ff, size_t pos) {
    if (pos > ff->size) return TB_ERR_INVALID_ARG;
    ff->cursor = pos;
    return TB_OK;
}

/* ================================================================
 * Model loader (ESP32 flash-backed)
 * ================================================================ */

int tb_model_load_tbm(TbModel* model, const char* path) {
    (void)path;  // path is "ignored" — model lives in a known flash partition

    if (!model) return TB_ERR_INVALID_ARG;

    TbFlashFile ff;
    int rc = tb_flash_open(&ff);
    if (rc != TB_OK) return rc;

    /* Read the 4-byte index length from the end */
    uint32_t index_len = 0;
    rc = tb_flash_seek(&ff, ff.size - sizeof(index_len));
    if (rc != TB_OK) return rc;
    rc = tb_flash_read(&ff, &index_len, sizeof(index_len));
    if (rc != TB_OK) return rc;
    if (index_len == 0 || index_len > 1024 * 1024) return TB_ERR_BAD_MAGIC;

    /* Read JSON index */
    char* json = (char*)tb_malloc(index_len + 1);
    if (!json) return TB_ERR_OOM;
    rc = tb_flash_seek(&ff, ff.size - sizeof(index_len) - index_len);
    if (rc != TB_OK) { tb_free(json); return rc; }
    rc = tb_flash_read(&ff, json, index_len);
    if (rc != TB_OK) { tb_free(json); return rc; }
    json[index_len] = '\0';

    /* Parse JSON index — hand-rolled parser (shared with model.c) */
    json[index_len] = '\0';
    int ret = parse_tbm_json(model, json);
    tb_free(json);
    if (ret != TB_OK) return ret;

    /* Read each layer's .tb header to determine offsets */
    for (int i = 0; i < model->num_layers; i++) {
        TbLayerDesc* layer = &model->layers[i];
        TBHeader    hdr;
        rc = tb_flash_seek(&ff, layer->tbm_offset);
        if (rc != TB_OK) return rc;
        rc = tb_flash_read(&ff, &hdr, TB_HEADER_SIZE);
        if (rc != TB_OK) return rc;
        if (hdr.magic != TB_MAGIC) return TB_ERR_BAD_MAGIC;
        layer->num_weights   = (size_t)hdr.num_weights;
        layer->num_mask_bytes = (size_t)hdr.num_mask_bytes;
        layer->weights_offset = layer->tbm_offset + (size_t)hdr.weights_offset;
        layer->masks_offset   = layer->tbm_offset + (size_t)hdr.masks_offset;
        layer->nm_n = hdr.nm_n;
        layer->nm_m = hdr.nm_m;
    }

    model->owns_mapping = true;
    model->mapped_data  = NULL;  // ESP32 reads on-demand from flash

    return TB_OK;
}

/* On-demand weight read from flash */
int tb_model_get_weight_tensor(TbModel* model, int index, TbTensor* out) {
    if (!model || index < 0 || index >= model->num_layers || !out) return TB_ERR_INVALID_ARG;

    TbLayerDesc* layer = &model->layers[index];
    out->data = tb_malloc(layer->num_weights * sizeof(float));
    if (!out->data) return TB_ERR_OOM;

    TbFlashFile ff;
    int rc = tb_flash_open(&ff);
    if (rc != TB_OK) { tb_free(out->data); return rc; }
    rc = tb_flash_seek(&ff, layer->weights_offset);
    if (rc != TB_OK) { tb_free(out->data); return rc; }
    rc = tb_flash_read(&ff, out->data, layer->num_weights * sizeof(float));
    if (rc != TB_OK) { tb_free(out->data); return rc; }

    out->shape[0] = layer->shape[0];
    out->shape[1] = layer->shape[1];
    out->rank = 2;
    out->dtype = TB_DTYPE_F32;
    out->device = TB_DEVICE_HOST;
    out->owns_data = true;

    return TB_OK;
}

#endif /* ESP_PLATFORM */
