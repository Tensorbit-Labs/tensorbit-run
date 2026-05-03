#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorbit/run/model.h"
#include "tensorbit/run/ops.h"

static int tests_run = 0;
static int tests_passed = 0;

#define EXPECT_TRUE(expr)                                \
    do {                                                 \
        tests_run++;                                     \
        if (!(expr)) {                                   \
            fprintf(stderr, "  FAIL: %s\n", #expr);      \
        } else {                                         \
            tests_passed++;                              \
        }                                                \
    } while (0)

int main(void) {
    const char* test_path = "test_model.tbm";

    /* Create a simple 2-layer mock model */
    {
        TbModelConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        strcpy(cfg.architecture, "test");
        cfg.hidden_size = 64;
        cfg.num_heads = 4;
        cfg.num_kv_heads = 4;
        cfg.intermediate_size = 256;
        cfg.vocab_size = 100;
        cfg.num_layers = 2;
        cfg.max_seq_len = 128;
        cfg.norm_eps = 1e-5f;
        cfg.rope_theta = 10000.0f;

        int    in_dim = 64;
        int    out_dim = 64;
        size_t nw = (size_t)out_dim * in_dim;
        size_t nb = nw * sizeof(float);
        float* w = (float*)malloc(nb);
        for (size_t i = 0; i < nw; i++)
            w[i] = ((float)(rand()) / RAND_MAX - 0.5f) * 0.02f;

        /* 2:4 mask: 1 byte per 4 weights per row */
        size_t   groups = ((size_t)in_dim / 4) * out_dim;
        uint8_t* mask = (uint8_t*)malloc(groups);
        for (size_t g = 0; g < groups; g++) mask[g] = 0x03; /* keep first 2 of 4 */

        TbLayerInput layers[1];
        memset(layers, 0, sizeof(layers));
        strcpy(layers[0].name, "test_layer.weight");
        layers[0].shape[0] = out_dim;
        layers[0].shape[1] = in_dim;
        layers[0].nm_n = 2;
        layers[0].nm_m = 4;
        layers[0].dtype = TB_DTYPE_F32;
        layers[0].weight_data = w;
        layers[0].weight_byte_size = nb;
        layers[0].mask_data = mask;
        layers[0].mask_byte_size = groups;

        int ret = tb_tbm_create(test_path, &cfg, layers, 1);
        EXPECT_TRUE(ret == TB_OK);

        free(w);
        free(mask);
    }

    /* Load back and verify */
    {
        TbModel model;
        memset(&model, 0, sizeof(model));
        int ret = tb_model_load_tbm(&model, test_path);
        EXPECT_TRUE(ret == TB_OK);
        EXPECT_TRUE(model.num_layers == 1);

        /* Check layer name */
        EXPECT_TRUE(strcmp(model.layers[0].name, "test_layer.weight") == 0);
        EXPECT_TRUE(model.layers[0].shape[0] == 64);
        EXPECT_TRUE(model.layers[0].shape[1] == 64);
        EXPECT_TRUE(model.layers[0].nm_n == 2);
        EXPECT_TRUE(model.layers[0].nm_m == 4);

        /* Get weight tensor */
        TbTensor wt;
        ret = tb_model_get_weight_tensor(&model, 0, &wt);
        EXPECT_TRUE(ret == TB_OK);
        EXPECT_TRUE(wt.data != NULL);
        EXPECT_TRUE(wt.dtype == TB_DTYPE_F32);
        EXPECT_TRUE(wt.shape[0] == 64 && wt.shape[1] == 64);

        /* Get mask tensor */
        TbTensor mt;
        ret = tb_model_get_mask_tensor(&model, 0, &mt);
        EXPECT_TRUE(ret == TB_OK);
        EXPECT_TRUE(mt.data != NULL);
        EXPECT_TRUE(mt.dtype == TB_DTYPE_U8);

        tb_model_free(&model);
    }

    /* Create and load a full transformer mock model */
    {
        TbModelConfig cfg;
        memset(&cfg, 0, sizeof(cfg));
        strcpy(cfg.architecture, "mock_llama");
        cfg.hidden_size = 128;
        cfg.num_heads = 8;
        cfg.num_kv_heads = 8;
        cfg.intermediate_size = 512;
        cfg.vocab_size = 100;
        cfg.num_layers = 1;
        cfg.max_seq_len = 32;
        cfg.norm_eps = 1e-6f;
        cfg.rope_theta = 10000.0f;

        int    hidden = 128;
        TbLayerInput layers[6];
        memset(layers, 0, sizeof(layers));

        /* Embedding */
        {
            size_t nw = (size_t)cfg.vocab_size * hidden;
            size_t nb = nw * sizeof(float);
            float* w = (float*)malloc(nb);
            memset(w, 0, nb);
            strcpy(layers[0].name, "model.embed_tokens.weight");
            layers[0].shape[0] = cfg.vocab_size;
            layers[0].shape[1] = hidden;
            layers[0].dtype = TB_DTYPE_F32;
            layers[0].weight_data = w;
            layers[0].weight_byte_size = nb;
            layers[0].mask_byte_size = 0;
        }

        /* Q projection */
        {
            size_t nw = (size_t)hidden * hidden;
            size_t nb = nw * sizeof(float);
            float* w = (float*)malloc(nb);
            memset(w, 0, nb);
            uint8_t* m = (uint8_t*)malloc((hidden / 4) * hidden);
            memset(m, 0x03, (hidden / 4) * hidden);
            strcpy(layers[1].name, "model.layers.0.self_attn.q_proj.weight");
            layers[1].shape[0] = hidden;
            layers[1].shape[1] = hidden;
            layers[1].nm_n = 2;
            layers[1].nm_m = 4;
            layers[1].dtype = TB_DTYPE_F32;
            layers[1].weight_data = w;
            layers[1].weight_byte_size = nb;
            layers[1].mask_data = m;
            layers[1].mask_byte_size = (hidden / 4) * hidden;
        }

        /* K projection */
        {
            size_t nw = (size_t)hidden * cfg.num_kv_heads * (hidden / cfg.num_heads);
            size_t nb = nw * sizeof(float);
            float* w = (float*)malloc(nb);
            memset(w, 0, nb);
            uint8_t* m = (uint8_t*)malloc((hidden / 4) * hidden);
            memset(m, 0x03, (hidden / 4) * hidden);
            strcpy(layers[2].name, "model.layers.0.self_attn.k_proj.weight");
            layers[2].shape[0] = hidden;
            layers[2].shape[1] = hidden;
            layers[2].nm_n = 2;
            layers[2].nm_m = 4;
            layers[2].dtype = TB_DTYPE_F32;
            layers[2].weight_data = w;
            layers[2].weight_byte_size = nb;
            layers[2].mask_data = m;
            layers[2].mask_byte_size = (hidden / 4) * hidden;
        }

        /* V projection */
        {
            size_t nw = (size_t)hidden * cfg.num_kv_heads * (hidden / cfg.num_heads);
            size_t nb = nw * sizeof(float);
            float* w = (float*)malloc(nb);
            memset(w, 0, nb);
            uint8_t* m = (uint8_t*)malloc((hidden / 4) * hidden);
            memset(m, 0x03, (hidden / 4) * hidden);
            strcpy(layers[3].name, "model.layers.0.self_attn.v_proj.weight");
            layers[3].shape[0] = hidden;
            layers[3].shape[1] = hidden;
            layers[3].nm_n = 2;
            layers[3].nm_m = 4;
            layers[3].dtype = TB_DTYPE_F32;
            layers[3].weight_data = w;
            layers[3].weight_byte_size = nb;
            layers[3].mask_data = m;
            layers[3].mask_byte_size = (hidden / 4) * hidden;
        }

        /* O projection */
        {
            size_t nw = (size_t)hidden * hidden;
            size_t nb = nw * sizeof(float);
            float* w = (float*)malloc(nb);
            memset(w, 0, nb);
            uint8_t* m = (uint8_t*)malloc((hidden / 4) * hidden);
            memset(m, 0x03, (hidden / 4) * hidden);
            strcpy(layers[4].name, "model.layers.0.self_attn.o_proj.weight");
            layers[4].shape[0] = hidden;
            layers[4].shape[1] = hidden;
            layers[4].nm_n = 2;
            layers[4].nm_m = 4;
            layers[4].dtype = TB_DTYPE_F32;
            layers[4].weight_data = w;
            layers[4].weight_byte_size = nb;
            layers[4].mask_data = m;
            layers[4].mask_byte_size = (hidden / 4) * hidden;
        }

        /* Input layernorm */
        {
            size_t nb = hidden * sizeof(float);
            float* w = (float*)malloc(nb);
            for (int i = 0; i < hidden; i++) w[i] = 1.0f;
            strcpy(layers[5].name, "model.layers.0.input_layernorm.weight");
            layers[5].shape[0] = hidden;
            layers[5].shape[1] = 0;
            layers[5].dtype = TB_DTYPE_F32;
            layers[5].weight_data = w;
            layers[5].weight_byte_size = nb;
        }

        const char* full_path = "test_full.tbm";
        int         ret = tb_tbm_create(full_path, &cfg, layers, 6);
        EXPECT_TRUE(ret == TB_OK);

        /* Cleanup */
        for (int i = 0; i < 6; i++) {
            if (layers[i].weight_data) free((void*)layers[i].weight_data);
            if (layers[i].mask_data) free((void*)layers[i].mask_data);
        }

        /* Load and verify */
        TbModel model;
        memset(&model, 0, sizeof(model));
        ret = tb_model_load_tbm(&model, full_path);
        EXPECT_TRUE(ret == TB_OK);
        EXPECT_TRUE(model.num_layers == 6);
        EXPECT_TRUE(model.config.hidden_size == 128);

        tb_model_free(&model);
        remove(full_path);
    }

    remove(test_path);
    printf("test_tbm: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
