#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    /* Simple scaled dot-product: 1 head, seq_len=2, head_dim=4 */
    {
        int n_heads = 1;
        int seq_len = 2;
        int head_dim = 4;

        /* q, k, v each [n_heads * seq_len * head_dim] = [8] */
        float q[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        float k[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
        float v[] = {2.0f, 0.0f, 2.0f, 0.0f, 0.0f, 2.0f, 0.0f, 2.0f};

        float* y = (float*)calloc(n_heads * seq_len * head_dim, sizeof(float));

        float scale = 1.0f / sqrtf((float)head_dim);

        int ret = tb_cpu_scaled_dot_product_f32(n_heads, seq_len, head_dim, q, k, v, NULL, y,
                                                  false, scale);
        EXPECT_TRUE(ret == TB_OK);

        /* Verify y is not NaN */
        for (int i = 0; i < n_heads * seq_len * head_dim; i++) {
            EXPECT_TRUE(!isnan(y[i]));
            EXPECT_TRUE(!isinf(y[i]));
        }

        free(y);
    }

    /* Causal attention: ensure position i can't attend to positions > i */
    {
        int n_heads = 1;
        int seq_len = 3;
        int head_dim = 2;

        float q[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        float k[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
        float v[] = {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};

        float* y = (float*)calloc(n_heads * seq_len * head_dim, sizeof(float));

        float scale = 1.0f;
        int   ret = tb_cpu_scaled_dot_product_f32(n_heads, seq_len, head_dim, q, k, v, NULL, y,
                                                    true, scale);
        EXPECT_TRUE(ret == TB_OK);

        for (int i = 0; i < n_heads * seq_len * head_dim; i++) {
            EXPECT_TRUE(!isnan(y[i]));
        }

        free(y);
    }

    printf("test_attention: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
