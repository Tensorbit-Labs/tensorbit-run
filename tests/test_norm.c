#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "tensorbit/run/ops.h"

static int tests_run = 0;
static int tests_passed = 0;

#define EXPECT_FLOAT_EQ(a, b, eps)                             \
    do {                                                       \
        tests_run++;                                           \
        if (fabsf((a) - (b)) > (eps)) {                        \
            fprintf(stderr, "  FAIL: %f != %f\n",              \
                    (double)(a), (double)(b));                  \
        } else {                                               \
            tests_passed++;                                    \
        }                                                      \
    } while (0)

int main(void) {
    /* RMS Norm test */
    {
        float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float y[4];
        int   ret = tb_cpu_rms_norm_f32(4, x, weight, y, 1e-6f);
        if (ret != TB_OK) { printf("rms_norm returned error\n"); return 1; }

        /* mean(x^2) = (1+4+9+16)/4 = 7.5
         * rms = sqrt(7.5 + eps) ≈ 2.7386
         * y = x / rms
         */
        float mean_sq = (1.0f + 4.0f + 9.0f + 16.0f) / 4.0f;
        float rms = sqrtf(mean_sq + 1e-6f);
        EXPECT_FLOAT_EQ(y[0], 1.0f / rms, 1e-4f);
        EXPECT_FLOAT_EQ(y[1], 2.0f / rms, 1e-4f);
        EXPECT_FLOAT_EQ(y[2], 3.0f / rms, 1e-4f);
        EXPECT_FLOAT_EQ(y[3], 4.0f / rms, 1e-4f);
    }

    /* RMS Norm with weight */
    {
        float x[] = {2.0f, 2.0f, 2.0f, 2.0f};
        float weight[] = {0.5f, 1.0f, 1.5f, 2.0f};
        float y[4];
        tb_cpu_rms_norm_f32(4, x, weight, y, 0.0f);

        /* mean(x^2) = 4, rms = 2
         * y = (x/2) * weight = x[i] * weight[i] / 2
         * = {0.5, 1.0, 1.5, 2.0}
         */
        EXPECT_FLOAT_EQ(y[0], 0.5f, 1e-4f);
        EXPECT_FLOAT_EQ(y[1], 1.0f, 1e-4f);
        EXPECT_FLOAT_EQ(y[2], 1.5f, 1e-4f);
        EXPECT_FLOAT_EQ(y[3], 2.0f, 1e-4f);
    }

    /* Layer Norm test */
    {
        float x[] = {1.0f, 3.0f, 5.0f, 7.0f};
        float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float b[] = {0.0f, 0.0f, 0.0f, 0.0f};
        float y[4];
        tb_cpu_layer_norm_f32(4, x, w, b, y, 1e-6f);

        /* mean = (1+3+5+7)/4 = 4
         * var = ((1-4)^2+(3-4)^2+(5-4)^2+(7-4)^2)/4
         *     = (9+1+1+9)/4 = 5
         * std = sqrt(5 + eps)
         * y[i] = (x[i] - 4) / std
         */
        float mean = 4.0f;
        float var = 5.0f;
        float std = sqrtf(var + 1e-6f);
        EXPECT_FLOAT_EQ(y[0], (1.0f - mean) / std, 1e-4f);
        EXPECT_FLOAT_EQ(y[1], (3.0f - mean) / std, 1e-4f);
        EXPECT_FLOAT_EQ(y[2], (5.0f - mean) / std, 1e-4f);
        EXPECT_FLOAT_EQ(y[3], (7.0f - mean) / std, 1e-4f);
    }

    printf("test_norm: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
