#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tensorbit/run/ops.h"

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name)                                \
    static void test_##name(void);                \
    static int test_##name##_reg = 0;             \
    static void test_##name(void)

#define EXPECT_TRUE(expr)                                \
    do {                                                 \
        tests_run++;                                     \
        if (!(expr)) {                                   \
            fprintf(stderr, "  FAIL: %s\n", #expr);      \
        } else {                                         \
            tests_passed++;                              \
        }                                                \
    } while (0)

#define EXPECT_FLOAT_EQ(a, b, eps)                             \
    do {                                                       \
        tests_run++;                                           \
        if (fabsf((a) - (b)) > (eps)) {                        \
            fprintf(stderr, "  FAIL: %f != %f (eps=%f)\n",     \
                    (double)(a), (double)(b), (double)(eps));   \
        } else {                                               \
            tests_passed++;                                    \
        }                                                      \
    } while (0)

TEST(sparse_linear_basic) {
    size_t in_feat = 8;
    size_t out_feat = 4;
    int    nm_n = 2;
    int    nm_m = 4;

    float   x[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    /* Weights: all 1.0 for first half, all 2.0 for second half */
    size_t  nw = out_feat * in_feat;
    float*  w = (float*)malloc(nw * sizeof(float));
    for (size_t i = 0; i < nw; i++) w[i] = (i < nw / 2) ? 1.0f : 2.0f;

    /* Mask: keep first N bits in each group of M */
    size_t   groups = (in_feat / nm_m) * out_feat;
    uint8_t* mask = (uint8_t*)malloc(groups);
    for (size_t g = 0; g < groups; g++) {
        mask[g] = 0;
        for (int b = 0; b < nm_n; b++) mask[g] |= (uint8_t)(1u << b);
    }

    /* Bias: 0.0 */
    float* bias = (float*)calloc(out_feat, sizeof(float));

    float* y = (float*)calloc(out_feat, sizeof(float));

    int ret = tb_cpu_sparse_linear_f32(out_feat, in_feat, x, w, mask, bias, y, nm_n, nm_m);
    EXPECT_TRUE(ret == TB_OK);

    /* Verification:
     * For each output feature j:
     *   acc = sum over groups: sum over kept bits k: x[g*M + k] * w[j * in_feat + g*M + k]
     * First half weights = 1.0, so acc = x[0] + x[1] for first group per row (N=2, M=4)
     */

    /* Row 0: weights = 1.0
     * Group 0 (features 0-3): x[0]=1 + x[1]=2 = 3
     * Group 1 (features 4-7): x[4]=5 + x[5]=6 = 11
     * Total = 14
     */
    EXPECT_FLOAT_EQ(y[0], 14.0f, 1e-4f);

    /* Row 1: weights = 1.0
     * Same as row 0: 14
     */
    EXPECT_FLOAT_EQ(y[1], 14.0f, 1e-4f);

    /* Row 2: weights = 2.0
     * Group 0: (1+2)*2 = 6
     * Group 1: (5+6)*2 = 22
     * Total = 28
     */
    EXPECT_FLOAT_EQ(y[2], 28.0f, 1e-4f);

    /* Row 3: weights = 2.0
     * Same as row 2: 28
     */
    EXPECT_FLOAT_EQ(y[3], 28.0f, 1e-4f);

    free(w);
    free(mask);
    free(bias);
    free(y);
}

TEST(sparse_linear_with_bias) {
    size_t in_feat = 4;
    size_t out_feat = 2;
    int    nm_n = 2;
    int    nm_m = 4;

    float x[] = {1.0f, 0.0f, 1.0f, 0.0f};

    float w[] = {1.0f, 0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 2.0f, 0.0f};

    uint8_t mask[] = {0x05, 0x05};
    /* mask[0] = 0b0101 (keep bits 0 and 2)
     * mask[1] = 0b0101 (keep bits 0 and 2) */

    float bias[] = {0.5f, -0.5f};

    float y[2] = {0};

    int ret = tb_cpu_sparse_linear_f32(out_feat, in_feat, x, w, mask, bias, y, nm_n, nm_m);
    EXPECT_TRUE(ret == TB_OK);

    /* Row 0: weighted by 1.0
     * features 0,2 kept: x[0]*1.0 + x[2]*1.0 = 1+1 = 2
     * + bias = 2.5
     */
    EXPECT_FLOAT_EQ(y[0], 2.5f, 1e-4f);

    /* Row 1: weighted by 2.0
     * features 0,2 kept: x[0]*2.0 + x[2]*2.0 = 2+2 = 4
     * + bias = 3.5
     */
    EXPECT_FLOAT_EQ(y[1], 3.5f, 1e-4f);
}

TEST(dense_linear_basic) {
    size_t in_feat = 4;
    size_t out_feat = 2;

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};

    float w[] = {1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f};

    float y[2] = {0};

    int ret = tb_cpu_dense_linear_f32(out_feat, in_feat, x, w, NULL, y);
    EXPECT_TRUE(ret == TB_OK);

    /* Row 0: 1*1 + 2*0 + 3*1 + 4*0 = 4
     * Row 1: 1*0 + 2*1 + 3*0 + 4*1 = 6 */
    EXPECT_FLOAT_EQ(y[0], 4.0f, 1e-4f);
    EXPECT_FLOAT_EQ(y[1], 6.0f, 1e-4f);
}

int main(void) {
    test_sparse_linear_basic();
    test_sparse_linear_with_bias();
    test_dense_linear_basic();

    printf("test_linear: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
