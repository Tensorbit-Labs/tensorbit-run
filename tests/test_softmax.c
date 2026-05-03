#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    /* Basic softmax */
    {
        float x[] = {1.0f, 2.0f, 3.0f};
        float y[3];
        int   ret = tb_cpu_softmax_f32(3, x, NULL, y);
        EXPECT_TRUE(ret == TB_OK);

        /* exp(1-3)/Z + exp(2-3)/Z + exp(3-3)/Z should equal 1 */
        float sum = y[0] + y[1] + y[2];
        EXPECT_FLOAT_EQ(sum, 1.0f, 1e-4f);
        EXPECT_TRUE(y[2] > y[1]);
        EXPECT_TRUE(y[1] > y[0]);
    }

    /* Softmax with all equal values */
    {
        float x[] = {5.0f, 5.0f, 5.0f, 5.0f};
        float y[4];
        tb_cpu_softmax_f32(4, x, NULL, y);

        EXPECT_FLOAT_EQ(y[0], 0.25f, 1e-4f);
        EXPECT_FLOAT_EQ(y[1], 0.25f, 1e-4f);
        EXPECT_FLOAT_EQ(y[2], 0.25f, 1e-4f);
        EXPECT_FLOAT_EQ(y[3], 0.25f, 1e-4f);
    }

    /* Softmax with mask */
    {
        float x[] = {1.0f, 2.0f, 3.0f};
        float mask[] = {0.0f, 0.0f, -100.0f}; /* mask last element */
        float y[3];
        tb_cpu_softmax_f32(3, x, mask, y);

        /* Last element should be effectively 0 */
        EXPECT_FLOAT_EQ(y[2], 0.0f, 1e-3f);
        EXPECT_TRUE(y[1] > y[0]);
    }

    printf("test_softmax: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
