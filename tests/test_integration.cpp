#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "tensorbit/run/backend.hpp"
#include "tensorbit/run/common.hpp"
#include "tensorbit/run/model.hpp"
#include "tensorbit/run/ops.hpp"
#include "tensorbit/run/tensor.hpp"

using namespace tensorbit::run;

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
        if (std::abs((a) - (b)) > (eps)) {                     \
            fprintf(stderr, "  FAIL: %f != %f\n",              \
                    (double)(a), (double)(b));                  \
        } else {                                               \
            tests_passed++;                                    \
        }                                                      \
    } while (0)

/* Simple test to verify Tensor RAII works */
static void test_tensor_raii() {
    {
        Tensor t({4, 8}, Dtype::kF32);
        EXPECT_TRUE(t.data() != nullptr);
        EXPECT_TRUE(t.rank() == 2);
        EXPECT_TRUE(t.shape(0) == 4);
        EXPECT_TRUE(t.shape(1) == 8);
        EXPECT_TRUE(t.size() == 32);
    }
    /* Destructor called — no leak */

    /* Move semantics */
    Tensor t1({2, 2}, Dtype::kF32);
    float* ptr = t1.f32();
    Tensor t2 = std::move(t1);
    EXPECT_TRUE(t2.f32() == ptr);
    EXPECT_TRUE(t1.data() == nullptr);
}

/* Test that backend registry initializes properly */
static void test_backend_init() {
    BackendRegistry::instance().init();
    EXPECT_TRUE(BackendRegistry::instance().count() >= 1);
    EXPECT_TRUE(BackendRegistry::instance().has_backend("cpu"));
}

/* Test that sparse linear dispatch works through C++ wrappers */
static void test_sparse_linear_dispatch() {
    BackendRegistry::instance().init();

    Tensor x({4}, Dtype::kF32);
    Tensor w({2, 4}, Dtype::kF32);
    Tensor mask({2}, Dtype::kU8);
    Tensor bias({2}, Dtype::kF32);
    Tensor y({2}, Dtype::kF32);

    /* x = [1, 0, 1, 0] */
    x.f32()[0] = 1.0f;
    x.f32()[1] = 0.0f;
    x.f32()[2] = 1.0f;
    x.f32()[3] = 0.0f;

    /* w = [[1, 1, 1, 1], [2, 2, 2, 2]] */
    for (int i = 0; i < 8; i++) w.f32()[i] = (i < 4) ? 1.0f : 2.0f;

    /* mask: keep first 2 of 4 (bit 0 and 1 set) */
    mask.u8()[0] = 0x03;
    mask.u8()[1] = 0x03;

    int ret = sparse_linear(y, x, w, mask, bias);
    EXPECT_TRUE(ret == TB_OK);

    /* Row 0: x[0]*1 + x[1]*0 = 1 (only first 2 kept per group, x[2],x[3] skipped)
     * Wait, mask is per row: 2 groups of size 4, each group has 2 bytes.
     * For in_features=4, each row has 1 group of 4.
     * mask[0] = row 0: group 0, mask[1] = row 1: group 0
     * Actually mask is [out_features * in_features/M] = [2 * 1] = 2
     * mask[0] for row 0, mask[1] for row 1
     * Row 0: only x[0] and x[1] used -> 1*1 + 0*1 = 1
     * Row 1: only x[0] and x[1] used -> 1*2 + 0*2 = 2
     */
    EXPECT_FLOAT_EQ(y.f32()[0], 1.0f, 1e-4f);
    EXPECT_FLOAT_EQ(y.f32()[1], 2.0f, 1e-4f);
}

int main() {
    test_tensor_raii();
    test_backend_init();
    test_sparse_linear_dispatch();

    fprintf(stderr, "test_integration: %d/%d passed\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
