#include "tensorbit/run/backend.hpp"
#include "tensorbit/run/common.hpp"
#include "tensorbit/run/model.hpp"
#include "tensorbit/run/ops.hpp"
#include "tensorbit/run/tensor.hpp"

namespace tensorbit {
namespace run {

/* model.cpp — C++ layer compilation unit.
 * Primary logic lives in model.c (C core) and headers (inline C++ wrappers).
 * This file exists to ensure the C++ wrapper headers compile correctly
 * and to provide a home for any C++-specific model utilities.
 *
 * Uses the same patterns as tensorbit-core:
 *  - Inline implementations in headers
 *  - .cpp files for explicit instantiation / compilation verification
 */

}  // namespace run
}  // namespace tensorbit
