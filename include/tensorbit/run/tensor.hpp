#ifndef TENSORBIT_RUN_TENSOR_HPP
#define TENSORBIT_RUN_TENSOR_HPP

#include <cstring>
#include <initializer_list>
#include <memory>
#include <span>
#include <vector>

#include "tensorbit/run/common.hpp"
#include "tensorbit/run/tensor.h"

namespace tensorbit {
namespace run {

/* ================================================================
 * Device location enum (C++ mirror of C TbDevice)
 * ================================================================ */

enum class DeviceLocation { kHost = 0, kCuda = 1, kMetal = 2, kVulkan = 3, kNpu = 4 };

/* ================================================================
 * Dtype enum (C++ mirror of C TbDtype)
 * ================================================================ */

enum class Dtype { kF32 = 0, kF16 = 1, kBF16 = 2, kF64 = 3, kI32 = 4, kU8 = 5, kU32 = 6 };

inline size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::kF32:
            return 4;
        case Dtype::kF16:
            return 2;
        case Dtype::kBF16:
            return 2;
        case Dtype::kF64:
            return 8;
        case Dtype::kI32:
            return 4;
        case Dtype::kU8:
            return 1;
        case Dtype::kU32:
            return 4;
        default:
            return 0;
    }
}

/* ================================================================
 * Tensor — RAII wrapper around C TbTensor
 * ================================================================ */

static constexpr size_t kMaxRank = 8;

class Tensor {
public:
    Tensor() : tensor_(tb_tensor_empty()) {}

    Tensor(std::initializer_list<size_t> shape, Dtype dtype = Dtype::kF32,
           DeviceLocation device = DeviceLocation::kHost) {
        size_t sh[kMaxRank];
        size_t r = 0;
        for (auto s : shape) {
            if (r >= kMaxRank) break;
            sh[r++] = s;
        }
        tensor_ = tb_tensor_create(sh, (uint8_t)r, static_cast<TbDtype>(dtype),
                                    static_cast<TbDevice>(device));
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : tensor_(other.tensor_) {
        other.tensor_ = tb_tensor_empty();
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            tb_tensor_free(&tensor_);
            tensor_ = other.tensor_;
            other.tensor_ = tb_tensor_empty();
        }
        return *this;
    }

    ~Tensor() { tb_tensor_free(&tensor_); }

    /* Construct from raw C tensor (takes ownership) */
    explicit Tensor(TbTensor t) : tensor_(t) {}

    /* Wrap external data (no ownership) */
    static Tensor wrap(void* data, std::initializer_list<size_t> shape, Dtype dtype = Dtype::kF32,
                       DeviceLocation device = DeviceLocation::kHost) {
        size_t sh[kMaxRank];
        size_t r = 0;
        for (auto s : shape) {
            if (r >= kMaxRank) break;
            sh[r++] = s;
        }
        TbTensor t = tb_tensor_wrap(data, sh, (uint8_t)r, static_cast<TbDtype>(dtype),
                                     static_cast<TbDevice>(device));
        Tensor   out;
        out.tensor_ = t;
        out.owns_ = false;
        return out;
    }

    /* Accessors */
    void*       data() { return tensor_.data; }
    const void* data() const { return tensor_.data; }

    float*       f32() { return tb_tensor_f32(&tensor_); }
    const float* f32() const { return tb_tensor_cf32(&tensor_); }

    uint8_t*       u8() { return (uint8_t*)tensor_.data; }
    const uint8_t* u8() const { return (const uint8_t*)tensor_.data; }

    size_t rank() const { return tensor_.rank; }

    std::span<const size_t> shape() const {
        return std::span<const size_t>(tensor_.shape, tensor_.rank);
    }

    size_t shape(size_t dim) const { return dim < tensor_.rank ? tensor_.shape[dim] : 0; }

    size_t size() const { return tb_tensor_nelem(&tensor_); }
    size_t bytes() const { return tb_tensor_bytes(&tensor_); }

    Dtype          dtype() const { return static_cast<Dtype>(tensor_.dtype); }
    DeviceLocation device() const { return static_cast<DeviceLocation>(tensor_.device); }

    /* Raw C access */
    TbTensor*       c_tensor() { return &tensor_; }
    const TbTensor* c_tensor() const { return &tensor_; }

private:
    TbTensor tensor_;
    bool     owns_ = true;
};

}  // namespace run
}  // namespace tensorbit

#endif /* TENSORBIT_RUN_TENSOR_HPP */
