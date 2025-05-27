#pragma once

#include <vector>
#include <cstddef>
#include <type_traits>
#include <cstdlib>    // posix_memalign, free
#include <memory>
#include <limits>
#include <cstdint>

// Aligned allocator with compile-time check
template<typename T, std::size_t Align>
struct AlignedAllocator {
    static_assert((Align & (Align - 1)) == 0 && Align >= alignof(T),
                  "Align must be power-of-two and at least alignof(T)");

    using value_type      = T;
    using pointer         = T*;
    using const_pointer   = const T*;
    using reference       = T&;
    using const_reference = const T&;
    using size_type       = std::size_t;
    using difference_type = std::ptrdiff_t;

    template<typename U>
    struct rebind { using other = AlignedAllocator<U, Align>; };

    AlignedAllocator() noexcept {}
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    bool operator==(AlignedAllocator const&) const noexcept { return true; }
    bool operator!=(AlignedAllocator const&) const noexcept { return false; }

    pointer allocate(size_type n) {
        void* ptr = nullptr;
        if (posix_memalign(&ptr, Align, n * sizeof(T)) != 0)
            throw std::bad_alloc();
        return static_cast<pointer>(ptr);
    }

    void deallocate(pointer p, size_type) noexcept {
        free(p);
    }
};

// Lookup table for product of two integers, 64-byte aligned
// W: weight type, A: activation type, P: product type
template <typename W, typename A, typename P = std::common_type_t<W, A>>
class ProductLookupTable {
public:
    using WeightType     = W;
    using ActivationType = A;
    using ProductType    = P;

    ProductLookupTable(std::size_t weight_levels, std::size_t a_range)
        : weight_levels_(weight_levels),
          a_range_(a_range),
          padded_a_range_(((a_range + 7) / 8) * 8),
          table_(weight_levels * padded_a_range_)
    {
        // Default: build LUT with raw indices as activations
        fill_impl([&](std::size_t a) -> int64_t {
            int64_t raw = static_cast<int64_t>(a);
            // two's-complement for 4-bit (uint8_t) activations
            int64_t act = raw;
            if constexpr (std::is_same<ActivationType, uint8_t>::value) {
                act = (raw < static_cast<int64_t>(weight_levels_ / 2))
                      ? raw : (raw - static_cast<int64_t>(weight_levels_));
            }
            return act;
        });
    }

    void fill_from_activation(const ActivationType* act_row) noexcept {
        // Refill LUT with actual activation values
        fill_impl([&](std::size_t a) -> int64_t {
            int64_t raw = static_cast<int64_t>(act_row[a]);
            if constexpr (std::is_same<ActivationType, uint8_t>::value) {
                // two's-complement mapping for 4-bit
                raw = (raw < static_cast<int64_t>(weight_levels_ / 2))
                      ? raw : (raw - static_cast<int64_t>(weight_levels_));
            }
            return raw;
        });
    }

    inline const P* get_row(std::size_t w) const noexcept {
        return &table_[w * padded_a_range_];
    }
    inline ProductType get(std::size_t w, std::size_t a) const noexcept {
        return table_[w * padded_a_range_ + a];
    }
    inline ProductType operator()(std::size_t w, std::size_t a) const noexcept {
        return get(w, a);
    }

    const P* data() const noexcept { return table_.data(); }
    std::size_t row_stride() const noexcept { return padded_a_range_; }
    std::size_t weight_levels() const noexcept { return weight_levels_; }
    std::size_t activation_range() const noexcept { return a_range_; }
    std::size_t lut_size_bytes() const noexcept { return table_.size() * sizeof(P); }

private:
    std::size_t weight_levels_, a_range_, padded_a_range_;
    std::vector<P, AlignedAllocator<P, 64>> table_;

    // Multiply and saturate in product type range
    static P compute_prod(int64_t signed_w, int64_t act_val) noexcept {
        int64_t prod = signed_w * act_val;
        if (prod > std::numeric_limits<P>::max()) prod = std::numeric_limits<P>::max();
        else if (prod < std::numeric_limits<P>::min()) prod = std::numeric_limits<P>::min();
        return static_cast<P>(prod);
    }

    // Core fill logic: computes table entries by combining signed weight and activation
    template<typename GetAct>
    void fill_impl(GetAct get_act) noexcept {
        for (std::size_t w = 0; w < weight_levels_; ++w) {
            int64_t signed_w = (w < weight_levels_ / 2)
                            ? static_cast<int64_t>(w)
                            : static_cast<int64_t>(w) - static_cast<int64_t>(weight_levels_);
            P* row_ptr = &table_[w * padded_a_range_];
            for (std::size_t a = 0; a < a_range_; ++a) {
                int64_t act = get_act(a);
                row_ptr[a] = compute_prod(signed_w, act);
            }
            // padding intentionally left uninitialized for performance
        }
    }
};
