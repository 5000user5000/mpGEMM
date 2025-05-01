#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>
#include <cstdlib>    // for posix_memalign, free
#include <memory>

// 對齊分配器：以 Align 對齊分配
template<typename T, std::size_t Align>
struct AlignedAllocator {
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

// lookup table for product of two integers
// LUT：64-byte 對齊，並將每列 padding 到 8 的倍數
template <typename W, typename A,
          typename P = std::common_type_t<W, A>>
class ProductLookupTable {
public:
    using WeightType     = W;
    using ActivationType = A;
    using ProductType    = P;

    ProductLookupTable(std::size_t w_range, std::size_t a_range)
    : w_range_(w_range),
        a_range_(a_range),
        padded_a_range_(((a_range + 7) / 8) * 8),
        table_(w_range * padded_a_range_)  // 用 padding 後的尺寸
    {
        for (std::size_t w = 0; w < w_range_; ++w)
            for (std::size_t a = 0; a < a_range_; ++a)
                table_[w * padded_a_range_ + a] =
                    static_cast<ProductType>(w) * static_cast<ProductType>(a);
    }

    // ---- scalar accessors ----
    inline ProductType get(W w, A a) const noexcept {
        return table_[size_t(w) * padded_a_range_ + size_t(a)];
    }
    inline ProductType operator()(W w, A a) const noexcept {
        return get(w, a);
    }

    // ---- helpers ----
    const ProductType* data()   const noexcept { return table_.data(); }
    std::size_t        row_stride() const noexcept { return padded_a_range_; }
    std::size_t        weight_range() const noexcept { return w_range_; }
    std::size_t        activation_range() const noexcept { return a_range_; }
    std::size_t lut_size_bytes() const noexcept { return table_.size() * sizeof(ProductType);}

private:
    std::size_t w_range_, a_range_, padded_a_range_;
    std::vector<ProductType, AlignedAllocator<ProductType,64>> table_;   // flat buffer (w * a_range + a)
};