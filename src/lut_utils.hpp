#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>

// =============================================================
//  ProductLookupTable  —  unified flat-storage LUT
//  -------------------------------------------------------------
//  * 只保留 **1‑D 連續記憶體 (table_)**，節省空間。
//  * 提供 get(w,a) 與 operator()(w,a) => 易用的 2‑D 介面。
//  * 內建 row‑stride() 方便 SIMD 計算線性 index。
//  * 同時暴露 data() 讓外部 SIMD kernel 直接 gather。
// =============================================================

template <typename W, typename A,
          typename P = std::common_type_t<W, A>>
class ProductLookupTable {
public:
    using WeightType     = W;
    using ActivationType = A;
    using ProductType    = P;

    ProductLookupTable(std::size_t w_range, std::size_t a_range)
        : w_range_(w_range), a_range_(a_range), table_(w_range * a_range)
    {
        for (std::size_t w = 0; w < w_range_; ++w)
            for (std::size_t a = 0; a < a_range_; ++a)
                table_[w * a_range_ + a] =
                    static_cast<ProductType>(w) * static_cast<ProductType>(a);
    }

    // ---- scalar accessors ----
    inline ProductType get(WeightType w, ActivationType a) const noexcept {
        return table_[static_cast<std::size_t>(w) * a_range_ + static_cast<std::size_t>(a)];
    }
    inline ProductType operator()(WeightType w, ActivationType a) const noexcept {
        return get(w, a);
    }

    // ---- helpers ----
    const ProductType* data()   const noexcept { return table_.data(); }
    std::size_t        row_stride() const noexcept { return a_range_; }
    std::size_t        weight_range() const noexcept { return w_range_; }
    std::size_t        activation_range() const noexcept { return a_range_; }
    std::size_t lut_size_bytes() const noexcept { return table_.size() * sizeof(ProductType);}

private:
    std::size_t              w_range_;
    std::size_t              a_range_;
    std::vector<ProductType> table_;   // flat buffer (w * a_range + a)
};