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

private:
    std::size_t              w_range_;
    std::size_t              a_range_;
    std::vector<ProductType> table_;   // flat buffer (w * a_range + a)
};

// =============================================================
//  SIMD utilities (free functions)
//  -------------------------------------------------------------
//  ‑ lookup8_sum_avx2 : 給定 8 組 (w,a) -> 回傳乘積水平加總 (int32)
//  ‑ lookup_batch_avx2: 給定陣列 w[], a[] -> 產生對應陣列 out[]
//  若無 AVX2 則自動退化成 scalar 迴圈
// =============================================================

#if defined(__AVX2__)
#include <immintrin.h>

// ---- dot‑product helper：一次處理 8 組，並水平加總 ----
inline int32_t lookup8_sum_avx2(const ProductLookupTable<uint8_t,uint8_t,int32_t>& lut,
                               __m256i w8, __m256i a8)
{
    const int32_t* base   = lut.data();
    __m256i stride = _mm256_set1_epi32(static_cast<int>(lut.row_stride()));

    // 將 8×uint8 擴成 8×int32 (0..255)
    __m256i w32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(w8));
    __m256i a32 = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(a8));

    __m256i idx  = _mm256_add_epi32(_mm256_mullo_epi32(w32, stride), a32);
    __m256i vals = _mm256_i32gather_epi32(base, idx, sizeof(int32_t));

    // horizontal add 8 lanes -> 1 int32
    __m128i low  = _mm256_castsi256_si128(vals);
    __m128i high = _mm256_extracti128_si256(vals, 1);
    __m128i sum  = _mm_add_epi32(low, high);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

// ---- gather helper：批量填 output[len] ----
inline void lookup_batch_avx2(const ProductLookupTable<uint8_t,uint8_t,int32_t>& lut,
                              const uint8_t* w_array,
                              const uint8_t* a_array,
                              int32_t*       out_array,
                              std::size_t    len)
{
    const int32_t* base = lut.data();
    const __m256i stride = _mm256_set1_epi32(static_cast<int>(lut.row_stride()));

    std::size_t i = 0;
    for (; i + 7 < len; i += 8) {
        __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w_array + i));
        __m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a_array + i));
        __m256i w32 = _mm256_cvtepu8_epi32(w8);
        __m256i a32 = _mm256_cvtepu8_epi32(a8);
        __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(w32, stride), a32);
        __m256i vals = _mm256_i32gather_epi32(base, idx, sizeof(int32_t));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(out_array + i), vals);
    }
    // tail
    for (; i < len; ++i)
        out_array[i] = lut.get(w_array[i], a_array[i]);
}

#else  // ---- non‑AVX2 fallback ----

inline int32_t lookup8_sum_avx2(const ProductLookupTable<uint8_t,uint8_t,int32_t>& lut,
                               __m256i w8, __m256i a8)
{
    alignas(32) uint8_t wb[32], ab[32];
    _mm256_store_si256(reinterpret_cast<__m256i*>(wb), w8);
    _mm256_store_si256(reinterpret_cast<__m256i*>(ab), a8);
    int32_t s = 0;
    for (int k=0;k<8;++k) s += lut.get(wb[k], ab[k]);
    return s;
}

inline void lookup_batch_avx2(const ProductLookupTable<uint8_t,uint8_t,int32_t>& lut,
                              const uint8_t* w_array,
                              const uint8_t* a_array,
                              int32_t*       out_array,
                              std::size_t    len)
{
    for (std::size_t i=0;i<len;++i)
        out_array[i] = lut.get(w_array[i], a_array[i]);
}

#endif // __AVX2__
