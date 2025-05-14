#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>
#include <cstdlib>    // for posix_memalign, free
#include <memory>
#include <limits>
#include <iostream>

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
template <typename W, typename A, typename P = std::common_type_t<W, A>>
class ProductLookupTable {
public:
    using WeightType     = W;
    using ActivationType = A;
    using ProductType    = P;

    // 建構内部初始化，接受權重量化層數與 activation 長度
    ProductLookupTable(std::size_t weight_levels, std::size_t a_range)
        : weight_levels_(weight_levels),
          a_range_(a_range),
          padded_a_range_(((a_range + 7) / 8) * 8),
          table_(weight_levels * padded_a_range_)
    {
        // Default initialization: scalar LUT for weight × activation (raw) values
        for (std::size_t w = 0; w < weight_levels_; ++w) {
            int signed_w = static_cast<int>(w < (weight_levels_/2) ? w : w - weight_levels_);
            P* row_ptr = &table_[w * padded_a_range_];
            for (std::size_t a = 0; a < a_range_; ++a) {
                // Activation index a treated as raw ActivationType
                ActivationType act_val = static_cast<ActivationType>(a);
                int64_t prod = static_cast<int64_t>(signed_w) * static_cast<int64_t>(act_val);
                // Saturate to ProductType range
                if (prod > std::numeric_limits<P>::max()) prod = std::numeric_limits<P>::max();
                else if (prod < std::numeric_limits<P>::min()) prod = std::numeric_limits<P>::min();
                row_ptr[a] = static_cast<P>(prod);
            }
            // padding region left uninitialized
        }
    }

    /**
     * fill_from_activation:
     * 根據輸入的 activation row (act_row, 長度 = a_range_)，
     * 為所有 weight level 預先計算並填充查表數據：
     * 對於每個 weight w (0..weight_levels_-1)，
     *   1. 轉為有符號值 signed_w
     *   2. 遍歷所有 activation 值 act_row[a]
     *   3. 計算 signed_w * act_row[a]，並做飽和處理
     *   4. 存入 table_[w * padded_a_range_ + a]
     *
     * 這樣，後續 matmul 可直接對每個 weight lookup 整列乘積向量，
     * 而不需即時計算乘法。
     */
    void fill_from_activation(const ActivationType* act_row) {
        for (std::size_t w = 0; w < weight_levels_; ++w) {
            int signed_w = static_cast<int>(w < (weight_levels_/2) ? w : w - weight_levels_);
            P* row_ptr = &table_[w * padded_a_range_];
            for (std::size_t a = 0; a < a_range_; ++a) {
                // 讀取 activation 值；若是 uint8_t，則認為是 quantized 4-bit 需做 signed 轉換
                P act_val;
                if constexpr (std::is_same<ActivationType, uint8_t>::value) {
                    uint8_t raw = act_row[a];
                    // two's complement mapping for 4-bit
                   act_val = static_cast<P>( raw < (weight_levels_/2) ? raw : raw - weight_levels_ );
               } else {
                    act_val = static_cast<P>(act_row[a]);
                }
                // 計算乘積

               int64_t prod_int = static_cast<int64_t>(signed_w) * static_cast<int64_t>(act_val);
                // 飽和處理
                if (prod_int > std::numeric_limits<P>::max()) prod_int = std::numeric_limits<P>::max();
                else if (prod_int < std::numeric_limits<P>::min()) prod_int = std::numeric_limits<P>::min();
                row_ptr[a] = static_cast<P>(prod_int);
            }
            // padding 部分不必初始化
        }
    }

    // 取得 weight-level w 的向量查表指標
    inline const P* get_row(std::size_t w) const noexcept {
        return &table_[w * padded_a_range_];
    }

    // 元素查表，保留舊接口
    inline ProductType get(std::size_t w, std::size_t a) const noexcept {
        return table_[w * padded_a_range_ + a];
    }
    inline ProductType operator()(std::size_t w, std::size_t a) const noexcept {
        return get(w, a);
    }

    // table 資料屬性
    const P* data() const noexcept { return table_.data(); }
    std::size_t row_stride() const noexcept { return padded_a_range_; }
    std::size_t weight_levels() const noexcept { return weight_levels_; }
    std::size_t activation_range() const noexcept { return a_range_; }
    std::size_t lut_size_bytes() const noexcept { return table_.size() * sizeof(P); }

private:
    std::size_t weight_levels_, a_range_, padded_a_range_;
    std::vector<P, AlignedAllocator<P, 64>> table_;  // flat buffer (w * padded + a)
};
