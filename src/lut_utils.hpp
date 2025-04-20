#pragma once
#include <vector>
#include <cstddef>
#include <type_traits>

// ProductLookupTable: precomputes multiplication results for two ranges
// W: weight input type, A: activation input type, P: product type (common type)
template<typename W, typename A,
         typename P = std::common_type_t<W, A>>
class ProductLookupTable {
public:
    using WeightType     = W;
    using ActivationType = A;
    using ProductType    = P;

    // Construct a lookup table of size [w_range][a_range]
    ProductLookupTable(std::size_t w_range, std::size_t a_range) {
        table_.assign(w_range, std::vector<ProductType>(a_range));
        for (std::size_t i = 0; i < w_range; ++i) {
            for (std::size_t j = 0; j < a_range; ++j) {
                table_[i][j] = static_cast<ProductType>(i) * static_cast<ProductType>(j);
            }
        }
    }

    // Retrieve precomputed product for weight w and activation a
    ProductType get(WeightType w, ActivationType a) const {
        return table_[static_cast<std::size_t>(w)]
                     [static_cast<std::size_t>(a)];
    }

    std::size_t weight_range()     const { return table_.size(); }
    std::size_t activation_range() const { return table_.empty() ? 0 : table_[0].size(); }

private:
    std::vector<std::vector<ProductType>> table_;
};
