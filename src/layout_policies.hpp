#pragma once
#include <cstddef>   // for size_t

struct RowMajor {
    static size_t index(size_t row, size_t col, size_t /*nrows*/, size_t ncols) {
        return row * ncols + col;
    }
};

struct ColMajor {
    static size_t index(size_t row, size_t col, size_t nrows, size_t /*ncols*/) {
        return col * nrows + row;
    }
};
