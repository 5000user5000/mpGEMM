#pragma once
#include "matrix_packed.hpp"
#include "matrix.hpp"  // 原本的 Row_Major_Matrix / Column_Major_Matrix

template<typename T>
Row_Major_Matrix<T> PackedInt4Matrix::to_row_major(float scale, float zero_point) const {
    Row_Major_Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uint8_t q = get(i, j);
            int8_t val = (q >= 8) ? q - 16 : q; // signed int4 [-8, 7]
            float real = val * scale + zero_point;
            result.set(i, j, static_cast<T>(real));
        }
    }
    return result;
}

template<typename T>
Column_Major_Matrix<T> PackedInt4Matrix::to_col_major(float scale, float zero_point) const {
    Column_Major_Matrix<T> result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            uint8_t q = get(i, j);
            int8_t val = (q >= 8) ? q - 16 : q;
            float real = val * scale + zero_point;
            result.set(i, j, static_cast<T>(real));
        }
    }
    return result;
}
