#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include "matrix.hpp"  // 原本的 Row_Major_Matrix / Column_Major_Matrix

class PackedInt4Matrix {
public:
    PackedInt4Matrix(int rows, int cols);

    void set(int i, int j, uint8_t val);       // val 必須是 0~15
    uint8_t get(int i, int j) const;

    void fill_random();  // 填入 0~15 的值
    void print() const;

    int num_rows() const { return rows; }
    int num_cols() const { return cols; }

    template <typename T>
    Row_Major_Matrix<T> to_row_major(float scale = 1.0f, float zero_point = 0) const;

    template <typename T>
    Column_Major_Matrix<T> to_col_major(float scale = 1.0f, float zero_point = 0) const;

private:
    int rows, cols;
    std::vector<uint8_t> data; // 每個 uint8_t 儲存 2 個 int4
};
