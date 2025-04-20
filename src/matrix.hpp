#pragma once
#include <vector>
#include <cstddef>
#include "layout_policies.hpp"
#include "storage_policies.hpp"

template<
    typename T,                              // logical element type
    typename LayoutPolicy    = RowMajor,     
    typename StoragePolicy   = PlainStorage<T>  
>
class Matrix {
public:
    using StorageType = typename StoragePolicy::StorageType;
    static constexpr size_t EPU = StoragePolicy::entries_per_unit;

    Matrix(size_t rows, size_t cols)
      : rows_(rows), cols_(cols)
    {
        size_t total_elems = rows * cols;
        size_t total_units = (total_elems + EPU - 1) / EPU;
        data_.resize(total_units);
    }

    T at(size_t r, size_t c) const {
        size_t lin = LayoutPolicy::index(r, c, rows_, cols_);
        size_t unit_idx = lin / EPU;
        size_t offset   = lin % EPU;
        return StoragePolicy::get(data_[unit_idx], offset);
    }

    void set(size_t r, size_t c, T value) {
        size_t lin = LayoutPolicy::index(r, c, rows_, cols_);
        size_t unit_idx = lin / EPU;
        size_t offset   = lin % EPU;
        StoragePolicy::set(data_[unit_idx], value, offset);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

private:
    size_t rows_, cols_;
    std::vector<StorageType> data_;
};
