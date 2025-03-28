#include "matrix.hpp"
#include <iostream>
#include <random>
#include <thread>
#include <chrono>
#include <stdexcept>

// =========================== Row_Major_Matrix Implementation ===========================

template <typename T>
Row_Major_Matrix<T>::Row_Major_Matrix(int rows, int cols)
    : all_row(rows, std::vector<T>(cols)) {
    fill_random();
}

template <typename T>
Row_Major_Matrix<T>::Row_Major_Matrix(const Row_Major_Matrix& other)
    : all_row(other.all_row) {}

template <typename T>
Row_Major_Matrix<T>& Row_Major_Matrix<T>::operator=(const Row_Major_Matrix& other) {
    if (this != &other)
        all_row = other.all_row;
    return *this;
}

template <typename T>
Row_Major_Matrix<T>::Row_Major_Matrix(Row_Major_Matrix&& other) noexcept
    : all_row(std::move(other.all_row)) {}

template <typename T>
Row_Major_Matrix<T>& Row_Major_Matrix<T>::operator=(Row_Major_Matrix&& other) noexcept {
    if (this != &other)
        all_row = std::move(other.all_row);
    return *this;
}

template <typename T>
void Row_Major_Matrix<T>::fill_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(1, 100);
    for (auto &row : all_row)
        for (auto &val : row)
            val = dist(gen);
}

template <typename T>
void Row_Major_Matrix<T>::print() const {
    for (const auto &row : all_row) {
        for (const auto &val : row)
            std::cout << val << " ";
        std::cout << "\n";
    }
}

template <typename T>
std::vector<T> Row_Major_Matrix<T>::getRow(int index) const {
    if (index >= 0 && index < static_cast<int>(all_row.size()))
        return all_row[index];
    else
        throw std::out_of_range("Row index out of range");
}

template <typename T>
void Row_Major_Matrix<T>::setRow(int index, const std::vector<T>& row) {
    if (index >= 0 && index < static_cast<int>(all_row.size()))
        all_row[index] = row;
    else
        throw std::out_of_range("Row index out of range");
}

// Single-threaded Matrix multiplication：Row_Major_Matrix * Column_Major_Matrix
template <typename T>
Row_Major_Matrix<T> Row_Major_Matrix<T>::operator*(const Column_Major_Matrix<T>& cm) const {
    int rows = all_row.size();
    int common = (rows > 0 ? all_row[0].size() : 0);
    int cols = cm.all_column.size();
    Row_Major_Matrix<T> result(rows, cols);

    //check input
    if(common == 0 || cm.all_column.empty() || all_row.empty() || cm.all_column[0].empty())
        throw std::runtime_error("Empty matrix");
    if(common != cm.all_column[0].size())
        throw std::runtime_error("Dimension mismatch for multiplication");

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            T sum = 0;
            for (int k = 0; k < common; ++k)
                sum += all_row[i][k] * cm.all_column[j][k]; // notice: cm.all_column[j][k] instead of cm.all_column[k][j] in Column_Major_Matrix
            result.all_row[i][j] = sum;
        }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Single-threaded Row_Major multiplication took " << duration.count() << " ms" << std::endl;
    return result;
}

// Multi-threaded Matrix multiplication：using 10 threads and print the time taken
template <typename T>
Row_Major_Matrix<T> Row_Major_Matrix<T>::operator%(const Column_Major_Matrix<T>& cm) const {
    int rows = all_row.size();
    int common = (rows > 0 ? all_row[0].size() : 0);
    int cols = cm.all_column.size();
    Row_Major_Matrix<T> result(rows, cols);

    //check input
    if(common == 0 || cm.all_column.empty() || all_row.empty() || cm.all_column[0].empty())
        throw std::runtime_error("Empty matrix");
    if(common != cm.all_column[0].size())
        throw std::runtime_error("Dimension mismatch for multiplication");

    auto multiply_range = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < cols; ++j) {
                T sum = 0;
                for (int k = 0; k < common; ++k)
                    sum += all_row[i][k] * cm.all_column[j][k];
                result.all_row[i][j] = sum;
            }
        }
    };

    const int num_threads = 10;
    int step = rows / num_threads;
    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_threads; ++i) {
        int start_idx = i * step;
        int end_idx = (i == num_threads - 1) ? rows : start_idx + step;
        threads.emplace_back(multiply_range, start_idx, end_idx);
    }
    for (auto &t : threads)
        t.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Multi-threaded Row_Major multiplication took " << duration.count() << " ms" << std::endl;

    return result;
}

// Type conversion：Row_Major_Matrix to Column_Major_Matrix
template <typename T>
Row_Major_Matrix<T>::operator Column_Major_Matrix<T>() const {
    if (all_row.empty()) return Column_Major_Matrix<T>(0, 0);
    int rows = all_row.size();
    int cols = all_row[0].size();
    Column_Major_Matrix<T> cm(rows, cols);  // Initialize with zeros
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            cm.all_column[j][i] = all_row[i][j];
    return cm;
}

// =========================== Column_Major_Matrix Implementation ===========================

template <typename T>
Column_Major_Matrix<T>::Column_Major_Matrix(int rows, int cols)
    : all_column(cols, std::vector<T>(rows)) {
    fill_random();
}

template <typename T>
Column_Major_Matrix<T>::Column_Major_Matrix(const Column_Major_Matrix& other)
    : all_column(other.all_column) {}

template <typename T>
Column_Major_Matrix<T>& Column_Major_Matrix<T>::operator=(const Column_Major_Matrix& other) {
    if (this != &other)
        all_column = other.all_column;
    return *this;
}

template <typename T>
Column_Major_Matrix<T>::Column_Major_Matrix(Column_Major_Matrix&& other) noexcept
    : all_column(std::move(other.all_column)) {}

template <typename T>
Column_Major_Matrix<T>& Column_Major_Matrix<T>::operator=(Column_Major_Matrix&& other) noexcept {
    if (this != &other)
        all_column = std::move(other.all_column);
    return *this;
}

template <typename T>
void Column_Major_Matrix<T>::fill_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(1, 100);
    for (auto &col : all_column)
        for (auto &val : col)
            val = dist(gen);
}

template <typename T>
void Column_Major_Matrix<T>::print() const {
    if (all_column.empty() || all_column[0].empty())
        return;
    int rows = all_column[0].size();
    int cols = all_column.size();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::cout << all_column[j][i] << " ";
        std::cout << "\n";
    }
}

template <typename T>
std::vector<T> Column_Major_Matrix<T>::getColumn(int index) const {
    if (index >= 0 && index < static_cast<int>(all_column.size()))
        return all_column[index];
    else
        throw std::out_of_range("Column index out of range");
}

template <typename T>
void Column_Major_Matrix<T>::setColumn(int index, const std::vector<T>& column) {
    if (index >= 0 && index < static_cast<int>(all_column.size()))
        all_column[index] = column;
    else
        throw std::out_of_range("Column index out of range");
}

// Single-threaded Matrix multiplication：Column_Major_Matrix * Row_Major_Matrix
template <typename T>
Column_Major_Matrix<T> Column_Major_Matrix<T>::operator*(const Row_Major_Matrix<T>& rm) const {
    if (all_column.empty() || rm.all_row.empty())
        throw std::runtime_error("Empty matrix");
    int A_rows = all_column[0].size();
    int A_cols = all_column.size();
    int B_rows = rm.all_row.size();
    int B_cols = rm.all_row[0].size();
    if (A_cols != B_rows)
        throw std::runtime_error("Dimension mismatch for multiplication");

    Column_Major_Matrix<T> result(A_rows, B_cols);
    // Save in column-major ：result.all_column[j][i] = ∑ A.all_column[k][i] * rm.all_row[k][j]
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < B_cols; ++j) {
        for (int i = 0; i < A_rows; ++i) {
            T sum = 0;
            for (int k = 0; k < A_cols; ++k)
                sum += all_column[k][i] * rm.all_row[k][j];
            result.all_column[j][i] = sum;
        }
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Single-threaded Column_Major multiplication took " << duration.count() << " ms" << std::endl;
    return result;
}

// Multi-threaded Matrix multiplication：using 10 threads and print the time taken
template <typename T>
Column_Major_Matrix<T> Column_Major_Matrix<T>::operator%(const Row_Major_Matrix<T>& rm) const {
    if (all_column.empty() || rm.all_row.empty())
        throw std::runtime_error("Empty matrix");
    int A_rows = all_column[0].size();
    int A_cols = all_column.size();
    int B_rows = rm.all_row.size();
    int B_cols = rm.all_row[0].size();
    if (A_cols != B_rows)
        throw std::runtime_error("Dimension mismatch for multiplication");

    Column_Major_Matrix<T> result(A_rows, B_cols);

    auto multiply_range = [&](int start, int end) {
        // partition by rows of the result matrix
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < B_cols; ++j) {
                T sum = 0;
                for (int k = 0; k < A_cols; ++k)
                    sum += all_column[k][i] * rm.all_row[k][j];
                result.all_column[j][i] = sum;
            }
        }
    };

    const int num_threads = 10;
    int step = A_rows / num_threads;
    std::vector<std::thread> threads;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < num_threads; ++t) {
        int start_idx = t * step;
        int end_idx = (t == num_threads - 1) ? A_rows : start_idx + step;
        threads.emplace_back(multiply_range, start_idx, end_idx);
    }
    for (auto &th : threads)
        th.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Multi-threaded Column_Major multiplication took " << duration.count() << " ms" << std::endl;

    return result;
}

// Type Conversion：Column_Major_Matrix to Row_Major_Matrix
template <typename T>
Column_Major_Matrix<T>::operator Row_Major_Matrix<T>() const {
    if (all_column.empty()) return Row_Major_Matrix<T>(0, 0);
    int rows = all_column[0].size();
    int cols = all_column.size();
    Row_Major_Matrix<T> rm(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            rm.all_row[i][j] = all_column[j][i];
    return rm;
}

// Explicit instantiation
template class Row_Major_Matrix<int>;
template class Column_Major_Matrix<int>;