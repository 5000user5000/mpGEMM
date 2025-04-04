#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <iostream>




template <typename T>
class Column_Major_Matrix;  // Forward declaration

template <typename T>
class Row_Major_Matrix {
public:
    std::vector<std::vector<T>> all_row;

    Row_Major_Matrix(int rows, int cols);

    Row_Major_Matrix(const Row_Major_Matrix& other);
    Row_Major_Matrix& operator=(const Row_Major_Matrix& other);
    Row_Major_Matrix(Row_Major_Matrix&& other) noexcept;
    Row_Major_Matrix& operator=(Row_Major_Matrix&& other) noexcept;

    // Fill the matrix with random values
    void fill_random();

    void print() const;

    void set(int i, int j, T val) {
        all_row[i][j] = val;
    }

    // Getter / Setter: Access by row
    std::vector<T> getRow(int index) const;
    void setRow(int index, const std::vector<T>& row);

    // Matrix multiplication: Single-threaded
    Row_Major_Matrix operator*(const Column_Major_Matrix<T>& cm) const;

    // Matrix multiplication: Multi-threaded
    Row_Major_Matrix operator%(const Column_Major_Matrix<T>& cm) const;

    // Type conversion to Column_Major_Matrix
    operator Column_Major_Matrix<T>() const;
};

template <typename T>
class Column_Major_Matrix {
public:
    std::vector<std::vector<T>> all_column;

    Column_Major_Matrix(int rows, int cols);
    
    Column_Major_Matrix(const Column_Major_Matrix& other);
    Column_Major_Matrix& operator=(const Column_Major_Matrix& other);
    Column_Major_Matrix(Column_Major_Matrix&& other) noexcept;
    Column_Major_Matrix& operator=(Column_Major_Matrix&& other) noexcept;

    // Fill the matrix with random values
    void fill_random();

    // Print the matrix
    void print() const;

    void set(int i, int j, T val) {
        all_column[j][i] = val;
    }

    // Getter / Setter: Access by column
    std::vector<T> getColumn(int index) const;
    void setColumn(int index, const std::vector<T>& column);

    // Matrix multiplication: Single-threaded
    Column_Major_Matrix operator*(const Row_Major_Matrix<T>& rm) const;
    // Matrix multiplication: Multi-threaded (using 10 threads and printing the time taken)
    Column_Major_Matrix operator%(const Row_Major_Matrix<T>& rm) const;

    // Type conversion to Row_Major_Matrix
    operator Row_Major_Matrix<T>() const;
};

#endif // MATRIX_HPP