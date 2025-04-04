#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <iostream>


struct Int4 {
    uint8_t value;

    Int4() : value(0) {}
    Int4(uint8_t val) : value(val & 0xF) {}

    Int4 operator+(const Int4& other) const {
        return Int4((value + other.value) & 0xF);
    }

    Int4& operator+=(const Int4& other) {
        value = (value + other.value) & 0xF;
        return *this;
    }

    Int4 operator*(const Int4& other) const {
        return Int4((value * other.value) & 0xF);
    }


    friend std::ostream& operator<<(std::ostream& os, const Int4& val) {
        os << static_cast<int>(val.value);
        return os;
    }
};



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