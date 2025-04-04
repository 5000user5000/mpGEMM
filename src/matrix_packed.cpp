#include "matrix_packed.hpp"
#include <random>
#include <stdexcept>

PackedInt4Matrix::PackedInt4Matrix(int rows, int cols)
    : rows(rows), cols(cols), data((rows * cols + 1) / 2, 0) {}

void PackedInt4Matrix::set(int i, int j, uint8_t val) {
    if (val > 15) throw std::invalid_argument("val must be 0~15");
    int pos = i * cols + j;
    int byte_idx = pos / 2;
    if (pos % 2 == 0)
        data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F); // set low 4 bits
    else
        data[byte_idx] = (data[byte_idx] & 0x0F) | ((val & 0x0F) << 4); // set high 4 bits
}

uint8_t PackedInt4Matrix::get(int i, int j) const {
    int pos = i * cols + j;
    int byte_idx = pos / 2;
    if (pos % 2 == 0)
        return data[byte_idx] & 0x0F;
    else
        return (data[byte_idx] >> 4) & 0x0F;
}

void PackedInt4Matrix::fill_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            set(i, j, dist(gen));
}

void PackedInt4Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::cout << static_cast<int>(get(i, j)) << " ";
        std::cout << "\n";
    }
}
