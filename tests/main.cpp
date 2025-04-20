#include "../src/layout_policies.hpp"
#include "../src/storage_policies.hpp"
#include "../src/matrix.hpp"
#include "../src/matrix_ops.hpp"
#include "../src/lut_utils.hpp"

#include <iostream>
#include <chrono>
#include <random>

int main() {
    // configure matrix dimensions
    const int A_ROWS = 200, A_COLS = 300;
    const int B_ROWS = 300, B_COLS = 200;

    // prepare random generators
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist_int(0, 100);
    std::uniform_int_distribution<int> dist_int4(0, 15);

    // ==== Test with plain int ====
    std::cout << "==== Test with int ====" << std::endl;
    using IntMatR = Matrix<int, RowMajor, PlainStorage<int>>;
    using IntMatC = Matrix<int, ColMajor, PlainStorage<int>>;
    IntMatR A_i(A_ROWS, A_COLS);
    IntMatC B_i(B_ROWS, B_COLS);

    // fill A_i and B_i with random ints
    for (int i = 0; i < A_ROWS; ++i)
        for (int j = 0; j < A_COLS; ++j)
            A_i.set(i, j, dist_int(rng));
    for (int i = 0; i < B_ROWS; ++i)
        for (int j = 0; j < B_COLS; ++j)
            B_i.set(i, j, dist_int(rng));

    // time naive int GEMM
    auto t0 = std::chrono::high_resolution_clock::now();
    auto C_i = matmul(A_i, B_i);
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt_i = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "INT Process took " << dt_i << " ms\n\n";

    // ==== Test with Int4 + Elementwise LUT ====
    std::cout << "==== Test with Int4 + Elementwise LUT ====" << std::endl;
    using Int4MatR = Matrix<uint8_t, RowMajor, Int4Storage>;
    using Int4MatC = Matrix<uint8_t, ColMajor, Int4Storage>;
    Int4MatR A4(A_ROWS, A_COLS);
    Int4MatC B4(B_ROWS, B_COLS);

    // fill A4 and B4 with random Int4 values (0–15)
    for (int i = 0; i < A_ROWS; ++i)
        for (int j = 0; j < A_COLS; ++j)
            A4.set(i, j, dist_int4(rng));
    for (int i = 0; i < B_ROWS; ++i)
        for (int j = 0; j < B_COLS; ++j)
            B4.set(i, j, dist_int4(rng));

    // unpack to plain int for baseline
    IntMatR A4_u(A_ROWS, A_COLS), B4_u(B_ROWS, B_COLS);
    for (int i = 0; i < A_ROWS; ++i)
        for (int j = 0; j < A_COLS; ++j)
            A4_u.set(i, j, A4.at(i, j));
    for (int i = 0; i < B_ROWS; ++i)
        for (int j = 0; j < B_COLS; ++j)
            B4_u.set(i, j, B4.at(i, j));

    // compute naive GEMM on unpacked matrices
    auto C4_naive = matmul(A4_u, B4_u);

    // build lookup table for weight range [0,16) and activation range [0,16)
    ProductLookupTable<uint8_t, uint8_t> lut(16, 16);

    // time LUT-based GEMM directly on packed matrices
    auto t2 = std::chrono::high_resolution_clock::now();
    IntMatR C4_lut(A_ROWS, B_COLS);
    for (int i = 0; i < A_ROWS; ++i) {
        for (int j = 0; j < B_COLS; ++j) {
            int sum = 0;
            for (int k = 0; k < A_COLS; ++k) {
                uint8_t w = A4.at(i, k);
                uint8_t a = B4.at(k, j);
                sum += lut.get(w, a);
            }
            C4_lut.set(i, j, sum);
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double dt4 = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "Int4 LUT Process took " << dt4 << " ms" << std::endl;

    // verify correctness
    bool ok = true;
    for (int i = 0; i < A_ROWS && ok; ++i) {
        for (int j = 0; j < B_COLS; ++j) {
            if (C4_naive.at(i, j) != C4_lut.at(i, j)) {
                ok = false;
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << C4_naive.at(i, j) << " vs "
                          << C4_lut.at(i, j) << std::endl;
            }
        }
    }
    std::cout << (ok
                       ? "✅ LUT matches naive"
                       : "❌ LUT mismatch")
              << std::endl;

    return 0;
}
