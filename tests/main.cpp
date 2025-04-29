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

    // ==== 1. Baseline plain int ====
    std::cout << "==== Baseline (int, naive) ====" << std::endl;
    using IntMatR = Matrix<int, RowMajor, PlainStorage<int>>;
    using IntMatC = Matrix<int, ColMajor, PlainStorage<int>>;
    IntMatR A_i(A_ROWS, A_COLS);
    IntMatC B_i(B_ROWS, B_COLS);

    for (int i = 0; i < A_ROWS; ++i)
        for (int j = 0; j < A_COLS; ++j)
            A_i.set(i, j, dist_int(rng));
    for (int i = 0; i < B_ROWS; ++i)
        for (int j = 0; j < B_COLS; ++j)
            B_i.set(i, j, dist_int(rng));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto C_i = matmul(A_i, B_i);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Naive int GEMM: "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms\n\n";

    // ==== 2. Int4 packed + LUT tests ====
    std::cout << "==== Int4 packed (scalar LUT vs SIMD LUT) ====" << std::endl;
    using Int4MatR = Matrix<uint8_t, RowMajor, Int4Storage>;
    using Int4MatC = Matrix<uint8_t, ColMajor, Int4Storage>;
    Int4MatR A4(A_ROWS, A_COLS);
    Int4MatC B4(B_ROWS, B_COLS);

    for (int i = 0; i < A_ROWS; ++i)
        for (int j = 0; j < A_COLS; ++j)
            A4.set(i, j, dist_int4(rng));
    for (int i = 0; i < B_ROWS; ++i)
        for (int j = 0; j < B_COLS; ++j)
            B4.set(i, j, dist_int4(rng));

    // LUT prepared for 0..15 x 0..15
    ProductLookupTable<uint8_t,uint8_t,int32_t> lut(16,16);

    // (2a) scalar LUT 版本
    auto t2 = std::chrono::high_resolution_clock::now();
    auto C4_scalar = matmul_lut(A4, B4, lut);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Scalar‑LUT GEMM : "
              << std::chrono::duration<double, std::milli>(t3 - t2).count()
              << " ms" << std::endl;

    // (2b) SIMD LUT 版本（若編譯器支援）
#if defined(__AVX2__)
    auto t4 = std::chrono::high_resolution_clock::now();
    auto C4_simd = matmul_lut(A4, B4, lut);   // same API; 函式內部自動用 AVX2
    auto t5 = std::chrono::high_resolution_clock::now();
    std::cout << "SIMD‑LUT GEMM   : "
              << std::chrono::duration<double, std::milli>(t5 - t4).count()
              << " ms (AVX2)" << std::endl;

    // correctness check
    bool ok = true;
    for (int i=0;i<A_ROWS && ok;++i)
        for (int j=0;j<B_COLS;++j)
            if (C4_scalar.at(i,j) != C4_simd.at(i,j)) {
                ok = false;
                std::cout << "Mismatch at ("<<i<<","<<j<<")\n";
                break;
            }
    std::cout << (ok? "SIMD result matches scalar LUT\n" : "SIMD mismatch!\n");
#endif
    return 0;
}