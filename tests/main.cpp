#include "../src/layout_policies.hpp"
#include "../src/storage_policies.hpp"
#include "../src/matrix.hpp"
#include "../src/matrix_ops.hpp"
#include "../src/lut_utils.hpp"

#include <iostream>
#include <chrono>
#include <random>

int main() {
    // matrix dims
    constexpr int M = 500, K = 600, N = 500;

    // rnd generators
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist_int(0, 100);

    // === baseline int (naive) ===
    std::cout << "==== Baseline (int, naive) ====\n";
    using IntR = Matrix<int, RowMajor, PlainStorage<int>>;
    using IntC = Matrix<int, ColMajor, PlainStorage<int>>;

    IntR A_i(M, K);
    IntC B_i(K, N);
    for (int i=0;i<M;++i)
        for (int k=0;k<K;++k)
            A_i.set(i,k, dist_int(rng));
    for (int k=0;k<K;++k)
        for (int j=0;j<N;++j)
            B_i.set(k,j, dist_int(rng));

    auto t0 = std::chrono::high_resolution_clock::now();
    auto C_i = matmul(A_i, B_i);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Naive int GEMM: "
              << std::chrono::duration<double,std::milli>(t1-t0).count()
              << " ms\n\n";

    // === Baseline FP32 (naive) ===
    std::cout << "==== Baseline (float, naive) ====\n";
    using FloatR = Matrix<float, RowMajor, PlainStorage<float>>;
    using FloatC = Matrix<float, ColMajor, PlainStorage<float>>;

    // 將 int baseline 轉 float
    FloatR A_f_naive(M, K);
    FloatC B_f_naive(K, N);

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A_f_naive.set(i, k, static_cast<float>(A_i.at(i, k)));

    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B_f_naive.set(k, j, static_cast<float>(B_i.at(k, j)));

    auto t6 = std::chrono::high_resolution_clock::now();
    auto C_f_naive = matmul(A_f_naive, B_f_naive);
    auto t7 = std::chrono::high_resolution_clock::now();

    std::cout << "Naive float GEMM: "
            << std::chrono::duration<double, std::milli>(t7 - t6).count()
            << " ms\n\n";


    // === Int4 packed test (derived from baseline int) ===
    std::cout << "==== Int4 packed (SIMD LUT) ====\n";
    using Int4R = Matrix<uint8_t, RowMajor, Int4Storage>;
    using Int4C = Matrix<uint8_t, ColMajor, Int4Storage>;
    Int4R A4(M, K);
    Int4C B4(K, N);

    // take lower 4‑bits of baseline int matrices so data is correlated
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k) {
            uint8_t q = static_cast<uint8_t>(A_i.at(i, k) & 0x0F);
            A4.set(i, k, q);
        }
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j) {
            uint8_t q = static_cast<uint8_t>(B_i.at(k, j) & 0x0F);
            B4.set(k, j, q);
        }

    // --- unpack once ---
    auto Au = unpack_int4(A4);
    auto Bu = unpack_int4(B4);

    ProductLookupTable<uint8_t,uint8_t,int32_t> lut(16,16);

    // lut size and shape
    std::cout << "> LUT size: " << lut.lut_size_bytes() << " bytes\n";
    std::cout << "> LUT Shape: (" << lut.weight_range()
          << ", " << lut.activation_range() << ")\n";

    // scalar (if compiled w/o AVX2) or SIMD depending on flag
    auto t2 = std::chrono::high_resolution_clock::now();
    auto C_fast = matmul_lut_fast(Au, Bu, M, K, N, lut);
    auto t3 = std::chrono::high_resolution_clock::now();

#if defined(__AVX2__)
    std::cout << "LUT GEMM (AVX2): ";
#else
    std::cout << "LUT GEMM (scalar): ";
#endif
    std::cout << std::chrono::duration<double,std::milli>(t3-t2).count()
              << " ms\n";


#ifdef USE_MKL
    // === MKL float GEMM (row‑major) ===
    std::cout << "\n==== MKL (float) ====\n";
    using F32R = Matrix<float, RowMajor, PlainStorage<float>>;

    F32R A_f(M, K);
    F32R B_f(K, N);

    // 將 int baseline 轉 float，方便對照
    for (int i=0;i<M;++i)
        for (int k=0;k<K;++k)
            A_f.set(i,k, static_cast<float>(A_i.at(i,k)));

    for (int k=0;k<K;++k)
        for (int j=0;j<N;++j)
            B_f.set(k,j, static_cast<float>(B_i.at(k,j)));

    auto t4 = std::chrono::high_resolution_clock::now();
    auto C_mkl = matmul_mkl(A_f, B_f);
    auto t5 = std::chrono::high_resolution_clock::now();

    std::cout << "MKL sgemm: "
            << std::chrono::duration<double,std::milli>(t5-t4).count()
            << " ms\n";
#endif
          

    return 0;
}