#include "../src/layout_policies.hpp"
#include "../src/storage_policies.hpp"
#include "../src/matrix.hpp"
#include "../src/matrix_ops.hpp"
#include "../src/lut_utils.hpp"

#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <cstdlib>

int main(int argc, char** argv) {
    // 默认参数
    int M = 500, K = 600, N = 500;
    bool run_naive_int   = true;
    bool run_naive_float = true;
    bool run_lut         = true;
    #ifdef USE_MKL
        bool run_mkl = true;    // 默认为 true
    #else
        bool run_mkl = false;
    #endif

    // 解析命令行
    for (int i = 1; i < argc; ++i) {
        if      (strcmp(argv[i], "--m")==0)          M = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--k")==0)          K = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--n")==0)          N = std::atoi(argv[++i]);
        else if (strcmp(argv[i], "--naive-only")==0) { run_naive_int = run_naive_float = run_lut = false; }
        else if (strcmp(argv[i], "--lut-only")==0)   { run_lut = true; run_naive_int = run_naive_float = run_mkl = false; }
        else if (strcmp(argv[i], "--mkl-only")==0)   { run_mkl = true; run_naive_int = run_naive_float = run_lut = false; }
    }

    std::cout << "[Shape] M=" << M << ", K=" << K << ", N=" << N << "\n\n";

    // 随机数生成
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist_int(0, 100);

    // 准备基准数据（int）
    Matrix<int,RowMajor,PlainStorage<int>> A_i(M, K), B_i(K, N);
    for (int i=0; i<M; ++i) for (int k=0; k<K; ++k) A_i.set(i,k, dist_int(rng));
    for (int k=0; k<K; ++k) for (int j=0; j<N; ++j) B_i.set(k,j, dist_int(rng));

    // 准备 LUT unpack
    // 先构造 Int4Storage 矩阵
    using Int4R = Matrix<uint8_t, RowMajor, Int4Storage>;
    using Int4C = Matrix<uint8_t, ColMajor, Int4Storage>;
    Int4R A4(M, K);
    Int4C B4(K, N);
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A4.set(i, k, static_cast<uint8_t>(A_i.at(i, k) & 0x0F));
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B4.set(k, j, static_cast<uint8_t>(B_i.at(k, j) & 0x0F));

    // 再 unpack
    auto Au = unpack_int4(A4);
    auto Bu = unpack_int4(B4);
    ProductLookupTable<uint8_t,uint8_t,int32_t> lut(16,16);

    // === Naive int ===
    if (run_naive_int) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto C = matmul(A_i, B_i);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << "[ naive_int ] Time: " << ms << " ms\n";
    }

    // === Naive float ===
    if (run_naive_float) {
        Matrix<float,RowMajor,PlainStorage<float>> A_f(M,K), B_f(K,N);
        for (int i=0; i<M; ++i)
            for (int k=0; k<K; ++k)
                A_f.set(i,k, static_cast<float>(A_i.at(i,k)));
        for (int k=0; k<K; ++k)
            for (int j=0; j<N; ++j)
                B_f.set(k,j, static_cast<float>(B_i.at(k,j)));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto C = matmul(A_f, B_f);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << "[naive_float] Time: " << ms << " ms\n";
    }

    // === Int4 LUT ===
    if (run_lut) {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto C = matmul_lut_fast(Au, Bu, M, K, N, lut);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << "[    LUT    ] Time: " << ms << " ms\n";
    }

#ifdef USE_MKL
    // === MKL float ===
    if (run_mkl) {
        Matrix<float,RowMajor,PlainStorage<float>> A_f(M,K), B_f(K,N);
        for (int i=0; i<M; ++i)
            for (int k=0; k<K; ++k)
                A_f.set(i,k, static_cast<float>(A_i.at(i,k)));
        for (int k=0; k<K; ++k)
            for (int j=0; j<N; ++j)
                B_f.set(k,j, static_cast<float>(B_i.at(k,j)));

        auto t0 = std::chrono::high_resolution_clock::now();
        auto C = matmul_mkl(A_f, B_f);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        std::cout << "[    MKL    ] Time: " << ms << " ms\n";
    }
#endif

    return 0;
}