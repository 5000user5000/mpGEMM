#include "matrix_ops.hpp"
#include "matrix.hpp"
#include <vector>
#include <chrono>
#include <iostream>

void run_float() {
    const size_t M = 1024;
    const size_t K = 1024;
    const size_t N = 1024;
    
    Matrix<float, RowMajor, PlainStorage<float>> A_float(M, K);
    Matrix<float, RowMajor, PlainStorage<float>> B_float(K, N);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            A_float.set(i, j, static_cast<float>(i + j) / 1000.0f);
        }
    }
    
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B_float.set(i, j, static_cast<float>(i * j) / 1000.0f);
        }
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = matmul(A_float, B_float);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Float time: " << time << " ms\n";
}

void run_lut() {
    const size_t M = 1024;
    const size_t K = 1024;
    const size_t N = 1024;
    
    Matrix<uint8_t, RowMajor, Int4Storage> A_int4(M, K);
    Matrix<uint8_t, RowMajor, Int4Storage> B_int4(K, N);
    
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < K; ++j) {
            A_int4.set(i, j, (i + j) % 16);
        }
    }
    
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            B_int4.set(i, j, (i * j) % 16);
        }
    }
    
    ProductLookupTable<uint8_t, uint8_t, int32_t> lut(16, 16);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto result = matmul_lut_fast(unpack_int4(A_int4), unpack_int4(B_int4), M, K, N, lut);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "LUT time: " << time << " ms\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " [float|lut]\n";
        return 1;
    }
    
    std::string mode = argv[1];
    if (mode == "float") {
        run_float();
    } else if (mode == "lut") {
        run_lut();
    } else {
        std::cout << "Invalid mode. Use 'float' or 'lut'\n";
        return 1;
    }
    
    return 0;
} 