#pragma once
#include "matrix.hpp"
#include "layout_policies.hpp"
#include "storage_policies.hpp"
#include "lut_utils.hpp"
#include <type_traits>
#include <vector>
#include <immintrin.h>
#include <thread>
#include <mutex>
#include <iostream>

// =============================================================
//  Helper: unpack a Matrix<> that uses Int4Storage into a
//  contiguous std::vector<uint8_t> (each element 0‥15).
//  Call once before the GEMM to avoid per‑element overhead.
// =============================================================

template<typename Mat4>
std::vector<uint8_t> unpack_int4(const Mat4& M)
{
    static_assert(std::is_same_v<typename Mat4::StorageType, uint8_t>,
                  "Matrix must use Int4Storage underlying uint8 byte");

    const size_t R = M.rows();
    const size_t C = M.cols();
    std::vector<uint8_t> out(R * C);
    for (size_t r = 0; r < R; ++r)
        for (size_t c = 0; c < C; ++c)
            out[r * C + c] = M.at(r, c);
    return out;
}

// =============================================================
//  High-performance parallel GEMM implementation
//  Supports any numeric type through templates
// =============================================================

template<typename MA, typename MB>
auto matmul(const MA& A, const MB& B, size_t num_threads = 4)
{
    using T = decltype(A.at(0, 0));
    static_assert(std::is_same_v<T, decltype(B.at(0, 0))>, "Element types must match");

    size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix<T, RowMajor, PlainStorage<T>> C(M, N);
    std::vector<std::thread> threads;

    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, M);

            Matrix<T, RowMajor, PlainStorage<T>> local_C(M, N);

            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    T a = A.at(i, k);
                    for (size_t j = 0; j < N; ++j) {
                        local_C.set(i, j, local_C.at(i, j) + a * B.at(k, j));
                    }
                }
            }

            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C.set(i, j, C.at(i, j) + local_C.at(i, j));
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return C;
}

// =============================================================
//  MKL GEMM (Intel MKL) — expects packed row-major matrices
//  * A shape: M × K  contiguous
//  * B shape: K × N  contiguous
//  * C shape: M × N  contiguous
//  * C is overwritten with the result
// =============================================================


#ifdef USE_MKL
#include <mkl.h>

template<typename T>
Matrix<T> matmul_mkl(const Matrix<T>& A, const Matrix<T>& B) {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                  "MKL only supports float/double GEMM with this wrapper");

    Matrix<T> C(A.rows(), B.cols());

    const CBLAS_LAYOUT layout = CblasRowMajor;
    const CBLAS_TRANSPOSE trans = CblasNoTrans;

    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(layout, trans, trans,
                    A.rows(), B.cols(), A.cols(),
                    1.0f,
                    A.data(), A.cols(),
                    B.data(), B.cols(),
                    0.0f,
                    C.data(), C.cols());
    } else { // double
        cblas_dgemm(layout, trans, trans,
                    A.rows(), B.cols(), A.cols(),
                    1.0,
                    A.data(), A.cols(),
                    B.data(), B.cols(),
                    0.0,
                    C.data(), C.cols());
    }
    return C;
}
#endif

// =============================================================
//  High‑speed LUT GEMM — expects *unpacked* uint8 buffers.
//  * Au shape: M × K  contiguous
//  * Bu shape: K × N  contiguous
//  Works with or without AVX2 (scalar fallback).
// =============================================================

// LUT-based mixed-precision GEMM kernel
template <typename A>
auto matmul_lut_fast(const std::vector<uint8_t>& W,
                     const std::vector<A>& A_mat,
                     size_t M, size_t K, size_t N,
                     ProductLookupTable<uint8_t, A, int32_t>& lut,
                     size_t block_size = 64,
                     size_t num_threads = 4) {
    // Result matrix
    Matrix<int32_t, RowMajor, PlainStorage<int32_t>> C(M, N);
    std::vector<std::thread> threads;
    std::mutex mtx;
    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t row_start = t * rows_per_thread;
            size_t row_end = std::min(row_start + rows_per_thread, M);
            Matrix<int32_t, RowMajor, PlainStorage<int32_t>> localC(M, N);
            for (size_t i = row_start; i < row_end; i += block_size) {
                size_t i_end = std::min(i + block_size, row_end);
                for (size_t k = 0; k < K; k += block_size) {
                    size_t k_end = std::min(k + block_size, K);
                    // For each k in this block, rebuild LUT and accumulate
                    for (size_t kk = k; kk < k_end; ++kk) {
                        const A* act_row = &A_mat[kk * N];
                        lut.fill_from_activation(act_row);
                        // Accumulate for each row i
                        for (size_t ii = i; ii < i_end; ++ii) {
                            uint8_t q = W[ii * K + kk];
                            const int32_t* lut_row = lut.get_row(q);
                            for (size_t j = 0; j < N; ++j) {
                                localC.set(ii, j, localC.at(ii, j) + lut_row[j]);
                            }
                        }
                    }
                }
            }
            // Merge into C
            std::lock_guard<std::mutex> lock(mtx);
            for (size_t ii = row_start; ii < row_end; ++ii) {
                for (size_t j = 0; j < N; ++j) {
                    C.set(ii, j, C.at(ii, j) + localC.at(ii, j));
                }
            }
        });
    }
    for (auto& thr : threads) thr.join();
    return C;
}