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

    // 計算每個執行緒處理的行數
    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    // 為每個執行緒分配工作
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, M);

            // 為每個執行緒創建局部結果矩陣
            Matrix<T, RowMajor, PlainStorage<T>> local_C(M, N);

            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    T a = A.at(i, k);
                    for (size_t j = 0; j < N; ++j) {
                        local_C.set(i, j, local_C.at(i, j) + a * B.at(k, j));
                    }
                }
            }

            // 合併結果
            static std::mutex mtx;
            std::lock_guard<std::mutex> lock(mtx);
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C.set(i, j, C.at(i, j) + local_C.at(i, j));
                }
            }
        });
    }

    // 等待所有執行緒完成
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

auto matmul_lut_fast(const std::vector<uint8_t>& Au,
                    const std::vector<uint8_t>& Bu,
                    size_t M, size_t K, size_t N,
                    const ProductLookupTable<uint8_t, uint8_t, int32_t>& lut,
                    size_t block_size = 64,  // 增加區塊大小以減少同步開銷
                    size_t num_threads = 4)
{
    const int32_t* lut_ptr = lut.data();
    const int32_t stride = static_cast<int32_t>(lut.row_stride());
    Matrix<int32_t, RowMajor, PlainStorage<int32_t>> C(M, N);
    std::vector<std::thread> threads;
    std::mutex mtx;

    // 預取 LUT 到 L1 cache
    for (size_t i = 0; i < 16; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            _mm_prefetch(reinterpret_cast<const char*>(&lut_ptr[i * stride + j]), _MM_HINT_T0);
        }
    }

    // 計算每個執行緒處理的行數
    size_t rows_per_thread = (M + num_threads - 1) / num_threads;

    // 為每個執行緒分配工作
    for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            size_t start_row = t * rows_per_thread;
            size_t end_row = std::min(start_row + rows_per_thread, M);

            // 為每個執行緒創建局部結果矩陣
            Matrix<int32_t, RowMajor, PlainStorage<int32_t>> local_C(M, N);

            // 分塊處理
            for (size_t i = start_row; i < end_row; i += block_size) {
                size_t i_end = std::min(i + block_size, end_row);
                
                for (size_t j = 0; j < N; j += block_size) {
                    size_t j_end = std::min(j + block_size, N);
                    
                    for (size_t k = 0; k < K; k += block_size) {
                        size_t k_end = std::min(k + block_size, K);
                        
                        // 處理當前區塊
                        for (size_t ii = i; ii < i_end; ++ii) {
                            const uint8_t* rowA = &Au[ii * K];
                            
                            for (size_t jj = j; jj < j_end; ++jj) {
                                int32_t acc = 0;
                                
#if defined(__AVX2__)
                                // 使用 AVX2 處理 8 個元素
                                for (size_t kk = k; kk + 7 < k_end; kk += 8) {
                                    __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rowA + kk));
                                    __m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&Bu[kk * N + jj]));
                                    
                                    __m256i w32 = _mm256_cvtepu8_epi32(w8);
                                    __m256i a32 = _mm256_cvtepu8_epi32(a8);
                                    __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(w32, _mm256_set1_epi32(stride)), a32);
                                    
                                    // 預取下一個 LUT 值
                                    _mm_prefetch(reinterpret_cast<const char*>(&lut_ptr[_mm256_extract_epi32(idx, 0)]), _MM_HINT_T0);
                                    
                                    __m256i vals = _mm256_i32gather_epi32(lut_ptr, idx, 4);
                                    
                                    // 水平加總
                                    __m128i low = _mm256_castsi256_si128(vals);
                                    __m128i high = _mm256_extracti128_si256(vals, 1);
                                    __m128i sum = _mm_add_epi32(low, high);
                                    sum = _mm_hadd_epi32(sum, sum);
                                    sum = _mm_hadd_epi32(sum, sum);
                                    acc += _mm_cvtsi128_si32(sum);
                                }
#endif
                                // 處理剩餘元素
                                for (size_t kk = k + ((k_end - k) & ~7); kk < k_end; ++kk) {
                                    acc += lut_ptr[rowA[kk] * stride + Bu[kk * N + jj]];
                                }
                                
                                local_C.set(ii, jj, local_C.at(ii, jj) + acc);
                            }
                        }
                    }
                }
            }

            // 合併結果
            std::lock_guard<std::mutex> lock(mtx);
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < N; ++j) {
                    C.set(i, j, C.at(i, j) + local_C.at(i, j));
                }
            }
        });
    }

    // 等待所有執行緒完成
    for (auto& thread : threads) {
        thread.join();
    }

    return C;
}