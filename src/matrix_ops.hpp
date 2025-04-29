#pragma once
#include "matrix.hpp"
#include "layout_policies.hpp"
#include "storage_policies.hpp"
#include "lut_utils.hpp"    // for lookup_batch_avx2 and ProductLookupTable
#include <type_traits>
#include <vector>

// =============================================================
// 1. Baseline naive GEMM 
// =============================================================
template<typename MA, typename MB>
auto matmul(const MA &A, const MB &B) {
    using T = decltype(A.at(0,0));
    static_assert(std::is_same_v<T, decltype(B.at(0,0))>,
                  "Element types must match");
    size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix<T, RowMajor, PlainStorage<T>> C(M, N);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            T a = A.at(i, k);
            for (size_t j = 0; j < N; ++j) {
                C.set(i, j, C.at(i,j) + a * B.at(k, j));
            }
        }
    }
    return C;
}

// =============================================================
// 2. LUT‑SIMD GEMM (RowMajor output)
//    * 支援 uint8_t 權重 / 激活 (例：Int4Storage 解包後)
//    * 乘積累加型別固定 int32_t
// =============================================================

template< typename MA /* Matrix<uint8_t,*,Int4Storage/Plain> */,
          typename MB /* Matrix<uint8_t,*,Int4Storage/Plain> */>
auto matmul_lut(const MA &A, const MB &B,
                     const ProductLookupTable<uint8_t,uint8_t,int32_t>& lut)
{
    using ET = uint8_t;
    static_assert(std::is_same_v<ET, decltype(A.at(0,0))>);
    static_assert(std::is_same_v<ET, decltype(B.at(0,0))>);

    size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix<int32_t, RowMajor, PlainStorage<int32_t>> C(M, N);

    // --- main GEMM ---
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            size_t k = 0;
#if defined(__AVX2__)
            for (; k + 7 < K; k += 8) {
                // gather 8 條 (w,a) — 先存到小暫存陣列再載入
                alignas(32) uint8_t w8[32];
                alignas(32) uint8_t a8[32];
                for (int t=0; t<8; ++t) {
                    w8[t] = A.at(i, k + t);
                    a8[t] = B.at(k + t, j);
                }
                __m256i vw = _mm256_load_si256(reinterpret_cast<const __m256i*>(w8));
                __m256i va = _mm256_load_si256(reinterpret_cast<const __m256i*>(a8));
                acc += lookup8_sum_avx2(lut, vw, va);
            }
#endif
            // tail (scalar)
            for (; k < K; ++k) {
                acc += lut.get(A.at(i,k), B.at(k,j));
            }
            C.set(i, j, acc);
        }
    }
    return C;
}
