#pragma once
#include "matrix.hpp"
#include "layout_policies.hpp"
#include "storage_policies.hpp"
#include "lut_utils.hpp"
#include <type_traits>
#include <vector>
#include <immintrin.h>

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
//  Naive reference GEMM (kept unchanged for correctness checks)
// =============================================================

template<typename MA, typename MB>
auto matmul(const MA& A, const MB& B)
{
    using T = decltype(A.at(0, 0));
    static_assert(std::is_same_v<T, decltype(B.at(0, 0))>, "Element types must match");

    size_t M = A.rows(), K = A.cols(), N = B.cols();
    Matrix<T, RowMajor, PlainStorage<T>> C(M, N);

    for (size_t i = 0; i < M; ++i)
        for (size_t k = 0; k < K; ++k) {
            T a = A.at(i, k);
            for (size_t j = 0; j < N; ++j)
                C.set(i, j, C.at(i, j) + a * B.at(k, j));
        }
    return C;
}

// =============================================================
//  High‑speed LUT GEMM — expects *unpacked* uint8 buffers.
//  * Au shape: M × K  contiguous
//  * Bu shape: K × N  contiguous
//  Works with or without AVX2 (scalar fallback).
// =============================================================

auto matmul_lut_fast(const std::vector<uint8_t>& Au,
                      const std::vector<uint8_t>& Bu,
                      size_t M, size_t K, size_t N,
                      const ProductLookupTable<uint8_t, uint8_t, int32_t>& lut)
{
    const int32_t* lut_ptr = lut.data();
    const int32_t  stride  = static_cast<int32_t>(lut.row_stride());

#if defined(__AVX2__)
    const __m256i vstride = _mm256_set1_epi32(stride);
#endif

    Matrix<int32_t, RowMajor, PlainStorage<int32_t>> C(M, N);

    for (size_t i = 0; i < M; ++i) {
        const uint8_t* rowA = &Au[i * K];
        for (size_t j = 0; j < N; ++j) {
            int32_t acc = 0;
            size_t k = 0;

#if defined(__AVX2__)
            // --- AVX2: process 8 elements (k dimension) per iteration ---
            for (; k + 7 < K; k += 8) {
                // load 8 uint8 from A row / B column (row‑major & col‑major buffers)
                __m128i w8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(rowA + k));
                __m128i a8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&Bu[k * N + j]));

                __m256i w32 = _mm256_cvtepu8_epi32(w8);
                __m256i a32 = _mm256_cvtepu8_epi32(a8);
                __m256i idx = _mm256_add_epi32(_mm256_mullo_epi32(w32, vstride), a32);
                __m256i vals = _mm256_i32gather_epi32(lut_ptr, idx, 4);

                // horizontal sum of 8 lanes
                __m128i low  = _mm256_castsi256_si128(vals);
                __m128i high = _mm256_extracti128_si256(vals, 1);
                __m128i sum  = _mm_add_epi32(low, high);
                sum = _mm_hadd_epi32(sum, sum);
                sum = _mm_hadd_epi32(sum, sum);
                acc += _mm_cvtsi128_si32(sum);
            }
#endif
            // --- scalar remainder (or full loop if no AVX2) ---
            for (; k < K; ++k)
                acc += lut_ptr[rowA[k] * stride + Bu[k * N + j]];

            C.set(i, j, acc);
        }
    }
    return C;
}
