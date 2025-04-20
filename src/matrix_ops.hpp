#pragma once
#include "matrix.hpp"
#include "layout_policies.hpp"
#include "storage_policies.hpp"
#include <type_traits>

// Naive GEMM: allow different Layout/Storage for A and B.
// Output is Row-major Matrix with PlainStorage of T.
template<typename MA, typename MB>
auto matmul(const MA &A, const MB &B) {
    using T = decltype(A.at(0,0));
    static_assert(std::is_same_v<T, decltype(B.at(0,0))>, "Element types must match");
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
