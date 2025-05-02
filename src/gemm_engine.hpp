#include <iostream>
#include <stdexcept>
#include "matrix.hpp"


enum class Backend {
    Naive,
    LUT,
    MKL
};

struct GemmEngine {
    Backend backend;

    template<typename T>
    Matrix<T> matmul(const Matrix<T>& A, const Matrix<T>& B) const {
        switch (backend) {
        case Backend::Naive: return matmul(A, B);
        case Backend::MKL:   return matmul_mkl(A, B);
        default: throw std::runtime_error("Unsupported backend");
        }
    }
};
