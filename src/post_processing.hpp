#pragma once
#include <vector>
#include <cmath>
#include "matrix.hpp"

/// 後處理可選激活函式
enum class Activation {
    Linear,
    ReLU,
    Sigmoid,
    Tanh
};

/// 1) bias 加法：對於每一列，將 bias[j] 加到 M(i,j)
template<typename T, typename Layout, typename Storage>
Matrix<T,Layout,Storage> add_bias(
    const Matrix<T,Layout,Storage>& M,
    const std::vector<T>& bias)
{
    size_t R = M.rows(), C = M.cols();
    Matrix<T,Layout,Storage> Rmat(R, C);
    for(size_t i = 0; i < R; ++i) {
        for(size_t j = 0; j < C; ++j) {
            Rmat.set(i, j, M.at(i,j) + bias[j]);
        }
    }
    return Rmat;
}

/// 2) element-wise activation
template<typename T, typename Layout, typename Storage>
Matrix<T,Layout,Storage> apply_activation(
    const Matrix<T,Layout,Storage>& M,
    Activation act)
{
    size_t R = M.rows(), C = M.cols();
    Matrix<T,Layout,Storage> Rmat(R, C);
    for(size_t i = 0; i < R; ++i) {
        for(size_t j = 0; j < C; ++j) {
            T v = M.at(i,j);
            switch(act) {
              case Activation::ReLU:    v = v>static_cast<T>(0)?v:static_cast<T>(0); break;
              case Activation::Sigmoid: v = static_cast<T>(1) / (static_cast<T>(1)+std::exp(-v)); break;
              case Activation::Tanh:    v = std::tanh(v); break;
              case Activation::Linear:  /* no-op */       break;
            }
            Rmat.set(i, j, v);
        }
    }
    return Rmat;
}
