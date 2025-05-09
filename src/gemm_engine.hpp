#pragma once
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#include "layout_policies.hpp"
#include "storage_policies.hpp"
#include "matrix.hpp"
#include "matrix_ops.hpp"
#include "lut_utils.hpp"
#include "post_processing.hpp"

enum class Backend {
    Naive,
    LUT
#ifdef USE_MKL
  , MKL
#endif
};

class Engine {
public:
    // 构造：支持 "naive"/"lut" (和 "mkl")
    Engine(const std::string &backend_str)
      : lut(nullptr)
    {
        if      (backend_str == "naive")   backend = Backend::Naive;
        else if (backend_str == "lut")         backend = Backend::LUT;
#ifdef USE_MKL
        else if (backend_str == "mkl")         backend = Backend::MKL;
#endif
        else throw std::invalid_argument("Unknown backend: " + backend_str);
    }

    // 仅在 LUT 模式下调用，bit_width 一般传 4
    void generate_lut(int bit_width) {
        if (backend != Backend::LUT)
            throw std::runtime_error("generate_lut only valid for LUT backend");
        if(bit_width!=4)
            throw std::invalid_argument("LUT currently only supports bit_width=4");
        int range = 1 << bit_width;
        lut = std::make_unique<ProductLookupTable<uint8_t,uint8_t,int32_t>>(range, range);
    }

    // matmul: Wflat 是 uint8_t(int4 存 raw)，Aflat 是 float(list)
    std::vector<float> matmul(
        const std::vector<uint8_t>& Wflat,
        const std::vector<float>&   Aflat,
        int M, int K, int N) const
    {
        std::vector<float> out;
        out.reserve(size_t(M) * N);

        switch (backend) {
        case Backend::Naive: {
            Matrix<int,RowMajor,PlainStorage<int>> Wi(M, K), Ai(K, N);
            // 填充 Wi, Ai，將 uint8_t 轉換為有符號 int
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < K; ++j) {
                    int val = Wflat[size_t(i)*K + j];
                    Wi.set(i, j, val < 8 ? val : val - 16);  // 轉換為有符號
                }
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < N; ++j)
                    Ai.set(i, j, (int)std::lround(Aflat[size_t(i)*N + j]));
            // 调用全局 ::matmul
            auto C = ::matmul(Wi, Ai);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    out.push_back(float(C.at(i,j)));
            break;
        }
        case Backend::LUT: {
            if (!lut) throw std::runtime_error("LUT not generated");
            // 从 raw uint8 构造 Int4Storage 矩阵并 unpack
            Matrix<uint8_t,RowMajor,Int4Storage> Wq(M,K);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < K; ++j)
                    Wq.set(i,j, Wflat[size_t(i)*K + j]);
            auto Wu = unpack_int4(Wq);
            // 把 Activation floats 截断、cast 为 uint8 索引，並轉換為有符號
            std::vector<uint8_t> Au(size_t(K)*N);
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < N; ++j) {
                    float val = Aflat[size_t(i)*N + j];
                    int q = std::lround(val);
                    q = std::clamp(q, -8, 7);  // 限制在有符號 int4 範圍
                    Au[size_t(i)*N + j] = uint8_t(q < 0 ? q + 16 : q);
                }
            auto Ci = matmul_lut_fast(Wu, Au, M, K, N, *lut);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    out.push_back(float(Ci.at(i,j)));
            break;
        }
#ifdef USE_MKL
        case Backend::MKL: {
            Matrix<float,RowMajor,PlainStorage<float>> Wf(M,K), Af(K,N);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < K; ++j)
                    Wf.set(i,j, float(Wflat[size_t(i)*K + j]));
            for (int i = 0; i < K; ++i)
                for (int j = 0; j < N; ++j)
                    Af.set(i,j, Aflat[size_t(i)*N + j]);
            auto C = matmul_mkl(Wf, Af);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    out.push_back(C.at(i,j));
            break;
        }
#endif
        default:
            throw std::runtime_error("Unsupported backend");
        }

        return out;
    }

    // Bias addition
    std::vector<float> add_bias(
        const std::vector<float>& Cflat,
        int M, int N,
        const std::vector<float>& bias) const
    {
        Matrix<float,RowMajor,PlainStorage<float>> C(M,N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C.set(i, j, Cflat[size_t(i)*N + j]);
        auto R = ::add_bias(C, bias);
        std::vector<float> out; out.reserve(size_t(M)*N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                out.push_back(R.at(i,j));
        return out;
    }

    // Activation
    std::vector<float> apply_activation(
        const std::vector<float>& Cflat,
        int M, int N,
        Activation act) const
    {
        Matrix<float,RowMajor,PlainStorage<float>> C(M,N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C.set(i, j, Cflat[size_t(i)*N + j]);
        auto R = ::apply_activation(C, act);
        std::vector<float> out; out.reserve(size_t(M)*N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                out.push_back(R.at(i,j));
        return out;
    }

private:
    Backend backend;
    std::unique_ptr<ProductLookupTable<uint8_t,uint8_t,int32_t>> lut;
};

