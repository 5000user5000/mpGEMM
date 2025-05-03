#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "matrix.hpp"
#include "post_processing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mpgemm, m) {
    m.doc() = "mpGEMM Python bindings";

    // Activation enum
    py::enum_<Activation>(m, "Activation")
        .value("Linear",  Activation::Linear)
        .value("ReLU",    Activation::ReLU)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Tanh",    Activation::Tanh)
        .export_values();

    // add_bias(C_flat, M, N, bias)
    m.def("add_bias",
        [](const std::vector<float>& flatC,
           int M, int N,
           const std::vector<float>& bias)
    {
        // 1) Wrap flatC into Matrix
        Matrix<float, RowMajor, PlainStorage<float>> C(M, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C.set(i, j, flatC[i * N + j]);

        // 2) Call C++ add_bias
        auto Rmat = add_bias(C, bias);

        // 3) Flatten result
        std::vector<float> out;
        out.reserve(M * N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                out.push_back(Rmat.at(i, j));

        // 4) Return to Python
        return out;
    },
    py::arg("C"), py::arg("M"), py::arg("N"), py::arg("bias"));

    // apply_activation(C_flat, M, N, act)
    m.def("apply_activation",
        [](const std::vector<float>& flatC,
           int M, int N,
           Activation act)
    {
        Matrix<float, RowMajor, PlainStorage<float>> C(M, N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C.set(i, j, flatC[i * N + j]);

        auto Rmat = apply_activation(C, act);

        std::vector<float> out;
        out.reserve(M * N);
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                out.push_back(Rmat.at(i, j));

        return out;
    },
    py::arg("C"), py::arg("M"), py::arg("N"), py::arg("act"));
}
