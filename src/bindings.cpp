#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "matrix.hpp"
#include "lut_utils.hpp"
#include "post_processing.hpp"
#include "gemm_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mpgemm, m) {
    m.doc() = "mpGEMM Python bindings";

    // --- Activation enum ---
    py::enum_<Activation>(m, "Activation")
        .value("Linear",  Activation::Linear)
        .value("ReLU",    Activation::ReLU)
        .value("Sigmoid", Activation::Sigmoid)
        .value("Tanh",    Activation::Tanh)
        .export_values();

    // --- Free functions ---
    m.def("add_bias",
        [](const std::vector<float>& flatC,
           int M, int N,
           const std::vector<float>& bias) {
            Matrix<float, RowMajor, PlainStorage<float>> C(M, N);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    C.set(i, j, flatC[i * N + j]);
            auto R = add_bias(C, bias);
            std::vector<float> out;
            out.reserve(M * N);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    out.push_back(R.at(i, j));
            return out;
        },
        py::arg("C"), py::arg("M"), py::arg("N"), py::arg("bias"));

    m.def("apply_activation",
        [](const std::vector<float>& flatC,
           int M, int N,
           Activation act) {
            Matrix<float, RowMajor, PlainStorage<float>> C(M, N);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    C.set(i, j, flatC[i * N + j]);
            auto R = apply_activation(C, act);
            std::vector<float> out;
            out.reserve(M * N);
            for (int i = 0; i < M; ++i)
                for (int j = 0; j < N; ++j)
                    out.push_back(R.at(i, j));
            return out;
        },
        py::arg("C"), py::arg("M"), py::arg("N"), py::arg("act"));

    // --- Engine class ---
    py::class_<Engine>(m, "Engine")
        .def(py::init<const std::string&>(), py::arg("backend"))
        .def("generate_lut", &Engine::generate_lut,
             "Generate LUT for INT4 backend", py::arg("bit_width"))
        .def("matmul", &Engine::matmul,
             "Perform GEMM with chosen backend",
             py::arg("weights"), py::arg("activations"),
             py::arg("M"), py::arg("K"), py::arg("N"))
        .def("add_bias", &Engine::add_bias,
             "Add bias vector to GEMM output",
             py::arg("C"), py::arg("M"), py::arg("N"), py::arg("bias"))
        .def("apply_activation", &Engine::apply_activation,
             "Apply activation to GEMM output",
             py::arg("C"), py::arg("M"), py::arg("N"), py::arg("act"));
}