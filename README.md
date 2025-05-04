# LUT-based Mixed-Precision GEMM (mpGEMM)

## Overview

mpGEMM is a high-performance mixed-precision General Matrix Multiplication 
(GEMM) library optimized for embedded and resource-constrained AI deployments. 
It leverages precomputed Lookup Tables (LUTs) to accelerate low-bit integer 
(INT4) and FP16 mixed-precision matrix multiplication, significantly improving 
inference speed and reducing computational overhead.

## Key Features

* **Mixed-Precision Computation**: Supports INT4 quantized weights combined 
with FP16 activation matrices.
* **Lookup Table (LUT) Optimization**: Replaces runtime dequantization with 
LUT lookups, greatly reducing computational complexity.
* **Multiple Backend Support**:

  * Naive GEMM (INT and FP32)
  * SIMD-optimized LUT GEMM (AVX2)
  * Intel MKL optimized GEMM
* **Post-Processing**: Provides bias addition and activation functions (ReLU, 
Sigmoid, Tanh, Linear).
* **Benchmarking Tools**: Includes tools for latency measurement across 
different matrix sizes and computational backends.
* **Quantization Utilities**: Functions for INT4 quantization/dequantization.
* **Python API Integration**: Seamlessly integrates a C++ backend with Python 
for ease of use.

## Installation

### Prerequisites

* Python 3.10 or later
* Pybind11
* Intel MKL (optional, for accelerated FP32 computations)

### Build and Setup

```bash
# Install dependencies
sudo apt-get install python3-pybind11 intel-mkl-full

# Clone the repository
git clone <repo_url>
cd mpGEMM

# Build the project with MKL
make USE_MKL=1

## Or build the project without MKL
make
```

## Usage

### Python API Example

```python
import mpgemm
import numpy as np

# === Step 1: Initialize engine ===
gemm = mpgemm.Engine(backend="lut")  # options: "lut", "naive", "mkl"

# === Step 2: Prepare inputs ===
M, K, N = 4, 4, 4  # Small size for demonstration
weights = np.random.randint(0, 16, (M, K), dtype=np.uint8)
activations = np.random.randn(K, N).astype(np.float16)
bias = np.random.randn(N).astype(np.float16)

# === Step 3: Generate LUT for int4 × fp16 ===
gemm.generate_lut(bit_width=4)

# === Step 4: Matrix multiplication
output = gemm.matmul(weights, activations, M=M, K=K, N=N)

# === Step 5: Optional post-processing ===
output = gemm.add_bias(output, bias)
output = gemm.apply_activation(output, "relu")

# === Step 6: Output ===
print("Output shape:", output.shape)
print("Output values:\n", output)
```

Full example: scripts/example.py

### Benchmarking

```bash
# Run built-in benchmarks
make run

# Automated benchmarking script (averaging multiple runs)
python3 scripts/benchmark.py --runs 10
```

## Project Structure

```
mpGEMM/
├── src/
│   ├── matrix.hpp
│   ├── matrix_ops.hpp
│   ├── layout_policies.hpp
│   ├── storage_policies.hpp
│   ├── lut_utils.hpp
│   ├── post_processing.hpp
│   ├── quant_utils.hpp
│   ├── gemm_engine.hpp
│   └── bindings.cpp
├── tests/
│   ├── test_correctness.cpp
│   ├── test_post_process.py
│   └── run_benchmark.cpp
├── scripts/
│   └── benchmark.py
├── doc/
│   └── proposal.md
├── .github/workflows/
│   └── ci.yml
├── Makefile
└── README.md
```

## Testing and Verification

* **Correctness tests**: Ensure the numerical accuracy of matrix operations.
* **Benchmark tests**: Compare latency across naive, LUT, and MKL backends.

Run tests with:

```bash
make test
make pytest
```

## References

* [DeepGEMM](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Ganji_DeepGEMM.pdf)
* [T-MAC](https://arxiv.org/html/2407.00088v1)
