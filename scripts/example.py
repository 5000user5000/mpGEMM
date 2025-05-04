#!/usr/bin/env python3
import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, '..')))

import numpy as np
import mpgemm
from mpgemm import Activation


def main():
    # Matrix dimensions
    M, K, N = 128, 128, 128

    # Random number generator
    rng = np.random.default_rng(42)

    # Generate random quantized weights (INT4 range 0-15)
    weights = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    # Generate random FP16 activations
    activations = rng.standard_normal(size=(K, N)).astype(np.float16)
    # Generate random bias (FP32)
    bias = rng.standard_normal(size=N).astype(np.float32)

    # Initialize GEMM engine (LUT backend)
    gemm = mpgemm.Engine("lut")
    gemm.generate_lut(bit_width=4)

    # Flatten inputs to Python lists
    w_flat = weights.flatten().tolist()
    # cast activations to Python float list
    a_flat = activations.flatten().astype(float).tolist()
    bias_list = bias.tolist()

    # Perform matrix multiplication
    out_flat = gemm.matmul(w_flat, a_flat, M, K, N)

    # Post-processing
    out_biased = gemm.add_bias(out_flat, M, N, bias_list)
    out_relu   = gemm.apply_activation(out_biased, M, N, Activation.ReLU)

    # Reshape back to matrix
    output = np.array(out_relu, dtype=np.float32).reshape(M, N)

    # Display results
    print(f"Output shape: {output.shape}")
    print("Sample output [0,:5]:", output[0, :5])


if __name__ == "__main__":
    main()