#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, '..')))

import numpy as np
import mpgemm
from mpgemm import Activation

def main():

    M, K, N = 12, 12, 12

    rng = np.random.default_rng(2025)

    # generate random int4 weights in range [-8, 7]
    weights = rng.integers(-8, 8, size=(M, K), dtype=np.int8)

    weights_unsigned = np.where(weights < 0, weights + 16, weights).astype(np.uint8)

    # generate random activations
    activations = rng.standard_normal(size=(K, N)).astype(np.float16)

    bias = rng.uniform(-1, 1, size=N).astype(np.float32)

    w_flat = weights_unsigned.flatten().tolist()
    a_flat = activations.flatten().astype(float).tolist()
    bias_list = bias.tolist()

    # === 1. Baseline: Naive GEMM ===
    gemm_ref = mpgemm.Engine("naive")
    ref_flat = gemm_ref.matmul(w_flat, a_flat, M, K, N)


    # === 2. LUT GEMM  ===
    gemm_lut = mpgemm.Engine("lut")
    gemm_lut.generate_lut(bit_width=4)
    out_flat = gemm_lut.matmul(w_flat, a_flat, M, K, N)


    # === 3. Post-processing ===
    out_biased = gemm_lut.add_bias(out_flat, M, N, bias_list)
    out_relu   = gemm_lut.apply_activation(out_biased, M, N, Activation.ReLU)

    # === 4. Output ===
    output = np.array(out_relu, dtype=np.float32).reshape(M, N)
    print(f"Output shape: {output.shape}")
    print("Sample row[0, :5]:", output[0, :5])

    # === 5. Error measurement ===
    stats = mpgemm.measure_error(ref_flat, out_flat)
    print(f"\nError relative to naive:")
    print(f"  MSE       = {stats['mse']:.6f}")
    print(f"  Max error = {stats['max_error']:.6f}")

if __name__ == "__main__":
    main()