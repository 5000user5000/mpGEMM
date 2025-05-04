#!/usr/bin/env python3
import os
import sys

# 确保项目根目录在 PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, '..')))

import numpy as np
import mpgemm
from mpgemm import Activation

def main():
    # 矩阵维度
    M, K, N = 128, 128, 128

    # 随机数生成器
    rng = np.random.default_rng(2025)

    # 生成随机 INT4 权重（0-15）
    weights = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    # 生成随机 FP16 激活
    activations = rng.standard_normal(size=(K, N)).astype(np.float16)
    # 随机 bias（FP32）
    bias = rng.standard_normal(size=N).astype(np.float32)

    # 扁平化并转为 Python 列表
    w_flat = weights.flatten().tolist()
    a_flat = activations.flatten().astype(float).tolist()
    bias_list = bias.tolist()

    # === 1. 基准参考输出 ===
    gemm_ref = mpgemm.Engine("naive")
    ref_flat = gemm_ref.matmul(w_flat, a_flat, M, K, N)

    # === 2. LUT 后端输出 ===
    gemm_lut = mpgemm.Engine("lut")
    gemm_lut.generate_lut(bit_width=4)
    out_flat = gemm_lut.matmul(w_flat, a_flat, M, K, N)

    # === 3. 后处理示例 ===
    out_biased = gemm_lut.add_bias(out_flat, M, N, bias_list)
    out_relu   = gemm_lut.apply_activation(out_biased, M, N, Activation.ReLU)

    # 还原成矩阵
    output = np.array(out_relu, dtype=np.float32).reshape(M, N)
    print(f"Output shape: {output.shape}")
    print("Sample row[0, :5]:", output[0, :5])

    # === 4. 误差分析示例 ===
    stats = mpgemm.measure_error(ref_flat, out_flat)
    print(f"\nError relative to naive:")
    print(f"  MSE       = {stats['mse']:.6f}")
    print(f"  Max error = {stats['max_error']:.6f}")

if __name__ == "__main__":
    main()