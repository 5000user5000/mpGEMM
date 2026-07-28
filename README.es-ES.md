# LUT-based Mixed-Precision GEMM (mpGEMM)

## Descripción General

mpGEMM es una librería de Multiplicación de Matrices General (GEMM) de precisión mixta y alto rendimiento, optimizada para despliegues de IA en sistemas embebidos y con recursos restringidos. Aprovecha Tablas de Búsqueda (LUTs) precomputadas para acelerar la multiplicación de matrices de precisión mixta entre enteros de pocos bits (INT4) y FP16, mejorando significativamente la velocidad de inferencia y reduciendo la carga computacional.

## Características Principales

* **Computación de Precisión Mixta**: Soporta pesos cuantizados en INT4 combinados con matrices de activación en FP16.
* **Optimización mediante Tablas de Búsqueda (LUT)**: Reemplaza la descuantización en tiempo de ejecución con búsquedas en LUT, reduciendo drásticamente la complejidad computacional.
* **Soporte para Múltiples Backends**:

  * GEMM Naive (INT y FP32)
  * GEMM LUT optimizado con SIMD (AVX2)
  * GEMM optimizado con Intel MKL
* **Post-procesamiento**: Proporciona adición de bias y funciones de activación (ReLU, Sigmoid, Tanh, Linear).
* **Herramientas de Benchmarking**: Incluye herramientas para la medición de latencia a través de diferentes tamaños de matriz y backends computacionales.
* **Utilidades de Cuantización**: Funciones para cuantización/descuantización INT4.
* **Integración de API de Python**: Integra a la perfección un backend en C++ con Python para facilitar su uso.

## Instalación

### Requisitos Previos

* Python 3.10 o superior
* Pybind11
* Intel MKL (opcional, para computaciones FP32 aceleradas)

### Compilación y Configuración

```bash
# Instalar dependencias
sudo apt-get install python3-pybind11 intel-mkl-full

# Clonar el repositorio
git clone <repo_url>
cd mpGEMM

# Compilar el proyecto con MKL
make USE_MKL=1

## O compilar el proyecto sin MKL
make
```

## Uso

### Ejemplo de API de Python

```python
import mpgemm
import numpy as np

# === Paso 1: Inicializar motor ===
gemm = mpgemm.Engine(backend="lut")  # opciones: "lut", "naive", "mkl"

# === Paso 2: Preparar entradas ===
M, K, N = 4, 4, 4  # Tamaño pequeño para demostración
weights = np.random.randint(0, 16, (M, K), dtype=np.uint8)
activations = np.random.randn(K, N).astype(np.float16)
bias = np.random.randn(N).astype(np.float16)

# === Paso 3: Generar LUT para int4 × fp16 ===
gemm.generate_lut(bit_width=4)

# === Paso 4: Multiplicación de matrices
output = gemm.matmul(weights, activations, M=M, K=K, N=N)

# === Paso 5: Post-procesamiento opcional ===
output = gemm.add_bias(output, bias)
output = gemm.apply_activation(output, "relu")

# === Paso 6: Salida ===
print("Output shape:", output.shape)
print("Output values:\n", output)
```

Ejemplo completo: scripts/example.py

### Benchmarking

```bash
# Ejecutar benchmarks integrados
make run

# Script de benchmarking automatizado (promediando múltiples ejecuciones)
python3 scripts/benchmark.py --runs 10
```

## Estructura del Proyecto

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
│   ├── test_api.py
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

## Pruebas y Verificación

* **Pruebas de corrección**: Aseguran la precisión numérica de las operaciones matriciales.
* **Pruebas de benchmark**: Comparan la latencia entre los backends naive, LUT y MKL.

Ejecutar pruebas con:

```bash
make test
make pytest
```

## Referencias

* [DeepGEMM](https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Ganji_DeepGEMM.pdf)
* [T-MAC](https://arxiv.org/html/2407.00088v1)
