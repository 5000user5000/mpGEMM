#include "../src/layout_policies.hpp"
#include "../src/storage_policies.hpp"
#include "../src/matrix.hpp"
#include "../src/matrix_ops.hpp"
#include "../src/lut_utils.hpp"
#include "../src/quant_utils.hpp"
#include "../src/post_processing.hpp"
#include "../src/accuracy_utils.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <random>
#include <cassert>

// Helper: compare two matrices for equality
template<typename T, typename Layout, typename Storage>
bool check_equal(const Matrix<T, Layout, Storage>& A,
                 const Matrix<T, Layout, Storage>& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            if (A.at(i,j) != B.at(i,j)) {
                return false;
            }
        }
    }
    return true;
}


// 1. Basic 2x2 test
bool run_basic_test() {
    std::cout << "Running basic 2x2 test...\n";
    Matrix<int, RowMajor, PlainStorage<int>> A(2,2), B(2,2);
    A.set(0,0,1); A.set(0,1,2);
    A.set(1,0,3); A.set(1,1,4);
    B.set(0,0,5); B.set(0,1,6);
    B.set(1,0,7); B.set(1,1,8);
    Matrix<int, RowMajor, PlainStorage<int>> expected(2,2);
    expected.set(0,0,19); expected.set(0,1,22);
    expected.set(1,0,43); expected.set(1,1,50);
    auto C = matmul(A,B);
    bool pass = check_equal(C, expected);
    std::cout << (pass ? "Basic test PASS\n" : "Basic test FAIL\n");
    return pass;
}

// 2. Negative/zero test
bool run_negative_test() {
    std::cout << "Running negative/zero test...\n";
    Matrix<int, RowMajor, PlainStorage<int>> A(2,3), B(3,2);
    A.set(0,0,0);  A.set(0,1,-2); A.set(0,2,1000);
    A.set(1,0,5);  A.set(1,1,0);   A.set(1,2,1);
    B.set(0,0,0);  B.set(0,1,3);
    B.set(1,0,-1); B.set(1,1,-1);
    B.set(2,0,2);  B.set(2,1,2);
    Matrix<int, RowMajor, PlainStorage<int>> expected(2,2);
    expected.set(0,0,2002); expected.set(0,1,2002);
    expected.set(1,0,2);    expected.set(1,1,17);
    auto C = matmul(A,B);
    bool pass = check_equal(C, expected);
    std::cout << (pass ? "Negative test PASS\n" : "Negative test FAIL\n");
    return pass;
}

// 3. Non-square test
bool run_non_square_test() {
    std::cout << "Running non-square test...\n";
    Matrix<int, RowMajor, PlainStorage<int>> A(3,2), B(2,4);
    int valsA[3][2] = {{1,2},{3,4},{5,6}};
    int valsB[2][4] = {{7,8,9,10},{11,12,13,14}};
    for (int i=0; i<3; ++i) for (int j=0; j<2; ++j) A.set(i,j, valsA[i][j]);
    for (int i=0; i<2; ++i) for (int j=0; j<4; ++j) B.set(i,j, valsB[i][j]);
    Matrix<int, RowMajor, PlainStorage<int>> expected(3,4);
    int exp[3][4] = {{29,32,35,38},{65,72,79,86},{101,112,123,134}};
    for (int i=0; i<3; ++i) for (int j=0; j<4; ++j) expected.set(i,j, exp[i][j]);
    auto C = matmul(A,B);
    bool pass = check_equal(C, expected);
    std::cout << (pass ? "Non-square test PASS\n" : "Non-square test FAIL\n");
    return pass;
}


// 4a. Int4 fixed test
bool run_int4_fixed_test() {
    std::cout << "Running int4 fixed test...\n";
    Matrix<uint8_t, RowMajor, Int4Storage> A4(2,2), B4(2,2);
    A4.set(0,0,1); A4.set(0,1,2); A4.set(1,0,3); A4.set(1,1,4);
    B4.set(0,0,5); B4.set(0,1,6); B4.set(1,0,7); B4.set(1,1,0);
    Matrix<int, RowMajor, PlainStorage<int>> A4u(2,2), B4u(2,2);
    for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) {
        A4u.set(i,j, A4.at(i,j));
        B4u.set(i,j, B4.at(i,j));
    }
    auto C = matmul(A4u, B4u);
    Matrix<int, RowMajor, PlainStorage<int>> expected(2,2);
    expected.set(0,0,19); expected.set(0,1,6);
    expected.set(1,0,43); expected.set(1,1,18);
    bool pass = check_equal(C, expected);
    std::cout << (pass ? "Int4 fixed test PASS\n" : "Int4 fixed test FAIL\n");
    return pass;
}

// 4b. Int4 boundary test
bool run_int4_boundary_test() {
    std::cout << "Running int4 boundary test...\n";
    Matrix<uint8_t, RowMajor, Int4Storage> A4(2,2), B4(2,2);
    A4.set(0,0,7); A4.set(0,1,8); A4.set(1,0,0); A4.set(1,1,15);
    B4 = A4;
    Matrix<int, RowMajor, PlainStorage<int>> A4u(2,2), B4u(2,2);
    for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) {
        int rawA = A4.at(i, j);
        int rawB = B4.at(i, j);
        int s4A  = (rawA < 8 ? rawA : rawA - 16);
        int s4B  = (rawB < 8 ? rawB : rawB - 16);
        A4u.set(i, j, s4A);
        B4u.set(i, j, s4B);
    }
    auto C = matmul(A4u, B4u);
    Matrix<int, RowMajor, PlainStorage<int>> expected(2,2);
    expected.set(0,0,49); expected.set(0,1,-48);
    expected.set(1,0,0);  expected.set(1,1,1);
    bool pass = check_equal(C, expected);
    std::cout << (pass ? "Int4 boundary test PASS\n" : "Int4 boundary test FAIL\n");
    return pass;
}

// 4c. Int4 dimension test
bool run_int4_dimension_test() {
    std::cout << "Running int4 dimension test...\n";
    Matrix<uint8_t, RowMajor, Int4Storage> A4(3,2), B4(2,4);
    int valsA[3][2] = {{1,2},{3,4},{5,6}};
    int valsB[2][4] = {{7,8,9,10},{11,12,13,14}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 2; ++j)
            A4.set(i, j, valsA[i][j]);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 4; ++j)
            B4.set(i, j, valsB[i][j]);

    // unpack with two's-complement mapping raw∈[0,15] → s4∈[-8,+7]
    Matrix<int, RowMajor, PlainStorage<int>> A4u(3,2), B4u(2,4);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            int raw = A4.at(i, j);
            int s4  = (raw < 8 ? raw : raw - 16);
            A4u.set(i, j, s4);
        }
    }
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            int raw = B4.at(i, j);
            int s4  = (raw < 8 ? raw : raw - 16);
            B4u.set(i, j, s4);
        }
    }

    auto C = matmul(A4u, B4u);

    Matrix<int, RowMajor, PlainStorage<int>> expected(3,4);
    expected.set(0,0,-3); expected.set(0,1,-16); expected.set(0,2,-13); expected.set(0,3,-10);
    expected.set(1,0,1);  expected.set(1,1,-40); expected.set(1,2,-33); expected.set(1,3,-26);
    expected.set(2,0,5);  expected.set(2,1,-64); expected.set(2,2,-53); expected.set(2,3,-42);

    bool pass = check_equal(C, expected);
    std::cout << (pass ? "Int4 dimension test PASS\n"
                       : "Int4 dimension test FAIL\n");
    return pass;
}


// 5a. Int4 x int16 test using LUT
bool run_int4_int16_test() {
    std::cout << "Running int4 x int16 test...\n";
    Matrix<uint8_t,RowMajor,Int4Storage> A4(2,3);
    Matrix<int16_t, RowMajor, PlainStorage<int16_t>> B16(3,2);
    for(int i=0;i<2;++i) for(int j=0;j<3;++j) A4.set(i,j,(i+j)&0xF);
    for(int i=0;i<3;++i) for(int j=0;j<2;++j) B16.set(i,j,(i*2+j)&0x7);
    Matrix<int16_t, RowMajor, PlainStorage<int16_t>> A4u(2,3);
    for(int i=0;i<2;++i) for(int j=0;j<3;++j) A4u.set(i,j,A4.at(i,j));
    auto C1 = matmul(A4u, B16);
    ProductLookupTable<uint8_t,int16_t> lut(16,8);
    Matrix<int16_t,RowMajor,PlainStorage<int16_t>> C_lut(2,2);
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        int16_t s=0;
        for(int k=0;k<3;++k) s += lut.get(A4.at(i,k), B16.at(k,j));
        C_lut.set(i,j,s);
    }
    bool pass = check_equal(C1, C_lut);
    std::cout << (pass ? "Int4 x int16 test PASS\n" : "Int4 x int16 test FAIL\n");
    return pass;
}

// 5b. Int4 x int32 test using LUT
bool run_int4_int32_test() {
    std::cout << "Running int4 x int32 test...\n";
    Matrix<uint8_t,RowMajor,Int4Storage> A4(2,3);
    Matrix<int32_t, RowMajor, PlainStorage<int32_t>> B32(3,2);
    for(int i=0;i<2;++i) for(int j=0;j<3;++j) A4.set(i,j,(i*3+j)&0xF);
    for(int i=0;i<3;++i) for(int j=0;j<2;++j) B32.set(i,j,(i+j)&0xF);
    Matrix<int32_t, RowMajor, PlainStorage<int32_t>> A4u(2,3);
    for(int i=0;i<2;++i) for(int j=0;j<3;++j) A4u.set(i,j,A4.at(i,j));
    auto C1 = matmul(A4u, B32);
    ProductLookupTable<uint8_t,int32_t> lut(16,16);
    Matrix<int32_t,RowMajor,PlainStorage<int32_t>> C_lut(2,2);
    for(int i=0;i<2;++i) for(int j=0;j<2;++j) {
        int32_t s=0;
        for(int k=0;k<3;++k) s += lut.get(A4.at(i,k), B32.at(k,j));
        C_lut.set(i,j,s);
    }
    bool pass = check_equal(C1, C_lut);
    std::cout << (pass ? "Int4 x int32 test PASS\n" : "Int4 x int32 test FAIL\n");
    return pass;
}

// 6. Int4 fast lut test
bool run_int4_fast_test(){
    std::cout << "Running Int4 fast‑kernel test...\n";
    constexpr int M=4,K=5,N=3;

    using Mat4R = Matrix<uint8_t, RowMajor, Int4Storage>;
    using Mat4C = Matrix<uint8_t, ColMajor, Int4Storage>;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> d4(0,15);

    Mat4R A4(M,K); Mat4C B4(K,N);
    for(int i=0;i<M;++i) for(int k=0;k<K;++k) A4.set(i,k,d4(rng));
    for(int k=0;k<K;++k) for(int j=0;j<N;++j) B4.set(k,j,d4(rng));

    // reference: unpack -> naive matmul
    auto Au = unpack_int4(A4);
    auto Bu = unpack_int4(B4);
    Matrix<int,RowMajor,PlainStorage<int>> Au_mat(M,K), Bu_mat(K,N);
    for(int i=0;i<M;++i) for(int k=0;k<K;++k) Au_mat.set(i,k,Au[i*K+k]);
    for(int k=0;k<K;++k) for(int j=0;j<N;++j) Bu_mat.set(k,j,Bu[k*N+j]);
    auto C_ref = matmul(Au_mat,Bu_mat);

    // LUT fast kernel
    ProductLookupTable<uint8_t,uint8_t,int32_t> lut(16,16);
    auto C_fast = matmul_lut_fast(Au,Bu,M,K,N,lut);

    bool pass = check_equal(C_ref,C_fast);
    std::cout << (pass?"PASS":"FAIL") << "\n";
    return pass;
}


// 7. Quantization/Dequantization test
bool run_quant_dequant_test() {
    std::cout << "Running INT4 quant-dequant test...\n";
    float scale = 0.25f;       // 假設
    bool pass = true;
    for (float v : {0.0f, 1.0f, 2.25f, 3.5f}) {
        uint8_t q = quantize_int4(v, scale);
        float   d = dequantize_int4(q, scale);
        if (std::abs(d - std::round(v/scale)*scale) > 1e-3f) pass = false;
    }
    std::cout << (pass ? "Quant-Dequant test PASS\n" : "FAIL\n");
    return pass;
}

// 8. MKL test
#ifdef USE_MKL
// 8. MKL test
bool run_mkl_test() {
    std::cout << "Running MKL test...\n";

    Matrix<float, RowMajor, PlainStorage<float>> A(2,3), B(3,2);
    for (int i=0; i<2; ++i) for (int j=0; j<3; ++j) A.set(i,j,i+j+1);
    for (int i=0; i<3; ++i) for (int j=0; j<2; ++j) B.set(i,j,i+j+1);

    auto C = matmul_mkl(A,B);

    Matrix<float, RowMajor, PlainStorage<float>> expected(2,2);
    expected.set(0,0,14); expected.set(0,1,20);
    expected.set(1,0,20); expected.set(1,1,29);

    bool pass = check_equal(C, expected);
    std::cout << (pass ? "MKL test PASS\n" : "MKL test FAIL\n");
    return pass;
}
#endif  // USE_MKL

// 9. Bias addition test
bool run_bias_test() {
    std::cout << "Running bias addition test...\n";
    // 2×3 範例
    Matrix<int,RowMajor,PlainStorage<int>> M(2,3);
    M.set(0,0,1); M.set(0,1,2); M.set(0,2,3);
    M.set(1,0,4); M.set(1,1,5); M.set(1,2,6);
    std::vector<int> bias = {10, 20, 30};
    auto R = add_bias(M, bias);
    Matrix<int,RowMajor,PlainStorage<int>> expected(2,3);
    expected.set(0,0,11); expected.set(0,1,22); expected.set(0,2,33);
    expected.set(1,0,14); expected.set(1,1,25); expected.set(1,2,36);
    bool pass = check_equal(R, expected);
    std::cout << (pass?"Bias test PASS\n":"Bias test FAIL\n");
    return pass;
}

// 10. ReLU activation test
bool run_relu_test() {
    std::cout << "Running ReLU test...\n";
    Matrix<int,RowMajor,PlainStorage<int>> M(2,2), E(2,2);
    // data
    M.set(0,0,-1); M.set(0,1,0);
    M.set(1,0, 5); M.set(1,1,-3);
    // expected
    E.set(0,0, 0); E.set(0,1, 0);
    E.set(1,0, 5); E.set(1,1, 0);
    auto R = apply_activation(M, Activation::ReLU);
    bool pass = check_equal(R, E);
    std::cout << (pass?"ReLU test PASS\n":"ReLU test FAIL\n");
    return pass;
}

// 11. Sigmoid activation test
bool run_sigmoid_test() {
    std::cout << "Running Sigmoid test...\n";
    Matrix<float,RowMajor,PlainStorage<float>> M(1,3);
    M.set(0,0, 0.0f);
    M.set(0,1, 2.0f);
    M.set(0,2,-2.0f);
    auto R = apply_activation(M, Activation::Sigmoid);

    // 理論值
    float s0 = 1.0f/(1+std::exp(-0.0f)); // 0.5
    float s1 = 1.0f/(1+std::exp(-2.0f));
    float s2 = 1.0f/(1+std::exp( 2.0f));

    const float eps = 1e-6f;
    bool pass = std::fabs(R.at(0,0)-s0)<eps
             && std::fabs(R.at(0,1)-s1)<eps
             && std::fabs(R.at(0,2)-s2)<eps;

    std::cout << (pass?"Sigmoid test PASS\n":"Sigmoid test FAIL\n");
    return pass;
}

// 12. Tanh activation test
bool run_tanh_test() {
    std::cout << "Running Tanh test...\n";
    Matrix<float,RowMajor,PlainStorage<float>> M(1,3);
    M.set(0,0, 0.0f);
    M.set(0,1, 1.0f);
    M.set(0,2,-1.0f);
    auto R = apply_activation(M, Activation::Tanh);

    float t0 = std::tanh(0.0f); // 0
    float t1 = std::tanh(1.0f);
    float t2 = std::tanh(-1.0f);

    const float eps = 1e-6f;
    bool pass = std::fabs(R.at(0,0)-t0)<eps
             && std::fabs(R.at(0,1)-t1)<eps
             && std::fabs(R.at(0,2)-t2)<eps;

    std::cout << (pass?"Tanh test PASS\n":"Tanh test FAIL\n");
    return pass;
}

// 13. Linear (identity) test
bool run_linear_test() {
    std::cout << "Running Linear (identity) test...\n";
    Matrix<int,RowMajor,PlainStorage<int>> M(2,2), E(2,2);
    M.set(0,0,1); M.set(0,1,-2);
    M.set(1,0, 0); M.set(1,1, 5);
    // Linear = no-op
    E = M;
    auto R = apply_activation(M, Activation::Linear);
    bool pass = check_equal(R, E);
    std::cout << (pass?"Linear test PASS\n":"Linear test FAIL\n");
    return pass;
}

// 14. accuracy test
bool run_accuracy_test() {
    std::cout << "Running accuracy test...\n";
    std::vector<float> A = {1.0f, 2.0f, 3.0f};
    std::vector<float> B = {1.1f, 1.9f, 2.5f};
    auto stats = measure_error(A, B);
    // manual:
    // diffs = {0.1, -0.1, -0.5}, sq = {0.01,0.01,0.25}, mse = 0.27/3 = 0.09
    assert(std::fabs(stats.mse - 0.09) < 1e-6);
    assert(std::fabs(stats.max_error - 0.5) < 1e-6);
    std::cout << "Accuracy test PASS\n";
    return true;
}


int main() {
    int passed=0;
    int total=16;
    if (run_basic_test()) ++passed;
    if (run_negative_test()) ++passed;
    if (run_non_square_test()) ++passed;
    if (run_int4_fixed_test()) ++passed;
    if (run_int4_boundary_test()) ++passed;
    if (run_int4_dimension_test()) ++passed;
    if (run_int4_int16_test()) ++passed;
    if (run_int4_int32_test()) ++passed;
    if(run_int4_fast_test()) ++passed;
    if (run_quant_dequant_test()) ++passed;
    if (run_bias_test()) ++passed;
    if (run_relu_test()) ++passed;
    if (run_sigmoid_test()) ++passed;
    if (run_tanh_test()) ++passed;
    if (run_linear_test()) ++passed;
    if (run_accuracy_test()) ++passed;
    #ifdef USE_MKL
        ++total;                  // 只有啟用 MKL 才加總數
        if (run_mkl_test()) ++passed;
    #endif
    std::cout << "\nTotal: " << passed << "/" << total << " tests passed.\n";
    return (passed == total ? 0 : 1);
}
