#include "../src/matrix.hpp"
#include "../src/matrix_packed.hpp"
#include "../src/matrix_packed_convert.hpp"
#include <iostream>
#include <chrono>

int main() {

    int a_rows = 200;
    int a_cols = 300;
    int b_rows = 300;
    int b_cols = 200;


    std::cout << "==== Test with int ====" << std::endl;
    Row_Major_Matrix<int> A(a_rows, a_cols);
    Column_Major_Matrix<int> B(b_rows, b_cols);
    auto start_time = std::chrono::high_resolution_clock::now();
    auto C = A * B;
    auto C2 = A % B; // Multi-threaded for 10 threads
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "INT Process took " << duration.count() << " ms" << std::endl;
    //C.print();

    std::cout << "\n==== Test with Int4 ====" << std::endl;
    PackedInt4Matrix A4(a_rows, a_cols);
    PackedInt4Matrix B4(b_rows, b_cols);
    A4.fill_random();
    B4.fill_random();
    // 解量化並轉成原本的矩陣格式（以 int8_t / int 為例）
    start_time = std::chrono::high_resolution_clock::now();
    auto A4_unpacked = A4.to_row_major<int>(2.0f);   // scale = 2.0f 可自己改
    auto B4_unpacked = B4.to_col_major<int>(2.0f);
    // 套用原本乘法邏輯
    auto C4 = A4_unpacked * B4_unpacked;
    auto C42 = A4_unpacked % B4_unpacked;
    end_time = std::chrono::high_resolution_clock::now();
    duration = end_time - start_time;
    std::cout << "INT4 Process took " << duration.count() << " ms" << std::endl;
    
    //C4.print();

    return 0;
}
