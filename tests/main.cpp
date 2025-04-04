#include "../src/matrix.hpp"
#include <iostream>

int main() {

    int a_rows = 200;
    int a_cols = 300;
    int b_rows = 300;
    int b_cols = 200;


    std::cout << "==== Test with int ====" << std::endl;
    Row_Major_Matrix<int> A(a_rows, a_cols);
    Column_Major_Matrix<int> B(b_rows, b_cols);
    auto C = A * B;
    auto C2 = A % B; // Multi-threaded for 10 threads
    //C.print();

    std::cout << "\n==== Test with Int4 ====" << std::endl;
    Row_Major_Matrix<Int4> A4(a_rows, a_cols);
    Column_Major_Matrix<Int4> B4(b_rows, b_cols);
    auto C4 = A4 * B4;
    auto C42 = A4 % B4; // Multi-threaded for 10 threads
    //C4.print();

    return 0;
}
