#include "../src/matrix.hpp"
#include "../src/matrix_packed.hpp"
#include "../src/matrix_packed_convert.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <sstream>
#include <vector>
#include <string>

//====================================================
// 1. 定義用來比對兩個 Row_Major_Matrix 是否相等的函式
template <typename T>
bool check_equal(const Row_Major_Matrix<T>& A, const Row_Major_Matrix<T>& B) {
    int A_rows = A.all_row.size();
    int A_cols = (A.all_row.empty() ? 0 : A.all_row[0].size());
    int B_rows = B.all_row.size();
    int B_cols = (B.all_row.empty() ? 0 : B.all_row[0].size());

    if (A_rows != B_rows || A_cols != B_cols) {
        std::cout << "Size mismatch: A(" << A_rows << "," << A_cols 
                  << ") vs B(" << B_rows << "," << B_cols << ")\n";
        return false;
    }
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < A_cols; ++j) {
            if (A.all_row[i][j] != B.all_row[i][j]) {
                std::cout << "Mismatch at (" << i << "," << j << "): "
                          << A.all_row[i][j] << " vs " << B.all_row[i][j] << "\n";
                return false;
            }
        }
    }
    return true;
}

//====================================================
// 2. 輔助函式：檔案檢查與建立
bool file_exists(const std::string &filename) {
    std::ifstream f(filename);
    return f.good();
}

void generate_test_file(const std::string &filename, const std::string &data) {
    std::ofstream out(filename);
    if (!out)
        throw std::runtime_error("Cannot create file: " + filename);
    out << data;
    out.close();
}

//====================================================
// 3. 載入檔案中的矩陣（假設以 row-major 格式存放）  
// 檔案格式：第一行為 "rows cols"，接著每行為一列的數據
Row_Major_Matrix<int> load_matrix_txt(const std::string &filename) {
    std::ifstream in(filename);
    if (!in)
        throw std::runtime_error("Cannot open file: " + filename);
    int rows, cols;
    in >> rows >> cols;
    Row_Major_Matrix<int> mat(rows, cols);
    // 覆蓋原本以隨機產生的數值
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) {
            int value;
            in >> value;
            mat.all_row[i][j] = value;
        }
    return mat;
}

//====================================================
// 4. 一般矩陣測試案例

// (1) 基本 2x2 測試
bool run_basic_test() {
    std::cout << "Running basic 2x2 test...\n";
    Row_Major_Matrix<int> A(2, 2);
    A.all_row = { {1, 2}, {3, 4} };

    Row_Major_Matrix<int> B_row(2, 2);
    B_row.all_row = { {5, 6}, {7, 8} };
    Column_Major_Matrix<int> B = B_row;

    // 預期乘法結果：
    // [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3*5+4*7, 3*6+4*8] = [43, 50]
    Row_Major_Matrix<int> expected(2, 2);
    expected.all_row = { {19, 22}, {43, 50} };

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, expected) &&
                check_equal(result_multi, expected);
    std::cout << (pass ? "Basic test PASS\n" : "Basic test FAIL\n");
    return pass;
}

// (2) 測試包含零與負數
bool run_negative_test() {
    std::cout << "Running negative/zero test...\n";
    Row_Major_Matrix<int> A(2, 3);
    A.all_row = { {0, -2, 1000}, {5, 0, 1} };

    Row_Major_Matrix<int> B_row(3, 2);
    B_row.all_row = { {0, 3}, {-1, -1}, {2, 2} };
    Column_Major_Matrix<int> B = B_row;

    // 預期結果計算：
    // Row0, Col0: 0*0 + (-2)*(-1) + 1000*2 = 2002
    // Row0, Col1: 同上 = 2002
    // Row1, Col0: 5*0 + 0*(-1) + 1*2 = 2
    // Row1, Col1: 5*3 + 0*(-1) + 1*2 = 17
    Row_Major_Matrix<int> expected(2, 2);
    expected.all_row = { {2002, 2002}, {2, 17} };

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, expected) &&
                check_equal(result_multi, expected);
    std::cout << (pass ? "Negative/zero test PASS\n" : "Negative/zero test FAIL\n");
    return pass;
}

// (3) 非方陣測試：A 為 3x2, B 為 2x4
bool run_non_square_test() {
    std::cout << "Running non-square matrix test...\n";
    Row_Major_Matrix<int> A(3, 2);
    A.all_row = { {1, 2}, {3, 4}, {5, 6} };

    Row_Major_Matrix<int> B_row(2, 4);
    B_row.all_row = { {7, 8, 9, 10}, {11, 12, 13, 14} };
    Column_Major_Matrix<int> B = B_row;

    // 預期結果：
    // Row0: [29, 32, 35, 38]
    // Row1: [65, 72, 79, 86]
    // Row2: [101,112,123,134]
    Row_Major_Matrix<int> expected(3, 4);
    expected.all_row = {
        {29, 32, 35, 38},
        {65, 72, 79, 86},
        {101,112,123,134}
    };

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, expected) &&
                check_equal(result_multi, expected);
    std::cout << (pass ? "Non-square test PASS\n" : "Non-square test FAIL\n");
    return pass;
}

// (4) 檔案測試：使用大矩陣檔案進行乘法，比對單、多執行緒結果
bool run_file_test() {
    std::cout << "Running file-based large matrix test...\n";
    std::string fileA = "../data/large_A.txt";
    std::string fileB = "../data/large_B.txt";

    if (!file_exists(fileA)) {
        int rows = 50, cols = 60;
        std::ostringstream oss;
        oss << rows << " " << cols << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                oss << (i * cols + j + 1) << " ";
            oss << "\n";
        }
        generate_test_file(fileA, oss.str());
        std::cout << "Generated " << fileA << "\n";
    }
    if (!file_exists(fileB)) {
        int rows = 60, cols = 40;
        std::ostringstream oss;
        oss << rows << " " << cols << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j)
                oss << (i * cols + j + 1) << " ";
            oss << "\n";
        }
        generate_test_file(fileB, oss.str());
        std::cout << "Generated " << fileB << "\n";
    }

    Row_Major_Matrix<int> A = load_matrix_txt(fileA);
    Row_Major_Matrix<int> B_row = load_matrix_txt(fileB);
    Column_Major_Matrix<int> B = B_row;

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, result_multi);
    std::cout << (pass ? "File-based matrix test PASS\n" : "File-based matrix test FAIL\n");
    return pass;
}

//====================================================
// 5. Int4 版本測試案例

// (5a) Int4 Fixed 測試：使用固定 2x2 數值（均小於8，不涉及負數）
bool run_int4_fixed_test() {
    std::cout << "Running int4 fixed 2x2 test...\n";
    int a_rows = 2, a_cols = 2, b_rows = 2, b_cols = 2;
    PackedInt4Matrix A4(a_rows, a_cols);
    PackedInt4Matrix B4(b_rows, b_cols);
    
    // 設定 A4: [ [1, 2], [3, 4] ]
    A4.set(0, 0, 1);
    A4.set(0, 1, 2);
    A4.set(1, 0, 3);
    A4.set(1, 1, 4);
    
    // 設定 B4: [ [5, 6], [7, 0] ]
    B4.set(0, 0, 5);
    B4.set(0, 1, 6);
    B4.set(1, 0, 7);
    B4.set(1, 1, 0);
    
    auto A4_unpacked = A4.to_row_major<int>(1.0f, 0.0f);
    auto B4_unpacked = B4.to_col_major<int>(1.0f, 0.0f);
    
    auto result_single = A4_unpacked * B4_unpacked;
    auto result_multi = A4_unpacked % B4_unpacked;
    
    // 預期結果：
    // [ 1*5+2*7, 1*6+2*0 ] = [19, 6]
    // [ 3*5+4*7, 3*6+4*0 ] = [43, 18]
    Row_Major_Matrix<int> expected(2, 2);
    expected.all_row = { {19, 6}, {43, 18} };
    
    bool pass = check_equal(result_single, expected) &&
                check_equal(result_multi, expected);
    std::cout << (pass ? "Int4 fixed test PASS\n" : "Int4 fixed test FAIL\n");
    return pass;
}

// (5b) Int4 Boundary 測試：檢查邊界數值轉換與運算
bool run_int4_boundary_test() {
    std::cout << "Running int4 boundary 2x2 test...\n";
    int a_rows = 2, a_cols = 2;
    int b_rows = a_cols, b_cols = 2;
    PackedInt4Matrix A4(a_rows, a_cols);
    PackedInt4Matrix B4(b_rows, b_cols);
    
    // 設定邊界值：
    // 使用數值: 7, 8, 0, 15
    // 解量化後：7 -> 7, 8 -> (8-16) = -8, 0 -> 0, 15 -> (15-16) = -1
    A4.set(0, 0, 7);
    A4.set(0, 1, 8);
    A4.set(1, 0, 0);
    A4.set(1, 1, 15);
    
    // 為 B4 同樣設定相同邊界數值
    B4.set(0, 0, 7);
    B4.set(0, 1, 8);
    B4.set(1, 0, 0);
    B4.set(1, 1, 15);
    
    auto A4_unpacked = A4.to_row_major<int>(1.0f, 0.0f);
    auto B4_unpacked = B4.to_col_major<int>(1.0f, 0.0f);
    
    auto result_single = A4_unpacked * B4_unpacked;
    auto result_multi = A4_unpacked % B4_unpacked;
    
    // 解量化後矩陣為：
    // [ [7, -8],
    //   [0, -1] ]
    // 乘法結果：
    // [0,0]: 7*7 + (-8)*0 = 49
    // [0,1]: 7*(-8) + (-8)*(-1) = -56 + 8 = -48
    // [1,0]: 0*7 + (-1)*0 = 0
    // [1,1]: 0*(-8) + (-1)*(-1) = 1
    Row_Major_Matrix<int> expected(2, 2);
    expected.all_row = { {49, -48}, {0, 1} };
    
    bool pass = check_equal(result_single, expected) &&
                check_equal(result_multi, expected);
    std::cout << (pass ? "Int4 boundary test PASS\n" : "Int4 boundary test FAIL\n");
    return pass;
}

// (5c) Int4 Dimension 測試：非方陣
bool run_int4_dimension_test() {
    std::cout << "Running int4 non-square test...\n";
    // A4: 3x2, B4: 2x4 (固定數值均小於8，無負轉換)
    int a_rows = 3, a_cols = 2, b_rows = 2, b_cols = 4;
    PackedInt4Matrix A4(a_rows, a_cols);
    PackedInt4Matrix B4(b_rows, b_cols);
    
    // 設定 A4 為：
    // [ [1, 2],
    //   [3, 4],
    //   [5, 6] ]
    A4.set(0, 0, 1); A4.set(0, 1, 2);
    A4.set(1, 0, 3); A4.set(1, 1, 4);
    A4.set(2, 0, 5); A4.set(2, 1, 6);
    
    // 設定 B4 為：
    // [ [7, 8, 9, 10],
    //   [11, 12, 13, 14] ]
    B4.set(0, 0, 7);  B4.set(0, 1, 8);
    B4.set(0, 2, 9);  B4.set(0, 3, 10);
    B4.set(1, 0, 11); B4.set(1, 1, 12);
    B4.set(1, 2, 13); B4.set(1, 3, 14);
    
    auto A4_unpacked = A4.to_row_major<int>(1.0f, 0.0f);
    auto B4_unpacked = B4.to_col_major<int>(1.0f, 0.0f);
    
    auto result_single = A4_unpacked * B4_unpacked;
    auto result_multi = A4_unpacked % B4_unpacked;
    
    // 正確的預期結果（依據 int4 轉換後的實際數值計算）：
    // A4 (unchanged): [ [1,2], [3,4], [5,6] ]
    // B4 解量化後：
    // 第一列： 7, (8->-8), (9->-7), (10->-6)
    // 第二列: (11->-5), (12->-4), (13->-3), (14->-2)
    // 計算得到：
    // Row0: [ 1*7 + 2*(-5) = -3,
    //         1*(-8) + 2*(-4) = -16,
    //         1*(-7) + 2*(-3) = -13,
    //         1*(-6) + 2*(-2) = -10 ]
    // Row1: [ 3*7 + 4*(-5) = 1,
    //         3*(-8) + 4*(-4) = -40,
    //         3*(-7) + 4*(-3) = -33,
    //         3*(-6) + 4*(-2) = -26 ]
    // Row2: [ 5*7 + 6*(-5) = 5,
    //         5*(-8) + 6*(-4) = -64,
    //         5*(-7) + 6*(-3) = -53,
    //         5*(-6) + 6*(-2) = -42 ]
    Row_Major_Matrix<int> expected(3, 4);
    expected.all_row = {
        { -3, -16, -13, -10 },
        {  1, -40, -33, -26 },
        {  5, -64, -53, -42 }
    };
    
    bool pass = check_equal(result_single, expected) &&
                check_equal(result_multi, expected);
    std::cout << (pass ? "Int4 non-square test PASS\n" : "Int4 non-square test FAIL\n");
    return pass;
}

//====================================================
// 6. 主程式：依序執行所有測試案例
int main() {
    int pass_count = 0;
    // 原有 4 測試 + 3 int4 測試 = 7 測試案例
    const int total_tests = 7;

    if (run_basic_test())         ++pass_count;
    if (run_negative_test())      ++pass_count;
    if (run_non_square_test())    ++pass_count;
    if (run_file_test())          ++pass_count;
    if (run_int4_fixed_test())    ++pass_count;
    if (run_int4_boundary_test()) ++pass_count;
    if (run_int4_dimension_test())++pass_count;

    std::cout << "\nTotal: " << pass_count << " / " << total_tests << " tests passed.\n";
    return (pass_count == total_tests) ? 0 : 1;
}
