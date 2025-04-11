#include "../src/matrix.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <sstream>
#include <vector>
#include <string>

//====================================================
// 定義資料檔路徑目錄
const std::string DATA_DIR = "../data/";

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
// 4. 測試案例

// (1) 基本 2x2 測試
bool run_basic_test() {
    std::cout << "Running basic 2x2 test...\n";
    // 利用直接指定數值來建立矩陣
    Row_Major_Matrix<int> A(2, 2);
    A.all_row = { {1, 2}, {3, 4} };

    // 為 B 建立一個 row-major 矩陣後，轉換成 column-major（透過類別的轉換運算子）
    Row_Major_Matrix<int> B_row(2, 2);
    B_row.all_row = { {5, 6}, {7, 8} };
    Column_Major_Matrix<int> B = B_row;

    // 預期乘法結果：
    // [ 1*5+2*7, 1*6+2*8 ] = [19, 22]
    // [ 3*5+4*7, 3*6+4*8 ] = [43, 50]
    Row_Major_Matrix<int> expected(2, 2);
    expected.all_row = { {19, 22}, {43, 50} };

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, expected) && check_equal(result_multi, expected);
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

    // 預期結果手動計算：
    // Row0, Col0: 0*0 + (-2)*(-1) + 1000*2 = 0 + 2 + 2000 = 2002
    // Row0, Col1: 同上 = 2002
    // Row1, Col0: 5*0 + 0*(-1) + 1*2 = 2
    // Row1, Col1: 5*3 + 0*(-1) + 1*2 = 15+2 = 17
    Row_Major_Matrix<int> expected(2, 2);
    expected.all_row = { {2002, 2002}, {2, 17} };

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, expected) && check_equal(result_multi, expected);
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
    // Row0: [1*7+2*11, 1*8+2*12, 1*9+2*13, 1*10+2*14] = [29,32,35,38]
    // Row1: [3*7+4*11, 3*8+4*12, 3*9+4*13, 3*10+4*14] = [65,72,79,86]
    // Row2: [5*7+6*11, 5*8+6*12, 5*9+6*13, 5*10+6*14] = [101,112,123,134]
    Row_Major_Matrix<int> expected(3, 4);
    expected.all_row = { {29, 32, 35, 38},
                         {65, 72, 79, 86},
                         {101,112,123,134} };

    auto result_single = A * B;
    auto result_multi = A % B;

    bool pass = check_equal(result_single, expected) && check_equal(result_multi, expected);
    std::cout << (pass ? "Non-square test PASS\n" : "Non-square test FAIL\n");
    return pass;
}

// (4) 檔案測試：使用大矩陣檔案進行乘法，並比對單執行緒與多執行緒結果
bool run_file_test() {
    std::cout << "Running file-based large matrix test...\n";
    // 設定檔案完整路徑
    std::string fileA = DATA_DIR + "large_A.txt";
    std::string fileB = DATA_DIR + "large_B.txt";

    // 若檔案不存在則自動建立
    if (!file_exists(fileA)) {
        // 產生一個 50x60 的矩陣，數值採用連續整數
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
        // 產生一個 60x40 的矩陣
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

    // 讀入檔案中的資料
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
// 5. 主程式：依序執行所有測試案例
int main() {
    int pass_count = 0;
    const int total_tests = 4;

    if (run_basic_test())         ++pass_count;
    if (run_negative_test())      ++pass_count;
    if (run_non_square_test())    ++pass_count;
    if (run_file_test())          ++pass_count;

    std::cout << "\nTotal: " << pass_count << " / " << total_tests << " tests passed.\n";
    return (pass_count == total_tests) ? 0 : 1;
}
