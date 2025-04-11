#!/usr/bin/env python3
import os

def generate_matrix_file(filepath, rows, cols, start_value=1):
    """
    產生一個矩陣資料檔案
    檔案格式：
      第一行: rows cols
      後續每行: 該列所有數值，空格分隔
    參數:
      filepath: 存檔路徑 (例如 "../data/large_A.txt")
      rows: 矩陣的行數
      cols: 矩陣的列數
      start_value: 產生數值的起始值 (預設為 1)
    """
    with open(filepath, 'w') as f:
        # 寫入矩陣的尺寸資訊
        f.write(f"{rows} {cols}\n")
        for i in range(rows):
            # 產生一行，每個數值為連續的整數
            row = [str(i * cols + j + start_value) for j in range(cols)]
            f.write(" ".join(row) + "\n")

def main():
    # 設定輸出資料夾為 ../data，若不存在則自動建立
    out_dir = os.path.join("..", "data")
    os.makedirs(out_dir, exist_ok=True)

    # 產生 large_A.txt：50 x 60 的矩陣
    file_a = os.path.join(out_dir, "large_A.txt")
    generate_matrix_file(file_a, 50, 60, start_value=1)
    print(f"Generated matrix file: {file_a}")

    # 產生 large_B.txt：60 x 40 的矩陣
    file_b = os.path.join(out_dir, "large_B.txt")
    generate_matrix_file(file_b, 60, 40, start_value=1)
    print(f"Generated matrix file: {file_b}")

if __name__ == "__main__":
    main()
