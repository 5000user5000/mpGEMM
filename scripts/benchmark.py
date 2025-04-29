import subprocess
import re
import statistics
import os
import sys

# 可自訂的次數
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10

# 編譯與執行命令
MAKE_CMD = ["make", "run"]

# 切換回專案根目錄（scripts/ 的上層）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)

# regex 抓時間
# PATTERN_INT = r"INT Process took ([\d.]+) ms"
# PATTERN_INT4 = r"INT4 Process took ([\d.]+) ms"
PATTERN_INT = r"Naive int GEMM: ([\d.]+) ms"
PATTERN_INT4 = r"SIMD‑LUT GEMM: ([\d.]+) ms"

int_times = []
int4_times = []

print(f"Running benchmark {RUNS} times...\n")

for i in range(RUNS):
    print(f"Run {i+1}...", end=' ', flush=True)
    result = subprocess.run(MAKE_CMD, capture_output=True, text=True)
    output = result.stdout + result.stderr

    match_int = re.search(PATTERN_INT, output)
    match_int4 = re.search(PATTERN_INT4, output)

    if match_int and match_int4:
        int_time = float(match_int.group(1))
        int4_time = float(match_int4.group(1))
        int_times.append(int_time)
        int4_times.append(int4_time)
        print(f"INT: {int_time:.3f} ms, INT4: {int4_time:.3f} ms")
    else:
        print("❌ Failed to parse output.")
        print("Output:\n", output)
        break

# 顯示平均
if int_times and int4_times:
    print("\n====== Benchmark Result ======")
    print(f"INT avg:  {statistics.mean(int_times):.3f} ms over {RUNS} runs")
    print(f"INT4 avg: {statistics.mean(int4_times):.3f} ms over {RUNS} runs")
