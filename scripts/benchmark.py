import subprocess
import re
import statistics
import os
import sys
import argparse

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Benchmark GEMM implementations.")
parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
args = parser.parse_args()
RUNS = args.runs

# === Setup ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)
MAKE_CMD = ["make", "run"]

# === Benchmark ===
results = {}  # backend_name -> list of times
pattern = r"\[\s*(.*?)\s*\]\s+Time:\s+([\d.]+) ms"

print(f"Running benchmark {RUNS} times...\n")

for i in range(RUNS):
    print(f"Run {i+1}...", end=' ', flush=True)
    result = subprocess.run(MAKE_CMD, capture_output=True, text=True)
    output = result.stdout + result.stderr

    if result.returncode != 0:
        print("❌ Run failed.")
        print(result.stdout)
        print(result.stderr)
        break

    matches = re.findall(pattern, output)
    if not matches:
        print("❌ Failed to parse benchmark output.")
        print("Output:\n", output)
        break

    for backend, time_str in matches:
        results.setdefault(backend, []).append(float(time_str))
    print("✔️ ", ", ".join([f"{b}: {float(t):.2f}ms" for b, t in matches]))

# === Summary ===
if results:
    print("\n====== Benchmark Summary ======")
    for backend, times in sorted(results.items(), key=lambda x: statistics.mean(x[1])):
        print(f"{backend:12} avg: {statistics.mean(times):.3f} ms over {len(times)} runs")
