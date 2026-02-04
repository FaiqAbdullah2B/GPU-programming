import subprocess
import random
import matplotlib.pyplot as plt
import os

# Configuration
SIZES = [100, 500, 1000, 2000, 5000, 8000, 20000] # Square matrix dimensions (NxN)
NUM_MATRICES = 2
INPUT_FILE = "bench_input.txt"
CPU_EXE = "./build/cpu_solver"
GPU_EXE = "./build/gpu_solver"

def generate_input_file(n, size):
    """Generates a file with 'n' matrices of size 'size x size'"""
    with open(INPUT_FILE, "w") as f:
        f.write(f"{n}\n")
        for _ in range(n):
            f.write(f"{size} {size}\n")
            # Generating random data - using simple pattern for speed
            data = " ".join(["1.0"] * (size * size))
            f.write(data + "\n")

def run_solver(executable):
    result = subprocess.run([executable, INPUT_FILE], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "TIME_MS:" in line:
            return float(line.split(":")[1].strip())
    return 0.0

cpu_times = []
gpu_times = []

print(f"Running benchmarks...")

for size in SIZES:
    print(f"  Testing Matrix Size: {size}x{size}")
    generate_input_file(NUM_MATRICES, size)
    
    # Run CPU
    t_cpu = run_solver(CPU_EXE)
    cpu_times.append(t_cpu)
    
    # Run GPU
    t_gpu = run_solver(GPU_EXE)
    gpu_times.append(t_gpu)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(SIZES, cpu_times, marker='o', label='CPU (Single Thread)', color='blue')
plt.plot(SIZES, gpu_times, marker='s', label='GPU (CUDA + Transfer)', color='red')

plt.title('Matrix Addition Performance: CPU vs GPU')
plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Time (milliseconds)')
plt.grid(True)
plt.legend()
plt.savefig("performance_graph.png")
print("Benchmark complete. Graph saved to performance_graph.png")