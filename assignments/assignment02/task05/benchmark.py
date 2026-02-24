import subprocess
import matplotlib.pyplot as plt
import os

# Configuration
SIZES = [100, 500, 1000, 2000, 5000, 10000, 12500, 15000] 
TILE_SIZES = [8, 16, 32]
NUM_MATRICES = 2
INPUT_FILE = "bench_input.txt"
TILED_EXE = "./build/gpu_tiled_solver"
GPU_EXE = "./build/gpu_solver"

def generate_input_file(n, size):
    """Generates a file with 'n' matrices of size 'size x size'"""
    print(f"    Generating {size}x{size} input file...")
    with open(INPUT_FILE, "w") as f:
        f.write(f"{n}\n")
        for _ in range(n):
            f.write(f"{size} {size}\n")
            # Using a slightly faster way to write dummy data
            data = "1.0 " * (size * size)
            f.write(data + "\n")

def run_solver(executable, tile_size=None):
    # Args: [exe, tile_size, file] OR [exe, file]
    args = [executable]
    if tile_size:
        args.append(str(tile_size))
    args.append(INPUT_FILE)
    
    result = subprocess.run(args, capture_output=True, text=True)
    
    # We'll look for "ms" in the output since our C++ code prints that
    for line in result.stdout.splitlines():
        if "ms" in line.lower():
            # Extract the number (handles "Time: 12.3 ms" or "12.3 ms")
            parts = line.split()
            for p in parts:
                try:
                    return float(p)
                except ValueError:
                    continue
    return 0.0

# Store results: { "8": [times], "16": [times], ... }
all_tiled_times = {str(ts): [] for ts in TILE_SIZES}
gpu_times = []

print(f"Starting Multi-Tile Benchmark...")

for size in SIZES:
    print(f"\nTesting Matrix Size: {size}x{size}")
    generate_input_file(NUM_MATRICES, size)
    
    # 1. Run Baseline (Non-Tiled)
    print(f"    Running Baseline GPU...")
    t_gpu = run_solver(GPU_EXE)
    gpu_times.append(t_gpu)
    
    # 2. Run Tiled for each size
    for ts in TILE_SIZES:
        print(f"    Running Tiled (Size {ts})...")
        t_tiled = run_solver(TILED_EXE, tile_size=ts)
        all_tiled_times[str(ts)].append(t_tiled)

# Plotting
plt.figure(figsize=(12, 7))

# Plot baseline
plt.plot(SIZES, gpu_times, marker='s', label='GPU Naive (Global Mem)', color='black', linewidth=2, linestyle='--')

# Plot each tile size
colors = ['blue', 'green', 'red']
for i, ts in enumerate(TILE_SIZES):
    plt.plot(SIZES, all_tiled_times[str(ts)], marker='o', label=f'GPU Tiled ({ts}x{ts})', color=colors[i])

plt.title('Matrix Multiplication: Naive vs Tiled (Various Sizes)')
plt.xlabel('Matrix Dimension (N x N)')
plt.ylabel('Execution Time (milliseconds)')
plt.xscale('log') # Log scale helps see the small sizes too
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()

plt.savefig("matrix_performance_comparison.png")
print("\nBenchmark complete. Graph saved to matrix_performance_comparison.png")