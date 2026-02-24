#include <iostream>
#include <cuda_runtime.h>

// Helper to get cores per SM based on Compute Capability
int getCoresPerSM(int major, int minor) {
    if (major == 3) return 192; // Kepler
    if (major == 5) return 128; // Maxwell
    if (major == 6) return (minor == 1 || minor == 2) ? 128 : 64; // Pascal
    if (major == 7) return 64;  // Volta/Turing
    if (major == 8) return (minor == 6) ? 128 : 64; // Ampere (8.6 = 128, 8.0 = 64)
    if (major == 9) return 128; // Hopper/Ada
    return 64; // Fallback
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found. Tragic." << std::endl;
        return 1;
    }

    int dev = 0; // Using first device
    cudaSetDevice(dev);

    // Get standard properties (for name, memory, etc.)
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);

    // Get clock rates in new GPUs through attributes
    int clockRate = 0;
    int memClockRate = 0;
    int busWidth = 0;
    int smCount = 0;

    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memClockRate, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busWidth, cudaDevAttrGlobalMemoryBusWidth, dev);
    cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);

    std::cout << "--- GPU Device Properties for: " << devProp.name << " ---" << std::endl;

    // Basic Props
    std::cout << "devProp.name: Name of the GPU. Value = " 
        << devProp.name << std::endl;

    std::cout << "devProp.major.minor: CUDA Compute Capability Version (The Hardware Architecture "
        << "Instruction set). Value = " << devProp.major << "." << devProp.minor << std::endl;

    std::cout << "devProp.totalGlobalMem: Total amount of VRAM available. Value = " 
        << devProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;

    std::cout << "devProp.sharedMemPerBlock: Maximum shared memory available per thread block. Value = " 
        << devProp.sharedMemPerBlock / 1024 << " KB" << std::endl;

    std::cout << "devProp.regsPerBlock: Number of 32-bit registers available per block. Value = " 
        << devProp.regsPerBlock << std::endl;

    std::cout << "devProp.warpSize: Number of threads in a warp. Value = " 
        << devProp.warpSize << std::endl;

    std::cout << "devProp.maxThreadsPerBlock: Maximum number of threads allowed in a single block. Value = " 
        << devProp.maxThreadsPerBlock << std::endl;

    std::cout << "devProp.maxThreadsDim[3]: Maximum sizes of each dimension of a block. Value = " 
        << devProp.maxThreadsDim[0] << " x " << devProp.maxThreadsDim[1] << " x " << devProp.maxThreadsDim[2] << std::endl;

    std::cout << "devProp.maxGridSize[3]: Maximum sizes of each dimension of a grid. Value = " 
        << devProp.maxGridSize[0] << " x " << devProp.maxGridSize[1] << " x " << devProp.maxGridSize[2] << std::endl;

    std::cout << "devProp.totalConstMem: Total constant memory available on the GPU. Value = " 
        << devProp.totalConstMem / 1024 << " KB" << std::endl;

    std::cout << "devProp.memoryBusWidth: Global memory bus width in bits. Value = " 
        << devProp.memoryBusWidth << " bits" << std::endl;
    
    // Attributes (Again, Specific to Newer CUDA versions)
    std::cout << "\nAttributes.multiProcessorCount: Number of Streaming Multiprocessors. Value = " 
        << smCount << std::endl;
    std::cout << "Attributes.clockRate: Peak Core Clock. Value = " 
        << clockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Attributes.memoryClockRate: Peak Memory Clock. Value = " 
        << memClockRate / 1000.0 << " MHz" << std::endl;
    std::cout << "Attributes.memoryBusWidth: Bus Width. Value = " 
        << busWidth << " bits" << std::endl;

    // Max Global Memory Bandwidth (GB/s)
    // Formula: (Memory Clock (KHz) * 1000 * 2 (DDR/GDDR) * Bus Width / 8) / 1e9
    // Multiply by 2 due to DDR (Double Data Rate)
    double memBandwidth = 2.0 * (double)memClockRate * 1000.0 * (busWidth / 8.0) / 1.0e9;

    // Peak Compute Performance (GFLOPS)
    // Formula: SMs * CoresPerSM * Clock * 2 (Fused Multiply Add, two ops in one clock)
    int coresPerSM = getCoresPerSM(devProp.major, devProp.minor);
    double peakPerformance = (double)smCount * coresPerSM * (clockRate * 1000.0) * 2.0 / 1.0e9;

    std::cout << "\n--- Calculated Performance ---" << std::endl;
    std::cout << "Cores per SM (Architecture Specific): " << coresPerSM << std::endl;
    std::cout << "Total CUDA Cores: " << smCount * coresPerSM << std::endl;
    std::cout << "Max Global Memory Bandwidth: " << memBandwidth << " GB/s" << std::endl;
    std::cout << "Peak Compute Performance: " << peakPerformance << " GFLOPS" << std::endl;

    return 0;
}