#include "gpu_ops.cuh"
#include <cuda_runtime.h>
#include <iostream>

// Error checking macro by GEMINI
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while (0)


__global__ void addKernel(double* d_acc, const double* d_in, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_acc[idx] += d_in[idx];
    }
}

void allocateDeviceMemory(double** d_ptr, size_t size) {
    CUDA_CHECK(cudaMalloc((void**)d_ptr, size * sizeof(double)));
}

void freeDeviceMemory(double* d_ptr) {
    CUDA_CHECK(cudaFree(d_ptr));
}

void copyToDevice(double* d_dest, const double* h_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(d_dest, h_src, size * sizeof(double), cudaMemcpyHostToDevice));
}

void copyToHost(double* h_dest, const double* d_src, size_t size) {
    CUDA_CHECK(cudaMemcpy(h_dest, d_src, size * sizeof(double), cudaMemcpyDeviceToHost));
}

void launchAddKernel(double* d_accumulator, const double* d_input, size_t numElements) {
    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize; // basically taking the ceiling
    
    addKernel<<<numBlocks, blockSize>>>(d_accumulator, d_input, numElements);
    
    CUDA_CHECK(cudaGetLastError()); // kernel error
    CUDA_CHECK(cudaDeviceSynchronize()); // w8 for GPU
}