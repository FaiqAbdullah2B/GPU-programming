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


__global__ void mulKernel(const double* A, const double* B, double* C, size_t rowsA, size_t colsA, size_t colsB) {
    size_t row = blockDim.y * blockIdx.y + threadIdx.y;
    size_t col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        double parSum = 0;
        for (size_t k = 0; k < colsA; k++) {
            parSum += A[row*colsA + k] * B[k*colsB + col];
        }
        C[row*colsB + col] = parSum;
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

void launchMulKernel(const double* A, const double* B, double* C, size_t rowsA, size_t colsA, size_t colsB) {
    dim3 threadsPerBlock(16, 16);

    dim3 numBlocks(
        (colsB + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y 
    );
    
    mulKernel<<<numBlocks, threadsPerBlock>>>(A, B, C, rowsA, colsA, colsB);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}