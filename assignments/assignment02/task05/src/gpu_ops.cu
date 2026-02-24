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


__global__ void mulKernelTiled(const double* A, const double* B, double* C, size_t tileSize, size_t rowsA, size_t colsA, size_t colsB) {

    extern __shared__ double sharedMem[]; 
    double* tileA = &sharedMem[0];                    
    double* tileB = &sharedMem[tileSize * tileSize];

    size_t tx = threadIdx.x;
    size_t ty = threadIdx.y;
    size_t bx = blockIdx.x;
    size_t by = blockIdx.y;

    size_t row = by * tileSize + ty;
    size_t col = bx * tileSize + tx;

    // Accumulator
    double val = 0.0;

    size_t numTiles = (colsA + tileSize - 1) / tileSize;

    for (size_t m = 0; m < numTiles; ++m) {
        
        size_t colA = m * tileSize + tx;
        if (row < rowsA && colA < colsA) {
            tileA[ty * tileSize + tx] = A[row * colsA + colA];
        } else {
            tileA[ty * tileSize + tx] = 0.0; 
        }

        size_t rowB = m * tileSize + ty;
        if (rowB < colsA && col < colsB) { // Note: rowsB == colsA
            tileB[ty * tileSize + tx] = B[rowB * colsB + col];
        } else {
            tileB[ty * tileSize + tx] = 0.0;
        }

        __syncthreads();

        for (size_t k = 0; k < tileSize; ++k) {
            val += tileA[ty * tileSize + k] * tileB[k * tileSize + tx];
        }

        __syncthreads();
    }

    if (row < rowsA && col < colsB) {
        C[row * colsB + col] = val;
    }
}

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

void launchMulKernelTiled(const double* A, const double* B, double* C, size_t tileSize, size_t rowsA, size_t colsA, size_t colsB) {
    dim3 threadsPerBlock(tileSize, tileSize);

    dim3 numBlocks(
        (colsB + threadsPerBlock.x - 1) / threadsPerBlock.x, 
        (rowsA + threadsPerBlock.y - 1) / threadsPerBlock.y 
    );

    size_t sharedMemSize = 2 * tileSize * tileSize * sizeof(double);
    
    mulKernelTiled<<<numBlocks, threadsPerBlock, sharedMemSize>>>(A, B, C, tileSize, rowsA, colsA, colsB);
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}