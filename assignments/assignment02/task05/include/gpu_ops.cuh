#pragma once

#include <cstddef>

void allocateDeviceMemory(double** d_ptr, size_t size);

void freeDeviceMemory(double* d_ptr);

void copyToDevice(double* d_dest, const double* h_src, size_t size);

void copyToHost(double* h_dest, const double* d_src, size_t size);

void launchMulKernel(const double* A,const double* B, double* C, size_t rowsA, size_t colsA, size_t colsB);

void launchMulKernelTiled(const double* A, const double* B, double* C, size_t tileSize, size_t rowsA, size_t colsA, size_t colsB);
