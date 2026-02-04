#pragma once

#include <cstddef>

void allocateDeviceMemory(double** d_ptr, size_t size);

void freeDeviceMemory(double* d_ptr);

void copyToDevice(double* d_dest, const double* h_src, size_t size);

void copyToHost(double* h_dest, const double* d_src, size_t size);

void launchAddKernel(double* d_accumulator, const double* d_input, size_t numElements);
