cmake_minimum_required(VERSION 3.10)
project(MatrixBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

set(CMAKE_CUDA_ARCHITECTURES 86)

# Source Definitions
set(COMMON_SRC src/Matrix.cpp)
set(GPU_OPS src/gpu_ops.cu)

# Executables
add_executable(gpu_tiled_solver src/main_gpu_tiled.cpp ${COMMON_SRC} ${GPU_OPS})
add_executable(gpu_solver src/main_gpu.cpp ${COMMON_SRC} ${GPU_OPS})

# Link CUDA
set_target_properties(gpu_tiled_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  