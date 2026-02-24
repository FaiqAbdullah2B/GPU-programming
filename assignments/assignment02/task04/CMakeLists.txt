cmake_minimum_required(VERSION 3.10)
project(MatrixBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include_directories(include)

# Source Definitions
set(COMMON_SRC src/Matrix.cpp)
set(GPU_OPS src/gpu_ops.cu)

# Executables
add_executable(cpu_solver src/main_cpu.cpp ${COMMON_SRC})
add_executable(gpu_solver src/main_gpu.cpp ${COMMON_SRC} ${GPU_OPS})

# Link CUDA
target_link_libraries(gpu_solver PRIVATE cuda)
set_target_properties(gpu_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  