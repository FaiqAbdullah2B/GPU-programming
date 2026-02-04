cmake_minimum_required(VERSION 3.10)
project(MatrixAddGPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enforce strict warning checking
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(
        -Wall 
        -Wextra 
        -Wshadow
        -Wnon-virtual-dtor
    )
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall,-Wextra")

set(SOURCES
    src/main.cpp
    src/Matrix.cpp
    src/gpu_ops.cu
)

# Create executable
add_executable(matrix_add_gpu ${SOURCES})

# Include directories
target_include_directories(matrix_add_gpu PRIVATE include)

# Ensure we link against CUDA runtime
set_target_properties(matrix_add_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)