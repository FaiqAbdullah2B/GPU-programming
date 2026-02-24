cmake_minimum_required(VERSION 3.10)
project(A2T1 LANGUAGES CXX CUDA)

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
    src/main.cu
)

# Create executable
add_executable(A2T1 ${SOURCES})

# Include directories
target_include_directories(A2T1 PRIVATE include)

# Ensure we link against CUDA runtime
set_target_properties(A2T1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)