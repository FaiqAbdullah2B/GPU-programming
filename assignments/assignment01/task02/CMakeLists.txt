cmake_minimum_required(VERSION 3.10)
project(MatrixAdd VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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

# Define source files
set(SOURCES
    src/main.cpp
    src/Matrix.cpp
)

# Create executable
add_executable(matrix_add ${SOURCES})

# Include directories
target_include_directories(matrix_add PRIVATE include)