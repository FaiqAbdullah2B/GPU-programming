#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "Matrix.h"
#include "gpu_ops.cuh"

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]" << std::endl;
        return 1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = (argc == 3) ? argv [2] : "";

    std::ifstream inputFile(inputFilename);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Input file could not be opened: " << inputFilename << std::endl;
        return 1;
    }

    int numMatrices = 0;
    // Reading number of matrices which is specified at the top of the input file
    if (!(inputFile >> numMatrices)) {
        std::cerr << "Error: Could not read number of matrices from the input file" << std::endl;
        return 1;
    }

    if (numMatrices < 2) {
        std::cerr << "Error: Number of matrices in input file must be greater than 1" << std::endl;
        return 1;
    }

    // ********** CUDAAAAAAAA ********** // 

    Matrix firstMatrix = Matrix::loadFromFile(inputFile);
    size_t totalElements = firstMatrix.getRows() * firstMatrix.getCols();
    size_t rows = firstMatrix.getRows();
    size_t cols = firstMatrix.getCols();

    double* d_accumulator = nullptr;
    double* d_temp = nullptr;

    allocateDeviceMemory(&d_accumulator, totalElements);
    allocateDeviceMemory(&d_temp, totalElements);

    // Copy first matrix
    copyToDevice(d_accumulator, firstMatrix.getRawData(), totalElements);

    for (int i = 1; i < numMatrices; ++i) {
        Matrix current = Matrix::loadFromFile(inputFile);
        
        if (current.getRows() != rows || current.getCols() != cols) {
            std::cerr << "Error: Matrix dimension mismatch." << std::endl;
            freeDeviceMemory(d_accumulator);
            freeDeviceMemory(d_temp);
            return 1;
        }

        // Copy current to Temp
        copyToDevice(d_temp, current.getRawData(), totalElements);

        // Accumulate on GPU
        launchAddKernel(d_accumulator, d_temp, totalElements);
    }

    inputFile.close();

    std::vector<double> resultData(totalElements);
    copyToHost(resultData.data(), d_accumulator, totalElements);

    freeDeviceMemory(d_accumulator);
    freeDeviceMemory(d_temp);

    // ********** CUDAAAAAAAA is END ********** // 

    Matrix result(rows, cols, resultData);
    
    if(!outputFilename.empty()) {
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            std::cerr << "Error: Output file could not be opened" << std::endl;
            return 1;
        }

        outputFile << "1\n";
        result.write(outputFile);
    } else {
        std::cout << "Result Matrix: \n";
        result.write(std::cout);
    }

    return 0;
}