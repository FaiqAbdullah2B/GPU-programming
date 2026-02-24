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

    // ********** CUDA CHAIN MULTIPLICATION START ********** //

    Matrix firstMatrix = Matrix::loadFromFile(inputFile);
    size_t currentRows = firstMatrix.getRows();
    size_t currentCols = firstMatrix.getCols();

    double* d_left = nullptr; // Pointer for the matrix on the Left side of the multiplication
    allocateDeviceMemory(&d_left, currentRows * currentCols);
    copyToDevice(d_left, firstMatrix.getRawData(), currentRows * currentCols);

    for (int i = 1; i < numMatrices; ++i) {
        Matrix nextMatrix = Matrix::loadFromFile(inputFile);
        
        if (currentCols != nextMatrix.getRows()) {
            std::cerr << "Error: Dimension mismatch at matrix " << i + 1 
                      << ". Left Cols (" << currentCols << ") != Right Rows (" 
                      << nextMatrix.getRows() << ")." << std::endl;
            freeDeviceMemory(d_left);
            return 1;
        }

        // Setup the Right side operand
        size_t nextRows = nextMatrix.getRows();
        size_t nextCols = nextMatrix.getCols();
        double* d_right = nullptr;
        allocateDeviceMemory(&d_right, nextRows * nextCols);
        copyToDevice(d_right, nextMatrix.getRawData(), nextRows * nextCols);

        // Setup the Result (Dimensions will be Left Rows x Right Cols)
        double* d_result = nullptr;
        allocateDeviceMemory(&d_result, currentRows * nextCols);

        // launchMulKernel(A, B, C, rowsA, colsA, colsB)
        launchMulKernel(d_left, d_right, d_result, currentRows, currentCols, nextCols);

        // Cleanup inputs
        freeDeviceMemory(d_left);
        freeDeviceMemory(d_right);

        // The Result becomes the new "Left" for the next iteration
        d_left = d_result;
        currentCols = nextCols; // currentRows remains the same (The rows of the very first matrix persist)
    }

    inputFile.close();

    // 4. Retrieve Final Result
    size_t totalElements = currentRows * currentCols;
    std::vector<double> resultData(totalElements);
    copyToHost(resultData.data(), d_left, totalElements);

    freeDeviceMemory(d_left);

    // ********** CUDA END ********** //

    Matrix result(currentRows, currentCols, resultData);
    
    if(!outputFilename.empty()) {
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            std::cerr << "Error: Output file could not be opened" << std::endl;
            return 1;
        }
        
        outputFile << "1\n"; 
        result.write(outputFile);
    } else {
        std::cout << "Result Matrix (" << currentRows << "x" << currentCols << "): \n";
        result.write(std::cout);
    }

    return 0;
}