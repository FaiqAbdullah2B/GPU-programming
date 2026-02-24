#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "Matrix.h"
#include "gpu_ops.cuh"

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <input_file> [output_file]" << std::endl;
        return 1;
    }

    size_t tileSize = static_cast<size_t>(std::stoull(argv[1]));
    std::string inputFilename = argv[2];
    std::string outputFilename = (argc == 4) ? argv [3] : "";
    

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

    // --- PHASE 1: PRE-LOAD (Exclude Disk I/O from timing) ---
    std::vector<Matrix> hostMatrices;
    hostMatrices.reserve(numMatrices);

    for (int i = 0; i < numMatrices; ++i) {
        hostMatrices.push_back(Matrix::loadFromFile(inputFile));
    }
    inputFile.close();

    // ********** CUDA CHAIN MULTIPLICATION START ********** //
    // Timing start //

    auto start_time = std::chrono::high_resolution_clock::now();

    size_t currentRows = hostMatrices[0].getRows();
    size_t currentCols = hostMatrices[0].getCols();

    double* d_left = nullptr; // Pointer for the matrix on the Left side of the multiplication
    allocateDeviceMemory(&d_left, currentRows * currentCols);
    copyToDevice(d_left, hostMatrices[0].getRawData(), currentRows * currentCols);

    for (int i = 1; i < numMatrices; ++i) {
        Matrix nextMatrix = hostMatrices[i];
        
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
        launchMulKernelTiled(d_left, d_right, d_result, tileSize, currentRows, currentCols, nextCols);

        // Cleanup inputs
        freeDeviceMemory(d_left);
        freeDeviceMemory(d_right);

        // The Result becomes the new "Left" for the next iteration
        d_left = d_result;
        currentCols = nextCols; // currentRows remains the same (The rows of the very first matrix persist)
    }

    // 4. Retrieve Final Result
    size_t totalElements = currentRows * currentCols;
    std::vector<double> resultData(totalElements);
    copyToHost(resultData.data(), d_left, totalElements);

    freeDeviceMemory(d_left);

    // ********** CUDA END ********** //

    // Timing End // 
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> execution_time = end_time - start_time;

    std::cout << "TIME_MS: " << execution_time.count() << std::endl;
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