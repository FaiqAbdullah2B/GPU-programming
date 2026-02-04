#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

#include "Matrix.h"

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


    // Loading all data into RAM first to exclude disk IO time
    std::vector<Matrix> matrices;
    matrices.reserve(numMatrices);
    for(int i=0; i<numMatrices; ++i) {
        matrices.push_back(Matrix::loadFromFile(inputFile));
    }
    inputFile.close();
    
    // * * * START TIMER * * * //
    auto start = std::chrono::high_resolution_clock::now();

    Matrix result = matrices[0];
    for (size_t i = 1; i < matrices.size(); i++) {
        result = result + matrices[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    // * * * END TIMER * * * //

    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "TIME_MS: " << duration.count() << std::endl;

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