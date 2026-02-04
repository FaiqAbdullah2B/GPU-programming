#include <iostream>
#include <fstream>
#include <string>
#include <vector>

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

    Matrix result = Matrix::loadFromFile(inputFile);

    for (int i = 1; i < numMatrices; i++) {
        Matrix current = Matrix::loadFromFile(inputFile);
        result = result + current;
    }

    inputFile.close();

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