#pragma once

#include <vector>
#include <string>
#include <iostream>

class Matrix {
public:

    Matrix() = default;
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, const std::vector<double>& data);

    static Matrix loadFromFile(std::istream& inputStream);
    void write(std::ostream& outputStream) const;
    
    Matrix operator+(const Matrix& other) const;

    size_t getRows() const { return rows_; }
    size_t getCols() const { return cols_; }
    double* getRawData() { return data_.data(); } // Helper for CUDA copy


    const std::vector<double>& getData() const { return data_; }

private:
    size_t rows_ = 0;
    size_t cols_ = 0;
    std::vector<double> data_;
};

