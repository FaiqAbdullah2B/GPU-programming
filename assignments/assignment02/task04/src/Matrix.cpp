#include "Matrix.h"
#include <stdexcept>
#include <iomanip>

Matrix::Matrix(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(rows * cols) {}

Matrix::Matrix(size_t rows, size_t cols, const std::vector<double>& data)
    : rows_(rows), cols_(cols), data_(data) {
    if (data.size() != rows * cols) {
        throw std::invalid_argument("Data size does not match matrix dimensions.");
    }
}

Matrix Matrix::loadFromFile(std::istream& inputStream) {
    size_t r, c;
    if (!(inputStream >> r >> c)) {
        throw std::runtime_error("Failed to read matrix dimensions.");
    }

    std::vector<double> data;
    data.reserve(r * c);
    double val;
    for (size_t i = 0; i < r * c; ++i) {
        if (!(inputStream >> val)) {
            throw std::runtime_error("Failed to read matrix data element.");
        }
        data.push_back(val);
    }

    return Matrix(r, c, data);
}

void Matrix::write(std::ostream& outputStream) const {
    outputStream << rows_ << " " << cols_ << "\n";
    outputStream << std::fixed << std::setprecision(2);
    
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            outputStream << data_[i * cols_ + j] << (j == cols_ - 1 ? "" : " ");
        }
        outputStream << "\n";
    }
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }

    std::vector<double> resultData(data_.size());
    for (size_t i = 0; i < data_.size(); ++i) {
        resultData[i] = data_[i] + other.data_[i];
    }

    return Matrix(rows_, cols_, resultData);
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols_ != other.rows_) {
        throw std::invalid_argument("First matrix Columns should be equal to the seconds rows for multiplication");
    }

    std::vector<double> resultData(rows_ * other.cols_);

    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < other.cols_; ++j) {
            double sum = 0;
            for(size_t k = 0; k < cols_; ++k) {
                sum += data_[i*cols_ + k] * other.data_[k*other.cols_ + j];
            }
            resultData[i*other.cols_ + j] = sum;
        }
    }

    return Matrix(rows_, other.cols_, resultData);
}
