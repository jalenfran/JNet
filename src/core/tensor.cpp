#include "tensor.h"
#include "threadpool.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <iomanip>
#include <thread>

namespace JNet {

// Constructors
Tensor::Tensor(const std::vector<int>& dimensions) : dimensions(dimensions) {
    total_size = calculateSize(dimensions);
    data.resize(total_size, 0.0);
}

Tensor::Tensor(const Tensor& other) : dimensions(other.dimensions), data(other.data), total_size(other.total_size) {
}

Tensor::Tensor() : total_size(0) {
}

Tensor::~Tensor() {
}

// Static factory methods
Tensor Tensor::zeros(const std::vector<int>& dimensions) {
    return Tensor(dimensions); // Constructor already initializes with zeros
}

Tensor Tensor::ones(const std::vector<int>& dimensions) {
    Tensor result(dimensions);
    result.fill(1.0);
    return result;
}

Tensor Tensor::random(const std::vector<int>& dimensions) {
    Tensor result(dimensions);
    result.fillRandom();
    return result;
}

// Assignment operator
Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        dimensions = other.dimensions;
        data = other.data;
        total_size = other.total_size;
    }
    return *this;
}

// Arithmetic operations
Tensor Tensor::operator+(const Tensor& other) const {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }
    Tensor result(dimensions);
    for (int i = 0; i < total_size; ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
    }
    Tensor result(dimensions);
    for (int i = 0; i < total_size; ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for element-wise multiplication.");
    }
    Tensor result(dimensions);
    for (int i = 0; i < total_size; ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Tensor Tensor::operator*(double scalar) const {
    Tensor result(dimensions);
    for (int i = 0; i < total_size; ++i) {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for addition.");
    }
    for (int i = 0; i < total_size; ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (dimensions != other.dimensions) {
        throw std::invalid_argument("Tensors must have the same dimensions for subtraction.");
    }
    for (int i = 0; i < total_size; ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(double scalar) {
    for (int i = 0; i < total_size; ++i) {
        data[i] *= scalar;
    }
    return *this;
}

// Element access
double& Tensor::operator[](const std::vector<int>& indices) {
    int index = calculateIndex(indices);
    return data[index];
}

const double& Tensor::operator[](const std::vector<int>& indices) const {
    int index = calculateIndex(indices);
    return data[index];
}

double& Tensor::at(int index) {
    if (index < 0 || index >= total_size) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

const double& Tensor::at(int index) const {
    if (index < 0 || index >= total_size) {
        throw std::out_of_range("Index out of range");
    }
    return data[index];
}

// Matrix operations
Tensor Tensor::dot(const Tensor& other) const {
    if (dimensions.size() != 2 || other.dimensions.size() != 2) {
        throw std::invalid_argument("Dot product requires 2D tensors (matrices)");
    }
    if (dimensions[1] != other.dimensions[0]) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    std::vector<int> result_dims = {dimensions[0], other.dimensions[1]};
    Tensor result(result_dims);
    
    int rows = dimensions[0];
    int cols = other.dimensions[1];
    int inner = dimensions[1];
    
    // Use multithreading for larger matrices
    size_t total_operations = static_cast<size_t>(rows) * cols;
    size_t thread_threshold = 100; // Threshold to decide if multithreading is worth it
    
    if (total_operations > thread_threshold) {
        size_t num_threads = std::min(static_cast<size_t>(rows), 
                                     static_cast<size_t>(std::thread::hardware_concurrency()));
        
        parallel_for(0, rows, num_threads, [&](size_t i) {
            for (int j = 0; j < cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < inner; ++k) {
                    sum += data[i * inner + k] * other.data[k * cols + j];
                }
                result.data[i * cols + j] = sum;
            }
        });
    } else {
        // Use single-threaded approach for smaller matrices
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double sum = 0.0;
                for (int k = 0; k < inner; ++k) {
                    sum += data[i * inner + k] * other.data[k * cols + j];
                }
                result.data[i * cols + j] = sum;
            }
        }
    }
    
    return result;
}

Tensor Tensor::transpose() const {
    if (dimensions.size() != 2) {
        throw std::invalid_argument("Transpose only supported for 2D tensors");
    }
    
    std::vector<int> result_dims = {dimensions[1], dimensions[0]};
    Tensor result(result_dims);
    
    for (int i = 0; i < dimensions[0]; ++i) {
        for (int j = 0; j < dimensions[1]; ++j) {
            result.data[j * dimensions[0] + i] = data[i * dimensions[1] + j];
        }
    }
    return result;
}

Tensor Tensor::sum(int axis) const {
    if (axis == -1) {
        // Sum all elements
        Tensor result({1});
        double total = 0.0;
        for (double val : data) {
            total += val;
        }
        result.data[0] = total;
        return result;
    }
    
    if (dimensions.size() != 2) {
        throw std::invalid_argument("Axis-specific sum only supported for 2D tensors currently");
    }
    
    if (axis == 0) {
        // Sum along rows (result has shape [1, cols])
        std::vector<int> result_dims = {1, dimensions[1]};
        Tensor result(result_dims);
        
        for (int j = 0; j < dimensions[1]; ++j) {
            double sum = 0.0;
            for (int i = 0; i < dimensions[0]; ++i) {
                sum += data[i * dimensions[1] + j];
            }
            result.data[j] = sum;
        }
        return result;
    }
    
    if (axis == 1) {
        // Sum along columns (result has shape [rows, 1])
        std::vector<int> result_dims = {dimensions[0], 1};
        Tensor result(result_dims);
        
        for (int i = 0; i < dimensions[0]; ++i) {
            double sum = 0.0;
            for (int j = 0; j < dimensions[1]; ++j) {
                sum += data[i * dimensions[1] + j];
            }
            result.data[i] = sum;
        }
        return result;
    }
    
    throw std::invalid_argument("Invalid axis for sum operation");
}

// Utility methods
std::vector<int> Tensor::shape() const {
    return dimensions;
}

int Tensor::size() const {
    return total_size;
}

void Tensor::reshape(const std::vector<int>& new_dimensions) {
    int new_size = calculateSize(new_dimensions);
    if (new_size != total_size) {
        throw std::invalid_argument("Total size must remain the same for reshape.");
    }
    dimensions = new_dimensions;
}

void Tensor::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

void Tensor::fillRandom() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dis(-1.0, 1.0);
    
    for (double& val : data) {
        val = dis(gen);
    }
}

void Tensor::print() const {
    std::cout << *this << std::endl;
}

// Stream output
std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
    os << "Tensor(shape=[";
    for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
        os << tensor.dimensions[i];
        if (i < tensor.dimensions.size() - 1) os << ", ";
    }
    os << "], data=[";
    
    // Limit output for large tensors
    int max_elements = 10;
    int elements_to_show = std::min(max_elements, tensor.total_size);
    
    for (int i = 0; i < elements_to_show; ++i) {
        os << std::fixed << std::setprecision(4) << tensor.data[i];
        if (i < elements_to_show - 1) os << ", ";
    }
    
    if (tensor.total_size > max_elements) {
        os << "...";
    }
    os << "])";
    return os;
}

// Friend function for scalar multiplication
Tensor operator*(double scalar, const Tensor& tensor) {
    return tensor * scalar;
}

// Private helper methods
int Tensor::calculateSize(const std::vector<int>& dimensions) const {
    int size = 1;
    for (int dim : dimensions) {
        size *= dim;
    }
    return size;
}

int Tensor::calculateIndex(const std::vector<int>& indices) const {
    if (indices.size() != dimensions.size()) {
        throw std::invalid_argument("Number of indices must match the number of dimensions.");
    }
    
    int index = 0;
    int stride = 1;
    for (int i = dimensions.size() - 1; i >= 0; --i) {
        if (indices[i] < 0 || indices[i] >= dimensions[i]) {
            throw std::out_of_range("Index out of range for dimension " + std::to_string(i));
        }
        index += indices[i] * stride;
        stride *= dimensions[i];
    }
    return index;
}

// Fast index calculation methods for performance
int Tensor::fast_index(const std::vector<int>& indices) const {
    int index = 0;
    int stride = 1;
    for (int i = dimensions.size() - 1; i >= 0; --i) {
        index += indices[i] * stride;
        stride *= dimensions[i];
    }
    return index;
}

int Tensor::fast_index_3d(int c, int h, int w) const {
    // For shape [channels, height, width]
    return c * dimensions[1] * dimensions[2] + h * dimensions[2] + w;
}

int Tensor::fast_index_4d(int f, int c, int h, int w) const {
    // For shape [filters, channels, height, width]
    return f * dimensions[1] * dimensions[2] * dimensions[3] + 
           c * dimensions[2] * dimensions[3] + 
           h * dimensions[3] + w;
}

}