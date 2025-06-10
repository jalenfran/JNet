#ifndef JNET_CORE_TENSOR_H
#define JNET_CORE_TENSOR_H

// JNet Neural Network Framework
// Tensor class - Multi-dimensional array operations for neural networks
// 
// Provides efficient tensor operations including:
// - Basic arithmetic operations
// - Matrix multiplication and linear algebra
// - Element access and manipulation
// - Memory management with proper copy semantics

#include <vector>
#include <iostream>

namespace JNet {

class Tensor {
public:
    // Constructors
    Tensor(const std::vector<int>& dimensions);
    Tensor(const Tensor& other);
    Tensor();
    ~Tensor();

    // Static factory methods
    static Tensor zeros(const std::vector<int>& dimensions);
    static Tensor ones(const std::vector<int>& dimensions);
    static Tensor random(const std::vector<int>& dimensions);

    // Basic operations
    Tensor& operator=(const Tensor& other);
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(double scalar) const;
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(double scalar);
    
    // Element access
    double& operator[](const std::vector<int>& indices);
    const double& operator[](const std::vector<int>& indices) const;
    double& at(int index);
    const double& at(int index) const;
    
    // Direct data access for performance (use with caution)
    double* data_ptr() { return data.data(); }
    const double* data_ptr() const { return data.data(); }
    
    // Fast index calculation without bounds checking (for performance)
    int fast_index(const std::vector<int>& indices) const;
    int fast_index_3d(int c, int h, int w) const; // Optimized for 3D access
    int fast_index_4d(int f, int c, int h, int w) const; // Optimized for 4D access
    
    // Matrix operations
    Tensor dot(const Tensor& other) const;
    Tensor matmul(const Tensor& other) const { return dot(other); }
    Tensor transpose() const;
    Tensor sum(int axis = -1) const;
    
    // Utility methods
    std::vector<int> shape() const;
    int size() const;
    void reshape(const std::vector<int>& new_dimensions);
    void fill(double value);
    void fillRandom();
    void print() const;
    
    // Stream output
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
    friend Tensor operator*(double scalar, const Tensor& tensor);

private:
    std::vector<int> dimensions;
    std::vector<double> data;
    int total_size;

    void allocateData();
    int calculateSize(const std::vector<int>& dimensions) const;
    int calculateIndex(const std::vector<int>& indices) const;
};

}

#endif // JNET_CORE_TENSOR_H