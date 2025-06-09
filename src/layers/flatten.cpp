#include "flatten.h"
#include <stdexcept>

namespace JNet {

Flatten::Flatten() {
}

Flatten::~Flatten() {
}

Tensor Flatten::forward(const Tensor& input) {
    // Store input shape for backward pass
    input_shape = input.shape();
    
    // Calculate total size
    int total_size = 1;
    for (int dim : input_shape) {
        total_size *= dim;
    }
    
    // Create flattened tensor as 2D: [1, total_size] (batch_size=1, features=total_size)
    Tensor output({1, total_size});
    
    // Copy data
    for (int i = 0; i < total_size; ++i) {
        output.at(i) = input.at(i);
    }
    
    return output;
}

Tensor Flatten::backward(const Tensor& output_gradient) {
    if (input_shape.empty()) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    // Reshape gradient back to original input shape
    Tensor input_gradient(input_shape);
    
    // Copy data
    for (int i = 0; i < output_gradient.size(); ++i) {
        input_gradient.at(i) = output_gradient.at(i);
    }
    
    return input_gradient;
}

}
