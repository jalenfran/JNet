#include "activation.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace JNet {

// ReLU activation
Tensor ActivationFunction::relu(const Tensor& input) {
    Tensor result = input;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = std::max(0.0, result.at(i));
    }
    return result;
}

// Sigmoid activation
Tensor ActivationFunction::sigmoid(const Tensor& input) {
    Tensor result = input;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = 1.0 / (1.0 + std::exp(-result.at(i)));
    }
    return result;
}

// Tanh activation
Tensor ActivationFunction::tanh(const Tensor& input) {
    Tensor result = input;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = std::tanh(result.at(i));
    }
    return result;
}

// Softmax activation
Tensor ActivationFunction::softmax(const Tensor& input) {
    Tensor result = input;
    
    // Find max value for numerical stability
    double max_val = result.at(0);
    for (int i = 1; i < result.size(); ++i) {
        max_val = std::max(max_val, result.at(i));
    }
    
    // Compute exponentials and sum
    double sum = 0.0;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = std::exp(result.at(i) - max_val);
        sum += result.at(i);
    }
    
    // Normalize
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) /= sum;
    }
    
    return result;
}

// ReLU derivative
Tensor ActivationFunction::relu_derivative(const Tensor& input) {
    Tensor result = input;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = (result.at(i) > 0.0) ? 1.0 : 0.0;
    }
    return result;
}

// Sigmoid derivative
Tensor ActivationFunction::sigmoid_derivative(const Tensor& input) {
    Tensor sig = sigmoid(input);
    Tensor result = sig;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = sig.at(i) * (1.0 - sig.at(i));
    }
    return result;
}

// Tanh derivative
Tensor ActivationFunction::tanh_derivative(const Tensor& input) {
    Tensor tanh_val = tanh(input);
    Tensor result = tanh_val;
    for (int i = 0; i < result.size(); ++i) {
        result.at(i) = 1.0 - std::pow(tanh_val.at(i), 2);
    }
    return result;
}

// Softmax derivative
Tensor ActivationFunction::softmax_derivative(const Tensor& input) {
    Tensor softmax_output = softmax(input);
    int n = softmax_output.size();
    
    // For vector input, return Jacobian matrix [n x n]
    Tensor jacobian({n, n});
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                // Diagonal: ∂s_i/∂x_i = s_i * (1 - s_i)
                jacobian.at(i * n + j) = softmax_output.at(i) * (1.0 - softmax_output.at(i));
            } else {
                // Off-diagonal: ∂s_i/∂x_j = -s_i * s_j
                jacobian.at(i * n + j) = -softmax_output.at(i) * softmax_output.at(j);
            }
        }
    }
    
    return jacobian;
}

// Apply activation function
Tensor ActivationFunction::apply(const Tensor& input, Activation activation) {
    switch (activation) {
        case Activation::ReLU:
            return relu(input);
        case Activation::Sigmoid:
            return sigmoid(input);
        case Activation::Tanh:
            return tanh(input);
        case Activation::Softmax:
            return softmax(input);
        case Activation::Linear:
            return input;
        default:
            throw std::invalid_argument("Unknown activation function");
    }
}

// Apply activation derivative
Tensor ActivationFunction::derivative(const Tensor& input, Activation activation) {
    switch (activation) {
        case Activation::ReLU:
            return relu_derivative(input);
        case Activation::Sigmoid:
            return sigmoid_derivative(input);
        case Activation::Tanh:
            return tanh_derivative(input);
        case Activation::Softmax:
            return softmax(input);
        case Activation::Linear:
            return Tensor::ones(input.shape());
        default:
            throw std::invalid_argument("Unknown derivative for activation function");
    }
}

}