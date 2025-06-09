#ifndef JNET_LAYERS_ACTIVATION_H
#define JNET_LAYERS_ACTIVATION_H

// JNet Neural Network Framework
// Activation functions - Neural network activation layers
//
// Provides functionality for:
// - Common activation functions (ReLU, Sigmoid, Tanh, Softmax)
// - Derivative computation for backpropagation
// - Efficient vectorized operations
// - Support for different activation types per layer

#include <vector>
#include "../core/tensor.h"

namespace JNet {

enum class Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Linear
};

class ActivationFunction {
public:
    static Tensor relu(const Tensor& input);
    static Tensor sigmoid(const Tensor& input);
    static Tensor tanh(const Tensor& input);
    static Tensor softmax(const Tensor& input);
    
    static Tensor relu_derivative(const Tensor& input);
    static Tensor sigmoid_derivative(const Tensor& input);
    static Tensor tanh_derivative(const Tensor& input);
    static Tensor softmax_derivative(const Tensor& input);
    
    static Tensor apply(const Tensor& input, Activation activation);
    static Tensor derivative(const Tensor& input, Activation activation);
};

}

#endif // JNET_LAYERS_ACTIVATION_H