#include "dense.h"
#include "../optimizers/sgd.h"
#include "../optimizers/adam.h"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <memory>

namespace JNet {

Dense::Dense(int output_size, Activation activation) 
    : output_size(output_size), activation_func(activation), weights_initialized(false), input_size(0), 
      weight_optimizer(nullptr), bias_optimizer(nullptr) {
}

Dense::Dense(int input_size, int output_size, Activation activation) 
    : input_size(input_size), output_size(output_size), activation_func(activation), weights_initialized(false),
      weight_optimizer(nullptr), bias_optimizer(nullptr) {
    initialize_weights(input_size);
}

Dense::~Dense() {
}

void Dense::initialize_weights(int input_size) {
    this->input_size = input_size;
    
    // Xavier/Glorot initialization
    double limit = std::sqrt(6.0 / (input_size + output_size));
    weights = Tensor::random({output_size, input_size});
    
    // Scale weights to [-limit, limit]
    for (int i = 0; i < weights.size(); ++i) {
        weights.at(i) *= limit;
    }
    
    biases = Tensor::zeros({output_size, 1});
    weights_initialized = true;
}

Tensor Dense::forward(const Tensor& input) {
    // Auto-initialize weights if not done yet
    if (!weights_initialized) {
        if (input.shape().size() != 2) {
            throw std::invalid_argument("Input must be 2D tensor (batch_size, features)");
        }
        initialize_weights(input.shape()[1]);
    }
    
    if (input.shape()[1] != input_size) {
        throw std::invalid_argument("Input size does not match layer input size.");
    }
    
    // Store input for backprop
    last_input = input;
    
    // Linear transformation: input * weights.T + biases
    Tensor linear_output = input.dot(weights.transpose());
    
    // Add biases (broadcast)
    for (int i = 0; i < linear_output.shape()[0]; ++i) {
        for (int j = 0; j < linear_output.shape()[1]; ++j) {
            linear_output.at(i * linear_output.shape()[1] + j) += biases.at(j);
        }
    }
    
    // Apply activation function
    last_output = ActivationFunction::apply(linear_output, activation_func);
    return last_output;
}

Tensor Dense::backward(const Tensor& grad_output) {
    if (!weights_initialized) {
        throw std::runtime_error("Cannot backpropagate through uninitialized layer");
    }
    
    // Apply activation derivative
    Tensor activation_grad = ActivationFunction::derivative(last_output, activation_func);
    Tensor grad_linear = grad_output;
    
    // Element-wise multiply with activation derivative
    for (int i = 0; i < grad_linear.size(); ++i) {
        grad_linear.at(i) *= activation_grad.at(i);
    }
    
    // Compute gradients
    Tensor grad_weights = grad_linear.transpose().dot(last_input);
    Tensor grad_biases_temp = grad_linear.sum(0);
    
    // Reshape grad_biases to match biases shape [output_size, 1]
    Tensor grad_biases({output_size, 1});
    for (int i = 0; i < output_size; ++i) {
        grad_biases.at(i) = grad_biases_temp.at(i);
    }
    
    // Compute input gradient for next layer
    Tensor grad_input = grad_linear.dot(weights);
    
    // Update weights and biases using optimizer
    if (weight_optimizer) {
        weight_optimizer->update(weights, grad_weights);
        bias_optimizer->update(biases, grad_biases);
    } else {
        // Fallback to simple gradient descent if no optimizer is set
        double learning_rate = 0.01;
        for (int i = 0; i < weights.size(); ++i) {
            weights.at(i) -= grad_weights.at(i) * learning_rate;
        }
        for (int i = 0; i < biases.size(); ++i) {
            biases.at(i) -= grad_biases.at(i) * learning_rate;
        }
    }
    
    return grad_input;
}

void Dense::set_weights(const Tensor& new_weights) {
    weights = new_weights;
    weights_initialized = true;
}

void Dense::set_biases(const Tensor& new_biases) {
    biases = new_biases;
}

Tensor Dense::get_weights() const {
    return weights;
}

Tensor Dense::get_biases() const {
    return biases;
}

void Dense::setOptimizer(std::shared_ptr<Optimizer> opt) {
    weight_optimizer = opt;
    
    // Create a separate optimizer instance for biases to avoid state conflicts
    if (auto sgd = std::dynamic_pointer_cast<SGD>(opt)) {
        bias_optimizer = std::make_shared<SGD>(sgd->getLearningRate());
    } else if (auto adam = std::dynamic_pointer_cast<Adam>(opt)) {
        // Create new Adam with same parameters but separate state
        bias_optimizer = std::make_shared<Adam>(adam->getLearningRate(), adam->getBeta1(), 
                                               adam->getBeta2(), adam->getEpsilon());
    } else {
        // For unknown optimizer types, use the same instance
        bias_optimizer = opt;
    }
}

void Dense::setOptimizerType(const std::string& type, double learning_rate) {
    if (type == "sgd") {
        weight_optimizer = std::make_shared<SGD>(learning_rate);
        bias_optimizer = std::make_shared<SGD>(learning_rate);
    } else if (type == "adam") {
        weight_optimizer = std::make_shared<Adam>(learning_rate);
        bias_optimizer = std::make_shared<Adam>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer type: " + type);
    }
}

}