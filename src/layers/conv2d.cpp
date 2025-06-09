#include "conv2d.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "../optimizers/sgd.h"

namespace JNet {

Conv2D::Conv2D(int num_filters, int kernel_size, int stride, int padding, Activation activation)
    : input_channels(-1), num_filters(num_filters), kernel_size(kernel_size), 
      stride(stride), padding(padding), activation_func(activation), weights_initialized(false) {
}

Conv2D::Conv2D(int input_channels, int num_filters, int kernel_size, int stride, int padding, Activation activation)
    : input_channels(input_channels), num_filters(num_filters), kernel_size(kernel_size),
      stride(stride), padding(padding), activation_func(activation), weights_initialized(false) {
    initialize_weights(input_channels, 0, 0);
}

Conv2D::~Conv2D() {
    // Smart pointers will automatically clean up
}

void Conv2D::initialize_weights(int input_channels, int height, int width) {
    this->input_channels = input_channels;
    
    // Initialize weights using Xavier/Glorot initialization
    // Shape: [num_filters, input_channels, kernel_size, kernel_size]
    std::vector<int> weight_shape = {num_filters, input_channels, kernel_size, kernel_size};
    weights = Tensor(weight_shape);
    
    // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
    int fan_in = input_channels * kernel_size * kernel_size;
    int fan_out = num_filters * kernel_size * kernel_size;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, scale);
    
    for (int i = 0; i < weights.size(); ++i) {
        weights.at(i) = dis(gen);
    }
    
    // Initialize biases to zero
    std::vector<int> bias_shape = {num_filters};
    biases = Tensor::zeros(bias_shape);
    
    weights_initialized = true;
}

std::vector<int> Conv2D::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Input must be 3D: [channels, height, width]");
    }
    
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    return {num_filters, output_height, output_width};
}

Tensor Conv2D::apply_padding(const Tensor& input) const {
    if (padding == 0) {
        return input;
    }
    
    std::vector<int> input_shape = input.shape();
    int channels = input_shape[0];
    int height = input_shape[1];
    int width = input_shape[2];
    
    std::vector<int> padded_shape = {channels, height + 2 * padding, width + 2 * padding};
    Tensor padded = Tensor::zeros(padded_shape);
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                padded[{c, h + padding, w + padding}] = input[{c, h, w}];
            }
        }
    }
    
    return padded;
}

Tensor Conv2D::forward(const Tensor& input) {
    std::vector<int> input_shape = input.shape();
    
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Conv2D input must be 3D: [channels, height, width]");
    }
    
    // Initialize weights if not done yet
    if (!weights_initialized) {
        initialize_weights(input_shape[0], input_shape[1], input_shape[2]);
    }
    
    // Store input for backpropagation
    last_input = input;
    
    // Apply padding if necessary
    Tensor padded_input = apply_padding(input);
    std::vector<int> padded_shape = padded_input.shape();
    
    int input_channels = padded_shape[0];
    int input_height = padded_shape[1];
    int input_width = padded_shape[2];
    
    // Calculate output dimensions
    std::vector<int> output_shape = calculate_output_shape(input.shape());
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    
    // Create output tensor
    Tensor output = Tensor::zeros(output_shape);
    
    // Perform convolution
    for (int f = 0; f < num_filters; ++f) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                double sum = 0.0;
                
                // Convolve with kernel
                for (int c = 0; c < input_channels; ++c) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (ih < input_height && iw < input_width) {
                                sum += padded_input[{c, ih, iw}] * weights[{f, c, kh, kw}];
                            }
                        }
                    }
                }
                
                // Add bias and store
                output[{f, oh, ow}] = sum + biases.at(f);
            }
        }
    }
    
    // Apply activation
    for (int i = 0; i < output.size(); ++i) {
        switch (activation_func) {
            case Activation::ReLU:
                output.at(i) = std::max(0.0, output.at(i));
                break;
            case Activation::Sigmoid:
                output.at(i) = 1.0 / (1.0 + std::exp(-output.at(i)));
                break;
            case Activation::Tanh:
                output.at(i) = std::tanh(output.at(i));
                break;
            case Activation::Linear:
                // Do nothing
                break;
            case Activation::Softmax:
                // Softmax will be applied after the loop
                break;
        }
    }
    
    last_output = output;
    return output;
}

Tensor Conv2D::backward(const Tensor& output_gradient) {
    if (last_input.size() == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    std::vector<int> input_shape = last_input.shape();
    std::vector<int> output_shape = last_output.shape();
    
    // Apply activation derivative
    Tensor activation_grad = output_gradient;
    for (int i = 0; i < activation_grad.size(); ++i) {
        switch (activation_func) {
            case Activation::ReLU:
                activation_grad.at(i) = (last_output.at(i) > 0) ? activation_grad.at(i) : 0.0;
                break;
            case Activation::Sigmoid: {
                double sigmoid_out = last_output.at(i);
                activation_grad.at(i) *= sigmoid_out * (1.0 - sigmoid_out);
                break;
            }
            case Activation::Tanh: {
                double tanh_out = last_output.at(i);
                activation_grad.at(i) *= (1.0 - tanh_out * tanh_out);
                break;
            }
            case Activation::Linear:
                // Do nothing
                break;
            case Activation::Softmax:
                // Softmax derivative is more complex, handle separately
                break;
        }
    }
    
    // Calculate gradients
    Tensor weight_gradients = Tensor::zeros(weights.shape());
    Tensor bias_gradients = Tensor::zeros(biases.shape());
    Tensor input_gradients = Tensor::zeros(input_shape);
    
    // Apply padding to input for gradient calculation
    Tensor padded_input = apply_padding(last_input);
    std::vector<int> padded_shape = padded_input.shape();
    Tensor padded_input_gradients = Tensor::zeros(padded_shape);
    
    int input_channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    
    // Calculate gradients
    for (int f = 0; f < num_filters; ++f) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                double grad = activation_grad[{f, oh, ow}];
                
                // Bias gradient
                bias_gradients.at(f) += grad;
                
                // Weight and input gradients
                for (int c = 0; c < input_channels; ++c) {
                    for (int kh = 0; kh < kernel_size; ++kh) {
                        for (int kw = 0; kw < kernel_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (ih < padded_shape[1] && iw < padded_shape[2]) {
                                // Weight gradient
                                weight_gradients[{f, c, kh, kw}] += grad * padded_input[{c, ih, iw}];
                                
                                // Input gradient
                                padded_input_gradients[{c, ih, iw}] += grad * weights[{f, c, kh, kw}];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Remove padding from input gradients
    if (padding > 0) {
        for (int c = 0; c < input_channels; ++c) {
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    input_gradients[{c, h, w}] = padded_input_gradients[{c, h + padding, w + padding}];
                }
            }
        }
    } else {
        input_gradients = padded_input_gradients;
    }
    
    // Update weights and biases using optimizer
    if (weight_optimizer) {
        weight_optimizer->update(weights, weight_gradients);
        bias_optimizer->update(biases, bias_gradients);
    } else {
        // Default SGD with learning rate 0.01
        weights = weights - weight_gradients * 0.01;
        biases = biases - bias_gradients * 0.01;
    }
    
    return input_gradients;
}

void Conv2D::setOptimizer(std::shared_ptr<Optimizer> opt) {
    weight_optimizer = opt;
    bias_optimizer = opt;
}

void Conv2D::setOptimizerType(const std::string& type, double learning_rate) {
    if (type == "sgd") {
        weight_optimizer = std::make_shared<SGD>(learning_rate);
        bias_optimizer = std::make_shared<SGD>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer type: " + type);
    }
}

void Conv2D::set_weights(const Tensor& new_weights) {
    if (new_weights.shape() != weights.shape()) {
        throw std::invalid_argument("Weight shapes do not match");
    }
    weights = new_weights;
}

void Conv2D::set_biases(const Tensor& new_biases) {
    if (new_biases.shape() != biases.shape()) {
        throw std::invalid_argument("Bias shapes do not match");
    }
    biases = new_biases;
}

Tensor Conv2D::get_weights() const {
    return weights;
}

Tensor Conv2D::get_biases() const {
    return biases;
}

}
