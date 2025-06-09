#ifndef JNET_LAYERS_CONV2D_H
#define JNET_LAYERS_CONV2D_H

// JNet Neural Network Framework
// Conv2D layer - 2D Convolutional neural network layer
//
// Provides functionality for:
// - 2D convolutional layer implementation
// - Multiple filters with configurable kernel size
// - Stride and padding support
// - Forward and backward propagation
// - Automatic gradient computation and weight updates

#include "../core/tensor.h"
#include "activation.h"
#include "../optimizers/optimizer.h"
#include "layer.h"
#include <memory>

namespace JNet {

class Conv2D : public Layer {
public:
    // Constructor with all parameters
    Conv2D(int num_filters, int kernel_size, int stride = 1, int padding = 0, 
           Activation activation = Activation::ReLU);
    
    // Constructor that infers input channels from first forward pass
    Conv2D(int input_channels, int num_filters, int kernel_size, int stride = 1, 
           int padding = 0, Activation activation = Activation::ReLU);
    
    ~Conv2D() override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    
    void set_weights(const Tensor& weights);
    void set_biases(const Tensor& biases);
    Tensor get_weights() const;
    Tensor get_biases() const;
    
    void initialize_weights(int input_channels, int height, int width);
    void setOptimizer(std::shared_ptr<Optimizer> opt) override;
    void setOptimizerType(const std::string& type, double learning_rate = 0.01) override;
    
    // Utility methods
    std::vector<int> calculate_output_shape(const std::vector<int>& input_shape) const;
    bool hasWeights() const override { return true; }
    bool hasBiases() const override { return true; }

private:
    // Layer parameters
    int input_channels;
    int num_filters;
    int kernel_size;
    int stride;
    int padding;
    
    // Weights and biases
    Tensor weights;  // Shape: [num_filters, input_channels, kernel_size, kernel_size]
    Tensor biases;   // Shape: [num_filters]
    
    // Storage for backpropagation
    Tensor last_input;
    Tensor last_output;
    
    // Configuration
    Activation activation_func;
    bool weights_initialized;
    
    // Optimizers
    std::shared_ptr<Optimizer> weight_optimizer;
    std::shared_ptr<Optimizer> bias_optimizer;
    
    // Helper methods
    Tensor im2col(const Tensor& input) const;
    Tensor col2im(const Tensor& col, const std::vector<int>& input_shape) const;
    Tensor apply_convolution(const Tensor& input) const;
    Tensor apply_padding(const Tensor& input) const;
    Tensor remove_padding(const Tensor& input) const;
};

}

#endif // JNET_LAYERS_CONV2D_H
