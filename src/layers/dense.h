#ifndef JNET_LAYERS_DENSE_H
#define JNET_LAYERS_DENSE_H

// JNet Neural Network Framework
// Dense layer - Fully connected neural network layer
//
// Provides functionality for:
// - Dense/fully-connected layer implementation
// - Xavier/Glorot weight initialization
// - Forward and backward propagation
// - Automatic gradient computation and weight updates

#include "../core/tensor.h"
#include "activation.h"
#include "../optimizers/optimizer.h"
#include "layer.h"
#include <memory>

namespace JNet {

class Dense : public Layer {
public:
    Dense(int output_size, Activation activation = Activation::ReLU);
    Dense(int input_size, int output_size, Activation activation = Activation::ReLU);
    ~Dense() override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    void set_weights(const Tensor& weights);
    void set_biases(const Tensor& biases);
    Tensor get_weights() const;
    Tensor get_biases() const;
    void initialize_weights(int input_size);
    void setOptimizer(std::shared_ptr<Optimizer> opt) override;
    void setOptimizerType(const std::string& type, double learning_rate = 0.01) override;
    
    // Model persistence
    std::vector<Tensor> getParameters() const override;
    void setParameters(const std::vector<Tensor>& params) override;
    std::string getLayerType() const override;
    
    bool hasWeights() const override { return true; }
    bool hasBiases() const override { return true; }

private:
    Tensor weights;
    Tensor biases;
    Tensor last_input;  // Store for backprop
    Tensor last_output; // Store for backprop
    int input_size;
    int output_size;
    Activation activation_func;
    bool weights_initialized;
    std::shared_ptr<Optimizer> weight_optimizer;
    std::shared_ptr<Optimizer> bias_optimizer;
};

}

#endif // JNET_LAYERS_DENSE_H