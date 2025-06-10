#ifndef JNET_LAYERS_POOLING_H
#define JNET_LAYERS_POOLING_H

// JNet Neural Network Framework
// Pooling layers - Downsampling operations for CNNs
//
// Provides functionality for:
// - MaxPool2D: Maximum pooling for feature extraction
// - AvgPool2D: Average pooling for feature extraction
// - Configurable pool size and stride
// - Proper gradient computation for backpropagation

#include "layer.h"
#include "../core/tensor.h"
#include "../optimizers/optimizer.h"
#include <memory>
#include <limits>

namespace JNet {

class MaxPool2D : public Layer {
private:
    int pool_size;
    int stride;
    Tensor last_input;
    Tensor mask; // Stores which elements were the maximum for backprop
    
public:
    MaxPool2D(int pool_size, int stride = -1);
    ~MaxPool2D();
    
    // Layer interface
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    void setOptimizer(std::shared_ptr<Optimizer> opt) override;
    void setOptimizerType(const std::string& type, double learning_rate) override;
    
    // Model persistence (no parameters for pooling layers)
    std::string getLayerType() const override { return "MaxPool2D"; }
    
    // Utility methods
    std::vector<int> calculate_output_shape(const std::vector<int>& input_shape) const;
};

class AvgPool2D : public Layer {
private:
    int pool_size;
    int stride;
    Tensor last_input;
    
public:
    AvgPool2D(int pool_size, int stride = -1);
    ~AvgPool2D();
    
    // Layer interface
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    void setOptimizer(std::shared_ptr<Optimizer> opt) override;
    void setOptimizerType(const std::string& type, double learning_rate) override;
    
    // Model persistence (no parameters for pooling layers)
    std::string getLayerType() const override { return "AvgPool2D"; }
    
    // Utility methods
    std::vector<int> calculate_output_shape(const std::vector<int>& input_shape) const;
};

}

#endif // JNET_LAYERS_POOLING_H
