#ifndef JNET_LAYERS_LAYER_H
#define JNET_LAYERS_LAYER_H

// JNet Neural Network Framework
// Layer base class - Abstract base class for all neural network layers
//
// Provides common interface for:
// - Forward propagation
// - Backward propagation
// - Optimizer management
// - Weight and bias handling

#include "../core/tensor.h"
#include "../optimizers/optimizer.h"
#include <memory>

namespace JNet {

class Layer {
public:
    virtual ~Layer() = default;
    
    // Pure virtual methods that must be implemented by derived classes
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& output_gradient) = 0;
    
    // Optional methods with default implementations
    virtual void setOptimizer(std::shared_ptr<Optimizer> opt) {}
    virtual void setOptimizerType(const std::string& type, double learning_rate = 0.01) {}
    
    // Model persistence (optional, layers without parameters don't need to implement)
    virtual std::vector<Tensor> getParameters() const { return {}; }
    virtual void setParameters(const std::vector<Tensor>& params) {}
    virtual std::string getLayerType() const = 0;
    
    // Utility methods
    virtual bool hasWeights() const { return false; }
    virtual bool hasBiases() const { return false; }
};

}

#endif // JNET_LAYERS_LAYER_H
