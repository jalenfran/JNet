#ifndef JNET_CORE_NETWORK_H
#define JNET_CORE_NETWORK_H

// JNet Neural Network Framework
// Network class - Neural network architecture and training
//
// Provides functionality for:
// - Building neural networks with multiple layers
// - Forward and backward propagation
// - Training with automatic differentiation
// - Layer management and composition

#include <vector>
#include <memory>
#include "tensor.h"
#include "../layers/layer.h"
#include "../layers/dense.h"
#include "../layers/conv2d.h"
#include "../layers/flatten.h"
#include "../optimizers/optimizer.h"

namespace JNet {

class Network {
public:
    Network();
    ~Network();

    // Polymorphic layer management
    void addLayer(std::unique_ptr<Layer> layer);
    void addLayer(Layer* layer);  // Takes ownership and wraps in unique_ptr
    
    void setOptimizer(std::shared_ptr<Optimizer> optimizer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& target);
    void train(const Tensor& input, const Tensor& target);
    
    // Epoch-based training methods
    void trainEpochs(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, 
                     int epochs, bool verbose = true);
    void trainBatch(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);
    double evaluateAccuracy(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);
    
    Tensor predict(const Tensor& input);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    Tensor last_output;
    
    double calculateLoss(const Tensor& predicted, const Tensor& target);
    Tensor calculateLossGradient(const Tensor& predicted, const Tensor& target);
    double calculateBatchLoss(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);
};

}

#endif // JNET_CORE_NETWORK_H