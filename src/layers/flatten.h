#ifndef JNET_LAYERS_FLATTEN_H
#define JNET_LAYERS_FLATTEN_H

// JNet Neural Network Framework
// Flatten layer - Flattens multi-dimensional tensors to 1D
//
// Provides functionality for:
// - Flattening N-dimensional tensors to 1D vectors
// - Reshaping gradients back to original dimensions during backprop
// - No learnable parameters

#include "../core/tensor.h"
#include "layer.h"

namespace JNet {

class Flatten : public Layer {
public:
    Flatten();
    ~Flatten() override;

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& output_gradient) override;
    
    // Model persistence (no parameters for flatten layer)
    std::string getLayerType() const override { return "Flatten"; }
    
    bool hasWeights() const override { return false; }
    bool hasBiases() const override { return false; }

private:
    std::vector<int> input_shape;  // Store original shape for backprop
};

}

#endif // JNET_LAYERS_FLATTEN_H
