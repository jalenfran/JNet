#ifndef JNET_OPTIMIZERS_OPTIMIZER_H
#define JNET_OPTIMIZERS_OPTIMIZER_H

// JNet Neural Network Framework
// Base Optimizer class - Abstract base for all optimization algorithms
//
// Provides functionality for:
// - Common interface for all optimizers
// - Polymorphic behavior for different optimization strategies
// - Parameter update abstraction
// - State management interface

#include "../core/tensor.h"

namespace JNet {

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(Tensor& weights, const Tensor& gradients) = 0;
    virtual void reset() {}
};

}

#endif // JNET_OPTIMIZERS_OPTIMIZER_H
