#ifndef JNET_OPTIMIZERS_ADAM_H
#define JNET_OPTIMIZERS_ADAM_H

// JNet Neural Network Framework
// Adam Optimizer - Adaptive Moment Estimation optimization
//
// Provides functionality for:
// - Advanced gradient descent with momentum and adaptive learning rates
// - Bias correction for first and second moment estimates
// - Efficient convergence on many optimization problems
// - State management for momentum and RMSprop-like behavior

#include "../core/tensor.h"
#include "optimizer.h"
#include <unordered_map>

namespace JNet {

class Adam : public Optimizer {
public:
    Adam(double learning_rate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
    void update(Tensor& weights, const Tensor& gradients) override;
    void reset() override;
    
    // Getters for parameters
    double getLearningRate() const { return learning_rate; }
    double getBeta1() const { return beta1; }
    double getBeta2() const { return beta2; }
    double getEpsilon() const { return epsilon; }

private:
    struct AdamState {
        Tensor m;  // First moment vector
        Tensor v;  // Second moment vector
        int t;     // Time step
        bool initialized;
        
        AdamState() : t(0), initialized(false) {}
    };
    
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    
    // Map to store separate state for each parameter tensor
    std::unordered_map<void*, AdamState> states;
};

}

#endif // JNET_OPTIMIZERS_ADAM_H