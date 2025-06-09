#ifndef JNET_OPTIMIZERS_SGD_H
#define JNET_OPTIMIZERS_SGD_H

// JNet Neural Network Framework
// SGD Optimizer - Stochastic Gradient Descent optimization
//
// Provides functionality for:
// - Standard gradient descent optimization
// - Configurable learning rate
// - Simple and efficient parameter updates
// - Foundation for more advanced optimizers

#include "../core/tensor.h"
#include "optimizer.h"

namespace JNet {

class SGD : public Optimizer {
public:
    SGD(double learning_rate = 0.01);
    void update(Tensor& weights, const Tensor& gradients) override;
    void setLearningRate(double lr);
    double getLearningRate() const;

private:
    double learning_rate;
};

}

#endif // JNET_OPTIMIZERS_SGD_H