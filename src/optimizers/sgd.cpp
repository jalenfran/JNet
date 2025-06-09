#include "sgd.h"

namespace JNet {

SGD::SGD(double learning_rate) : learning_rate(learning_rate) {}

void SGD::update(Tensor& weights, const Tensor& gradients) {
    if (weights.shape() != gradients.shape()) {
        throw std::invalid_argument("Weights and gradients must have the same shape");
    }
    
    for (int i = 0; i < weights.size(); ++i) {
        weights.at(i) -= learning_rate * gradients.at(i);
    }
}

void SGD::setLearningRate(double lr) {
    learning_rate = lr;
}

double SGD::getLearningRate() const {
    return learning_rate;
}

}