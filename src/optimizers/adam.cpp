#include "adam.h"
#include <cmath>
#include <algorithm>

namespace JNet {

Adam::Adam(double learning_rate, double beta1, double beta2, double epsilon)
    : learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon) {}

void Adam::update(Tensor& weights, const Tensor& gradients) {
    if (weights.shape() != gradients.shape()) {
        throw std::invalid_argument("Weights and gradients must have the same shape");
    }
    
    // Use the tensor's memory address as a unique key
    void* key = static_cast<void*>(&weights);
    
    // Get or create state for this tensor
    AdamState& state = states[key];
    
    // Initialize state if needed
    if (!state.initialized) {
        state.m = Tensor::zeros(weights.shape());
        state.v = Tensor::zeros(weights.shape());
        state.initialized = true;
    }
    
    state.t++;
    
    // Update biased first and second moment estimates
    for (int i = 0; i < weights.size(); ++i) {
        state.m.at(i) = beta1 * state.m.at(i) + (1.0 - beta1) * gradients.at(i);
        state.v.at(i) = beta2 * state.v.at(i) + (1.0 - beta2) * gradients.at(i) * gradients.at(i);
        
        // Compute bias-corrected moment estimates
        double m_hat = state.m.at(i) / (1.0 - std::pow(beta1, state.t));
        double v_hat = state.v.at(i) / (1.0 - std::pow(beta2, state.t));
        
        // Update weights
        weights.at(i) -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
    }
}

void Adam::reset() {
    states.clear();
}

}