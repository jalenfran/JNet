#include "network.h"
#include "../layers/conv2d.h"
#include "../layers/flatten.h"
#include <iostream>
#include <cmath>

namespace JNet {

Network::Network() {
}

Network::~Network() {
    // Smart pointers will automatically clean up
}

void Network::addLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

void Network::addLayer(Layer* layer) {
    layers.push_back(std::unique_ptr<Layer>(layer));
}

void Network::setOptimizer(std::shared_ptr<Optimizer> optimizer) {
    for (auto& layer : layers) {
        layer->setOptimizer(optimizer);
    }
}

Tensor Network::forward(const Tensor& input) {
    Tensor output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    last_output = output;
    return output;
}

void Network::backward(const Tensor& target) {
    // Calculate loss gradient (using mean squared error)
    Tensor loss_gradient = calculateLossGradient(last_output, target);
    
    // Backpropagate through layers in reverse order
    Tensor gradient = loss_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        gradient = (*it)->backward(gradient);
    }
}

void Network::train(const Tensor& input, const Tensor& target) {
    // Forward pass
    forward(input);
    
    // Backward pass
    backward(target);
}

Tensor Network::predict(const Tensor& input) {
    return forward(input);
}

void Network::trainEpochs(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, 
                         int epochs, bool verbose) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    if (verbose) {
        std::cout << "Starting training for " << epochs << " epochs with " << inputs.size() << " samples\n";
        std::cout << "Epoch\tLoss\n";
        std::cout << "-----\t----\n";
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Train on all samples
        trainBatch(inputs, targets);
        
        // Calculate and display loss if verbose
        if (verbose) {
            double loss = calculateBatchLoss(inputs, targets);
            std::cout << epoch + 1 << "\t" << loss << std::endl;
        }
    }
    
    if (verbose) {
        std::cout << "Training completed!\n";
    }
}

void Network::trainBatch(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        train(inputs[i], targets[i]);
    }
}

double Network::evaluateAccuracy(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction = predict(inputs[i]);
        
        // Find predicted class (highest value)
        int predicted_class = 0;
        int actual_class = 0;
        
        for (int j = 1; j < prediction.size(); ++j) {
            if (prediction.at(j) > prediction.at(predicted_class)) {
                predicted_class = j;
            }
            if (targets[i].at(j) > targets[i].at(actual_class)) {
                actual_class = j;
            }
        }
        
        if (predicted_class == actual_class) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / inputs.size();
}

double Network::calculateBatchLoss(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction = predict(inputs[i]);
        total_loss += calculateLoss(prediction, targets[i]);
    }
    return total_loss / inputs.size();
}

double Network::calculateLoss(const Tensor& predicted, const Tensor& target) {
    if (predicted.shape() != target.shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }
    
    double loss = 0.0;
    for (int i = 0; i < predicted.size(); ++i) {
        double diff = predicted.at(i) - target.at(i);
        loss += diff * diff;
    }
    return loss / (2.0 * predicted.size()); // Mean squared error
}

Tensor Network::calculateLossGradient(const Tensor& predicted, const Tensor& target) {
    if (predicted.shape() != target.shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }
    
    Tensor gradient = predicted;
    for (int i = 0; i < gradient.size(); ++i) {
        gradient.at(i) = (predicted.at(i) - target.at(i)) / predicted.size();
    }
    return gradient;
}

}