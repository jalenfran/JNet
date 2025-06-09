#include "../include/jnet.h"
#include <iostream>
#include <vector>

int main() {
    using namespace JNet;
    
    std::cout << "=== JNet Epoch-Based Training Example ===" << std::endl;
    
    // Create a simple neural network for XOR problem
    Network network;
    network.addLayer(new Dense(2, 4, Activation::ReLU));  // Input: 2, Hidden: 4
    network.addLayer(new Dense(4, 1, Activation::Sigmoid)); // Hidden: 4, Output: 1
    
    // XOR dataset
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    
    // Input [0, 0] -> Target 0
    Tensor input1({1, 2});
    input1.at(0) = 0.0; input1.at(1) = 0.0;
    inputs.push_back(input1);
    
    // Input [0, 1] -> Target 1
    Tensor input2({1, 2});
    input2.at(0) = 0.0; input2.at(1) = 1.0;
    inputs.push_back(input2);
    
    // Input [1, 0] -> Target 1
    Tensor input3({1, 2});
    input3.at(0) = 1.0; input3.at(1) = 0.0;
    inputs.push_back(input3);
    
    // Input [1, 1] -> Target 0
    Tensor input4({1, 2});
    input4.at(0) = 1.0; input4.at(1) = 1.0;
    inputs.push_back(input4);
    
    // Targets
    Tensor target1({1, 1}); target1.at(0) = 0.0; targets.push_back(target1);
    Tensor target2({1, 1}); target2.at(0) = 1.0; targets.push_back(target2);
    Tensor target3({1, 1}); target3.at(0) = 1.0; targets.push_back(target3);
    Tensor target4({1, 1}); target4.at(0) = 0.0; targets.push_back(target4);
    
    std::cout << "\nBefore training:" << std::endl;
    for (int i = 0; i < inputs.size(); ++i) {
        Tensor output = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i].at(0) << ", " << inputs[i].at(1) 
                  << "] -> Output: " << output.at(0) 
                  << " (Target: " << targets[i].at(0) << ")" << std::endl;
    }
    
    // Train the network for multiple epochs
    std::cout << "\n=== Training Progress ===" << std::endl;
    network.trainEpochs(inputs, targets, 1000, true);
    
    std::cout << "\nAfter training:" << std::endl;
    for (int i = 0; i < inputs.size(); ++i) {
        Tensor output = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i].at(0) << ", " << inputs[i].at(1) 
                  << "] -> Output: " << output.at(0) 
                  << " (Target: " << targets[i].at(0) << ")" << std::endl;
    }
    
    // Calculate final accuracy
    double accuracy = network.evaluateAccuracy(inputs, targets);
    std::cout << "\nFinal Accuracy: " << (accuracy * 100.0) << "%" << std::endl;
    
    return 0;
}