#include "../include/jnet.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace JNet;

int main() {
    std::cout << "=== JNet Optimizer Comparison Demo ===" << std::endl;
    
    // Training data for XOR problem
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
    Tensor target1({1, 1});
    target1.at(0) = 0.0;
    targets.push_back(target1);
    
    Tensor target2({1, 1});
    target2.at(0) = 1.0;
    targets.push_back(target2);
    
    Tensor target3({1, 1});
    target3.at(0) = 1.0;
    targets.push_back(target3);
    
    Tensor target4({1, 1});
    target4.at(0) = 0.0;
    targets.push_back(target4);
    
    // Test SGD optimizer
    std::cout << "\n--- Training with SGD Optimizer ---" << std::endl;
    {
        Network sgd_network;
        sgd_network.addLayer(new Dense(2, 4, Activation::ReLU));
        sgd_network.addLayer(new Dense(4, 1, Activation::Sigmoid));
        
        // Set SGD optimizer with learning rate 0.1
        auto sgd_optimizer = std::make_shared<SGD>(0.1);
        sgd_network.setOptimizer(sgd_optimizer);
        
        std::cout << "Learning rate: " << sgd_optimizer->getLearningRate() << std::endl;
        
        // Train for a few epochs and show progress
        for (int epoch = 1; epoch <= 500; epoch += 100) {
            sgd_network.trainBatch(inputs, targets);
            if (epoch % 100 == 1) {
                double accuracy = sgd_network.evaluateAccuracy(inputs, targets);
                std::cout << "Epoch " << epoch << " - Accuracy: " << (accuracy * 100) << "%" << std::endl;
            }
        }
        
        // Final test
        std::cout << "Final SGD Results:" << std::endl;
        for (int i = 0; i < inputs.size(); ++i) {
            Tensor prediction = sgd_network.predict(inputs[i]);
            std::cout << "Input: [" << inputs[i].at(0) << ", " << inputs[i].at(1) 
                      << "] -> Output: " << prediction.at(0) 
                      << " (Expected: " << targets[i].at(0) << ")" << std::endl;
        }
    }
    
    // Test Adam optimizer
    std::cout << "\n--- Training with Adam Optimizer ---" << std::endl;
    {
        Network adam_network;
        adam_network.addLayer(new Dense(2, 4, Activation::ReLU));
        adam_network.addLayer(new Dense(4, 1, Activation::Sigmoid));
        
        // Set Adam optimizer with default parameters
        auto adam_optimizer = std::make_shared<Adam>(0.01);
        adam_network.setOptimizer(adam_optimizer);
        
        std::cout << "Using Adam optimizer with lr=0.01" << std::endl;
        
        // Train for a few epochs and show progress
        for (int epoch = 1; epoch <= 500; epoch += 100) {
            adam_network.trainBatch(inputs, targets);
            if (epoch % 100 == 1) {
                double accuracy = adam_network.evaluateAccuracy(inputs, targets);
                std::cout << "Epoch " << epoch << " - Accuracy: " << (accuracy * 100) << "%" << std::endl;
            }
        }
        
        // Final test
        std::cout << "Final Adam Results:" << std::endl;
        for (int i = 0; i < inputs.size(); ++i) {
            Tensor prediction = adam_network.predict(inputs[i]);
            std::cout << "Input: [" << inputs[i].at(0) << ", " << inputs[i].at(1) 
                      << "] -> Output: " << prediction.at(0) 
                      << " (Expected: " << targets[i].at(0) << ")" << std::endl;
        }
    }
    
    std::cout << "\n=== Optimizer Integration Complete! ===" << std::endl;
    return 0;
}
