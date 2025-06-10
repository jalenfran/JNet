#include "../include/jnet.h"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace JNet;

int main() {
    std::cout << "=== JNet Enhanced Training Demo ===" << std::endl;
    std::cout << "Demonstrating training with progress bar and verbose statistics" << std::endl;
    
    // Create a simple neural network for XOR problem
    Network network;
    network.addLayer(new Dense(2, 8, Activation::ReLU));   // Input: 2, Hidden: 8
    network.addLayer(new Dense(8, 4, Activation::ReLU));   // Hidden: 8->4
    network.addLayer(new Dense(4, 1, Activation::Sigmoid)); // Hidden: 4, Output: 1
    
    // Set optimizer
    auto optimizer = std::make_shared<SGD>(0.1);
    network.setOptimizer(optimizer);
    
    // XOR dataset
    std::vector<Tensor> inputs;
    std::vector<Tensor> targets;
    
    // Create multiple copies of the XOR dataset for better training
    for (int repeat = 0; repeat < 1000; ++repeat) {
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
    }
    
    std::cout << "\nDataset size: " << inputs.size() << " samples" << std::endl;
    
    // Test before training
    std::cout << "\nBefore training:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        Tensor output = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i].at(0) << ", " << inputs[i].at(1) 
                  << "] -> Output: " << std::fixed << std::setprecision(4) << output.at(0) 
                  << " (Target: " << targets[i].at(0) << ")" << std::endl;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "1. Training with BASIC progress bar (default settings)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Train with basic progress bar
    Network::TrainingConfig basic_config;
    network.trainEpochsAdvanced(inputs, targets, 50, basic_config);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "2. Training with DETAILED progress bar (with accuracy)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Train with detailed progress bar
    Network::TrainingConfig detailed_config;
    detailed_config.show_accuracy = true;
    detailed_config.print_every = 5;  // Print every 5 epochs
    network.trainEpochsAdvanced(inputs, targets, 25, detailed_config);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "3. Training with COMPACT output (no progress bar)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Train with compact output
    Network::TrainingConfig compact_config;
    compact_config.show_progress_bar = false;
    compact_config.show_accuracy = true;
    compact_config.print_every = 10;
    network.trainEpochsAdvanced(inputs, targets, 25, compact_config);
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "4. Silent training (verbose = false)" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    // Silent training
    Network::TrainingConfig silent_config(false);
    network.trainEpochsAdvanced(inputs, targets, 50, silent_config);
    std::cout << "Silent training completed (no output during training)" << std::endl;
    
    // Test after training
    std::cout << "\nAfter training:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        Tensor output = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i].at(0) << ", " << inputs[i].at(1) 
                  << "] -> Output: " << std::fixed << std::setprecision(4) << output.at(0) 
                  << " (Target: " << targets[i].at(0) << ")" << std::endl;
    }
    
    // Calculate final accuracy
    double accuracy = network.evaluateAccuracy(inputs, targets);
    std::cout << "\nFinal Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0) << "%" << std::endl;
    
    std::cout << "\n=== Demo Features Showcased ===" << std::endl;
    std::cout << "✓ Progress bar with percentage and ETA" << std::endl;
    std::cout << "✓ Real-time loss monitoring" << std::endl;
    std::cout << "✓ Optional accuracy tracking" << std::endl;
    std::cout << "✓ Timing information (ms/epoch)" << std::endl;
    std::cout << "✓ Configurable verbosity levels" << std::endl;
    std::cout << "✓ Compact vs detailed output modes" << std::endl;
    std::cout << "✓ Silent training option" << std::endl;
    
    return 0;
}
