#include <iostream>
#include <vector>
#include "../include/jnet.h"

using namespace JNet;

int main() {
    std::cout << "=== JNet Convolutional Neural Network Example ===\n\n";
    
    // Create a simple CNN for image classification
    // Input: 14x14 grayscale image (optimized for speed)
    Network cnn;
    
    std::cout << "Building CNN architecture...\n";
    
    // First convolutional layer: 1 input channel, 8 filters, 3x3 kernel (reduced for speed)
    cnn.addLayer(new Conv2D(1, 8, 3, 1, 1, Activation::ReLU));
    std::cout << "Added Conv2D layer: 8 filters, 3x3 kernel, stride=1, padding=1\n";
    
    // Add pooling to reduce spatial dimensions for speed
    cnn.addLayer(new MaxPool2D(2, 2));
    std::cout << "Added MaxPool2D layer: 2x2 pooling\n";
    
    // Second convolutional layer: 8 input channels, 16 filters, 3x3 kernel (reduced for speed)
    cnn.addLayer(new Conv2D(8, 16, 3, 1, 1, Activation::ReLU));
    std::cout << "Added Conv2D layer: 16 filters, 3x3 kernel, stride=1, padding=1\n";
    
    // Add another pooling layer
    cnn.addLayer(new MaxPool2D(2, 2));
    std::cout << "Added MaxPool2D layer: 2x2 pooling\n";
    
    // Flatten layer to convert 2D feature maps to 1D vector
    cnn.addLayer(new Flatten());
    std::cout << "Added Flatten layer\n";
    
    // Dense layers for classification (reduced size for speed)
    cnn.addLayer(new Dense(32, Activation::ReLU));
    std::cout << "Added Dense layer: 32 neurons, ReLU activation\n";
    
    cnn.addLayer(new Dense(10, Activation::Linear));  // 10 classes
    std::cout << "Added Dense layer: 10 neurons (output layer)\n";
    
    // Set optimizer with higher learning rate for faster convergence
    auto optimizer = std::make_shared<SGD>(0.01);  // Higher learning rate for faster demo
    cnn.setOptimizer(optimizer);
    std::cout << "Set SGD optimizer with learning rate 0.01\n\n";
    
    // Create training dataset (simulating smaller 14x14 grayscale images for speed)
    std::cout << "Creating training dataset...\n";
    
    std::vector<Tensor> train_inputs;
    std::vector<Tensor> train_targets;
    
    // Generate 1000 random training samples with different classes (reduced for speed)
    for (int i = 0; i < 100; ++i) {
        // Input tensor shape: [channels, height, width] = [1, 14, 14] (smaller for speed)
        Tensor input({1, 14, 14});
        input.fillRandom();  // Fill with random values to simulate an image
        train_inputs.push_back(input);
        
        // Target tensor (one-hot encoded for random classes)
        Tensor target({1, 10});  // Shape: [1, 10] to match Dense output
        target.fill(0.0);
        int class_label = i % 10;  // Cycle through classes 0-9
        target[{0, class_label}] = 1.0;
        train_targets.push_back(target);
    }
    
    std::cout << "Created " << train_inputs.size() << " training samples\n";
    std::cout << "Input shape: [1, 14, 14] (small grayscale images for demo)\n";
    std::cout << "Output classes: 0-9 (10 classes total)\n\n";
    
    // Test forward pass
    std::cout << "Testing forward pass...\n";
    try {
        Tensor output = cnn.forward(train_inputs[0]);
        std::cout << "Forward pass successful!\n";
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";
        std::cout << "Output values: ";
        for (int i = 0; i < std::min(5, output.size()); ++i) {
            std::cout << output.at(i) << " ";
        }
        if (output.size() > 5) std::cout << "...";
        std::cout << "\n\n";
        
        // Test basic training functionality
        std::cout << "Testing single training step...\n";
        cnn.train(train_inputs[0], train_targets[0]);
        std::cout << "Training step successful!\n\n";
        
        // Show predictions before training
        std::cout << "=== Initial Predictions (before training) ===\n";
        for (int i = 0; i < 3; ++i) {
            Tensor prediction = cnn.predict(train_inputs[i]);
            int predicted_class = 0;
            for (int j = 1; j < prediction.size(); ++j) {
                if (prediction.at(j) > prediction.at(predicted_class)) {
                    predicted_class = j;
                }
            }
            int actual_class = 0;
            for (int j = 1; j < train_targets[i].size(); ++j) {
                if (train_targets[i].at(j) > train_targets[i].at(actual_class)) {
                    actual_class = j;
                }
            }
            std::cout << "Sample " << (i+1) << " - Predicted: " << predicted_class 
                      << ", Actual: " << actual_class << std::endl;
        }
        std::cout << std::endl;
        
        // Now demonstrate the enhanced training with progress bar
        std::cout << "=== Training CNN with Enhanced Progress Bar ===\n";
        
        // Configure training with progress bar and accuracy tracking
        Network::TrainingConfig config;
        config.verbose = true;
        config.show_progress_bar = true;
        config.show_accuracy = true;
        config.print_every = 2;  // Print stats every 2 epochs for faster demo
        config.progress_bar_width = 30; // Smaller progress bar
        
        // Train the CNN for 10 epochs (reduced for speed)
        cnn.trainEpochsAdvanced(train_inputs, train_targets, 50, config);
        
        // Show predictions after training
        std::cout << "\n=== Final Predictions (after training) ===\n";
        for (int i = 0; i < std::min(5, static_cast<int>(train_inputs.size())); ++i) {
            Tensor prediction = cnn.predict(train_inputs[i]);
            int predicted_class = 0;
            for (int j = 1; j < prediction.size(); ++j) {
                if (prediction.at(j) > prediction.at(predicted_class)) {
                    predicted_class = j;
                }
            }
            int actual_class = 0;
            for (int j = 1; j < train_targets[i].size(); ++j) {
                if (train_targets[i].at(j) > train_targets[i].at(actual_class)) {
                    actual_class = j;
                }
            }
            std::cout << "Sample " << (i+1) << " - Predicted: " << predicted_class 
                      << ", Actual: " << actual_class;
            if (predicted_class == actual_class) {
                std::cout << " ✓";
            } else {
                std::cout << " ✗";
            }
            std::cout << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error during training: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "=== CNN Example completed successfully! ===\n";
    std::cout << "\nThis example demonstrates:\n";
    std::cout << "- Creating convolutional layers with pooling for efficiency\n";
    std::cout << "- Using a flatten layer to transition from conv to dense layers\n"; 
    std::cout << "- Training a complete optimized CNN architecture\n";
    std::cout << "- Processing 2D image data through the network efficiently\n";
    std::cout << "- Fast performance with optimized tensor operations\n";
    
    return 0;
}