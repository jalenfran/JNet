#include <iostream>
#include <vector>
#include "../include/jnet.h"

using namespace JNet;

int main() {
    std::cout << "=== JNet Convolutional Neural Network Example ===\n\n";
    
    // Create a simple CNN for image classification
    // Input: 28x28 grayscale image (like MNIST)
    Network cnn;
    
    std::cout << "Building CNN architecture...\n";
    
    // First convolutional layer: 1 input channel, 32 filters, 3x3 kernel
    cnn.addLayer(new Conv2D(1, 32, 3, 1, 1, Activation::ReLU));
    std::cout << "Added Conv2D layer: 32 filters, 3x3 kernel, stride=1, padding=1\n";
    
    // Second convolutional layer: 32 input channels, 64 filters, 3x3 kernel
    cnn.addLayer(new Conv2D(32, 64, 3, 1, 1, Activation::ReLU));
    std::cout << "Added Conv2D layer: 64 filters, 3x3 kernel, stride=1, padding=1\n";
    
    // Flatten layer to convert 2D feature maps to 1D vector
    cnn.addLayer(new Flatten());
    std::cout << "Added Flatten layer\n";
    
    // Dense layers for classification
    cnn.addLayer(new Dense(128, Activation::ReLU));
    std::cout << "Added Dense layer: 128 neurons, ReLU activation\n";
    
    cnn.addLayer(new Dense(10, Activation::Linear));  // 10 classes
    std::cout << "Added Dense layer: 10 neurons (output layer)\n";
    
    // Set optimizer
    auto optimizer = std::make_shared<SGD>(0.001);  // Lower learning rate for CNN
    cnn.setOptimizer(optimizer);
    std::cout << "Set SGD optimizer with learning rate 0.001\n\n";
    
    // Create sample data (simulating a 28x28 grayscale image)
    std::cout << "Creating sample data...\n";
    
    // Input tensor shape: [channels, height, width] = [1, 28, 28]
    Tensor input({1, 28, 28});
    input.fillRandom();  // Fill with random values to simulate an image
    
    // Target tensor (one-hot encoded for class 3)
    Tensor target({1, 10});  // Shape: [1, 10] to match Dense output
    target.fill(0.0);
    target[{0, 3}] = 1.0;  // Class 3
    
    std::cout << "Input shape: [1, 28, 28] (grayscale image)\n";
    std::cout << "Target: Class 3 (one-hot encoded)\n\n";
    
    // Test forward pass
    std::cout << "Testing forward pass...\n";
    try {
        Tensor output = cnn.forward(input);
        std::cout << "Forward pass successful!\n";
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";
        std::cout << "Output values: ";
        for (int i = 0; i < std::min(5, output.size()); ++i) {
            std::cout << output.at(i) << " ";
        }
        if (output.size() > 5) std::cout << "...";
        std::cout << "\n\n";
        
        // Test training step
        std::cout << "Testing training step...\n";
        cnn.train(input, target);
        std::cout << "Training step successful!\n\n";
        
        // Test prediction after training
        std::cout << "Testing prediction after training...\n";
        Tensor prediction = cnn.predict(input);
        std::cout << "Prediction values: ";
        for (int i = 0; i < prediction.size(); ++i) {
            std::cout << prediction.at(i) << " ";
        }
        std::cout << "\n";
        
        // Find predicted class
        int predicted_class = 0;
        for (int i = 1; i < prediction.size(); ++i) {
            if (prediction.at(i) > prediction.at(predicted_class)) {
                predicted_class = i;
            }
        }
        std::cout << "Predicted class: " << predicted_class << "\n";
        std::cout << "Actual class: 3\n\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error during forward pass: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "=== CNN Example completed successfully! ===\n";
    std::cout << "\nThis example demonstrates:\n";
    std::cout << "- Creating convolutional layers with different filter sizes\n";
    std::cout << "- Using a flatten layer to transition from conv to dense layers\n"; 
    std::cout << "- Training a complete CNN architecture\n";
    std::cout << "- Processing 2D image data through the network\n";
    
    return 0;
}
