#include <iostream>
#include <vector>
#include "../include/jnet.h"

using namespace JNet;

int main() {
    std::cout << "=== JNet Practical CNN Example ===\n\n";
    
    // Create a CNN for smaller images (e.g., 8x8 or 16x16)
    Network cnn;
    
    std::cout << "Building practical CNN architecture...\n";
    
    // First convolutional layer: 1 input channel, 8 filters, 3x3 kernel
    cnn.addLayer(new Conv2D(1, 8, 3, 1, 1, Activation::ReLU));
    std::cout << "Added Conv2D layer: 8 filters, 3x3 kernel, stride=1, padding=1\n";
    
    // Second convolutional layer: 8 input channels, 16 filters, 3x3 kernel  
    cnn.addLayer(new Conv2D(8, 16, 3, 1, 1, Activation::ReLU));
    std::cout << "Added Conv2D layer: 16 filters, 3x3 kernel, stride=1, padding=1\n";
    
    // Flatten layer
    cnn.addLayer(new Flatten());
    std::cout << "Added Flatten layer\n";
    
    // Dense layers for classification
    cnn.addLayer(new Dense(32, Activation::ReLU));
    std::cout << "Added Dense layer: 32 neurons, ReLU activation\n";
    
    cnn.addLayer(new Dense(4, Activation::Linear));  // 4 classes
    std::cout << "Added Dense layer: 4 neurons (output layer)\n";
    
    // Set optimizer
    auto optimizer = std::make_shared<SGD>(0.01);
    cnn.setOptimizer(optimizer);
    std::cout << "Set SGD optimizer with learning rate 0.01\n\n";
    
    // Create sample data (16x16 grayscale image)
    std::cout << "Creating sample data...\n";
    
    // Input tensor shape: [channels, height, width] = [1, 16, 16]
    Tensor input({1, 16, 16});
    input.fillRandom();
    
    // Target tensor (one-hot encoded for class 2)
    Tensor target({1, 4});
    target.fill(0.0);
    target[{0, 2}] = 1.0;  // Class 2
    
    std::cout << "Input shape: [1, 16, 16] (grayscale image)\n";
    std::cout << "Target: Class 2 (one-hot encoded)\n\n";
    
    // Test forward pass
    std::cout << "Testing forward pass...\n";
    try {
        Tensor output = cnn.forward(input);
        std::cout << "Forward pass successful!\n";
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n";
        std::cout << "Output values: ";
        for (int i = 0; i < output.shape()[1]; ++i) {
            std::cout << output[{0, i}] << " ";
        }
        std::cout << "\n\n";
        
        // Test training step
        std::cout << "Testing training step...\n";
        cnn.train(input, target);
        std::cout << "Training step successful!\n\n";
        
        // Test prediction after training
        std::cout << "Testing prediction after training...\n";
        Tensor prediction = cnn.predict(input);
        std::cout << "Prediction values: ";
        for (int i = 0; i < prediction.shape()[1]; ++i) {
            std::cout << prediction[{0, i}] << " ";
        }
        std::cout << "\n\n";
        
        // Multiple training steps
        std::cout << "Training for 5 steps...\n";
        for (int i = 0; i < 5; ++i) {
            cnn.train(input, target);
            if ((i + 1) % 2 == 0) {
                Tensor pred = cnn.predict(input);
                std::cout << "After " << (i + 1) << " steps - Prediction: ";
                for (int j = 0; j < pred.shape()[1]; ++j) {
                    std::cout << pred[{0, j}] << " ";
                }
                std::cout << "\n";
            }
        }
        
        std::cout << "\nCNN example completed successfully!\n";
        std::cout << "\nArchitecture summary:\n";
        std::cout << "- Input: 16x16 grayscale image\n";
        std::cout << "- Conv2D: 1→8 channels, 3x3 filters\n";
        std::cout << "- Conv2D: 8→16 channels, 3x3 filters\n";
        std::cout << "- Flatten: 16x16x16 → 4096 features\n";
        std::cout << "- Dense: 4096 → 32 → 4 classes\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
