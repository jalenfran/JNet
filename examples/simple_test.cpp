#include <iostream>
#include <vector>
#include "jnet.h"

int main() {
    std::cout << "=== JNet Neural Network Framework Test ===" << std::endl;
    
    // Test 1: Basic Tensor Operations
    std::cout << "\n1. Testing Tensor Operations:" << std::endl;
    JNet::Tensor t1({2, 3});
    t1.fill(2.0);
    std::cout << "Tensor filled with 2.0: " << t1 << std::endl;
    
    JNet::Tensor t2 = JNet::Tensor::random({2, 3});
    std::cout << "Random tensor: " << t2 << std::endl;
    
    JNet::Tensor t3 = t1 + t2;
    std::cout << "Sum: " << t3 << std::endl;
    
    // Test 2: Matrix Operations
    std::cout << "\n2. Testing Matrix Operations:" << std::endl;
    JNet::Tensor m1({2, 3});
    JNet::Tensor m2({3, 2});
    m1.fillRandom();
    m2.fillRandom();
    
    std::cout << "Matrix 1 (2x3): " << m1 << std::endl;
    std::cout << "Matrix 2 (3x2): " << m2 << std::endl;
    
    JNet::Tensor product = m1.dot(m2);
    std::cout << "Matrix multiplication (2x2): " << product << std::endl;
    
    // Test 3: Activation Functions
    std::cout << "\n3. Testing Activation Functions:" << std::endl;
    JNet::Tensor input({1, 5});
    for (int i = 0; i < 5; ++i) {
        input.at(i) = -2.0 + i; // Values: -2, -1, 0, 1, 2
    }
    std::cout << "Input: " << input << std::endl;
    
    JNet::Tensor relu_output = JNet::ActivationFunction::relu(input);
    std::cout << "ReLU: " << relu_output << std::endl;
    
    JNet::Tensor sigmoid_output = JNet::ActivationFunction::sigmoid(input);
    std::cout << "Sigmoid: " << sigmoid_output << std::endl;
    
    // Test 4: Simple Neural Network
    std::cout << "\n4. Testing Neural Network:" << std::endl;
    JNet::Network network;
    
    // Add layers: input(4) -> hidden(8, ReLU) -> output(3, Sigmoid)
    network.addLayer(new JNet::Dense(8, JNet::Activation::ReLU));
    network.addLayer(new JNet::Dense(3, JNet::Activation::Sigmoid));
    
    // Create sample input (batch_size=1, features=4)
    JNet::Tensor input_data({1, 4});
    input_data.fillRandom();
    std::cout << "Network input: " << input_data << std::endl;
    
    // Forward pass
    JNet::Tensor output = network.forward(input_data);
    std::cout << "Network output: " << output << std::endl;
    
    // Training step
    JNet::Tensor target({1, 3});
    target.fill(0.5); // Simple target
    
    std::cout << "Target: " << target << std::endl;
    
    // Train one step
    network.train(input_data, target);
    
    // Forward pass after training
    JNet::Tensor output_after = network.forward(input_data);
    std::cout << "Output after training step: " << output_after << std::endl;
    
    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    
    return 0;
}
