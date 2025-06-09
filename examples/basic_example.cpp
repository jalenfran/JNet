#include <iostream>
#include "jnet.h"

int main() {
    // Create a simple neural network
    JNet::Network network;

    // Add layers to the network
    network.addLayer(new JNet::Dense(10, JNet::Activation::ReLU));
    network.addLayer(new JNet::Dense(1, JNet::Activation::Sigmoid));

    // Create a sample input tensor
    JNet::Tensor input({1, 10});
    input.fillRandom(); 

    // Forward pass
    JNet::Tensor output = network.forward(input);

    // Print the output
    std::cout << "Output: " << output << std::endl;

    // Assuming we have some target values for training
    JNet::Tensor target({1, 1});
    target.fill(1.0);

    // Backward pass and update weights
    network.backward(target);

    return 0;
}