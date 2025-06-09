# JNet - Custom Neural Network Framework in C++

JNet is a lightweight, custom neural network framework written in C++ from scratch. It provides basic building blocks for creating and training neural networks with modern optimization algorithms.

## Features

- **Tensor Operations**: Multi-dimensional array operations with support for basic math
- **Neural Network Layers**: 
  - Dense (fully connected) layers with various activation functions
  - **Convolutional layers (Conv2D)** with configurable filters, kernels, stride, and padding
  - **Flatten layers** for bridging between Conv2D and Dense layers
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, Linear
- **Matrix Operations**: Dot product, transpose, element-wise operations
- **Advanced Optimizers**: 
  - SGD (Stochastic Gradient Descent) with configurable learning rates
  - Adam optimizer with adaptive learning rates and momentum
  - Polymorphic optimizer architecture for easy extensibility
- **Smart State Management**: Separate optimizer instances for weights and biases
- **Polymorphic Layer Architecture**: Single `addLayer()` method handles all layer types
- **Simple API**: Easy-to-use interface for building and training neural networks

## Building

```bash
cd JNet
mkdir build && cd build
cmake ..
make
```

## Usage

### Basic Neural Network Example

```cpp
#include "jnet.h"
using namespace JNet;

int main() {
    // Create a neural network
    Network network;
    
    // Add layers: input -> hidden(10, ReLU) -> output(1, Sigmoid)
    network.addLayer(new Dense(10, Activation::ReLU));
    network.addLayer(new Dense(1, Activation::Sigmoid));
    
    // Set optimizer (SGD or Adam)
    auto optimizer = std::make_shared<Adam>(0.001);  // Adam with lr=0.001
    network.setOptimizer(optimizer);
    
    // Create input data (batch_size=1, features=4)
    Tensor input({1, 4});
    input.fillRandom();
    
    // Forward pass
    Tensor output = network.forward(input);
    
    // Training
    Tensor target({1, 1});
    target.fill(1.0);
    
    network.train(input, target);
    
    return 0;
}
```

### Convolutional Neural Network Example

```cpp
#include "jnet.h"
using namespace JNet;

int main() {
    // Create a CNN for image classification
    Network cnn;
    
    // Convolutional layers
    cnn.addLayer(new Conv2D(1, 32, 3, 1, 1, Activation::ReLU));  // 1->32 channels, 3x3 kernel
    cnn.addLayer(new Conv2D(32, 64, 3, 1, 1, Activation::ReLU)); // 32->64 channels, 3x3 kernel
    
    // Flatten for dense layers
    cnn.addLayer(new Flatten());
    
    // Dense layers for classification
    cnn.addLayer(new Dense(128, Activation::ReLU));
    cnn.addLayer(new Dense(10, Activation::Linear));  // 10 classes
    
    // Set optimizer
    auto optimizer = std::make_shared<SGD>(0.001);
    cnn.setOptimizer(optimizer);
    
    // Create input image (1 channel, 16x16)
    Tensor input({1, 16, 16});
    input.fillRandom();
    
    // Create target (one-hot encoded)
    Tensor target({1, 10});
    target.fill(0.0);
    target[{0, 3}] = 1.0;  // Class 3
    
    // Train the network
    cnn.train(input, target);
    
    return 0;
}
```

### Optimizer Comparison Example

```cpp
#include "jnet.h"
using namespace JNet;

int main() {
    // Training data for XOR problem
    std::vector<Tensor> inputs, targets;
    // ... setup XOR data ...
    
    // Test SGD optimizer
    Network sgd_network;
    sgd_network.addLayer(new Dense(2, 4, Activation::ReLU));
    sgd_network.addLayer(new Dense(4, 1, Activation::Sigmoid));
    auto sgd_optimizer = std::make_shared<SGD>(0.1);
    sgd_network.setOptimizer(sgd_optimizer);
    sgd_network.trainBatch(inputs, targets);
    
    // Test Adam optimizer  
    Network adam_network;
    adam_network.addLayer(new Dense(2, 4, Activation::ReLU));
    adam_network.addLayer(new Dense(4, 1, Activation::Sigmoid));
    auto adam_optimizer = std::make_shared<Adam>(0.01);
    adam_network.setOptimizer(adam_optimizer);
    adam_network.trainBatch(inputs, targets);
    
    return 0;
}
```

## Running Examples

```bash
# Basic neural network example
./build/example

# Comprehensive test with multiple layers
./build/simple_test

# Epoch-based training example
./build/epoch_example

# Optimizer comparison (SGD vs Adam)
./build/optimizer_comparison

# Convolutional Neural Network examples
./build/cnn_example    # 16x16 image classification
```

## Architecture Overview

### Layer Architecture
JNet features a clean polymorphic layer system:

- **Base Layer Class**: Abstract interface for all neural network layers
- **Dense Layers**: Fully connected layers with configurable activations
- **Convolutional Layers (Conv2D)**: 2D convolution with multiple filters, configurable kernel size, stride, and padding
- **Flatten Layers**: Reshape multi-dimensional tensors for dense layer input
- **Unified Interface**: Single `addLayer()` method handles all layer types polymorphically

### Optimizer Integration
JNet features a sophisticated optimizer architecture:

- **Base Optimizer Class**: Polymorphic interface for all optimization algorithms
- **SGD Optimizer**: Classic stochastic gradient descent with configurable learning rates
- **Adam Optimizer**: Advanced adaptive optimization with momentum and bias correction
- **Smart State Management**: Each parameter tensor (weights/biases) gets its own optimizer state
- **Network-Level Configuration**: Set optimizers across all layers with `network.setOptimizer()`

### Key Design Decisions

1. **Separate Optimizer Instances**: Weights and biases each get their own optimizer instance to prevent state conflicts
2. **Map-Based State Management**: Adam optimizer maintains separate momentum vectors for each parameter tensor
3. **Polymorphic Architecture**: Easy to add new optimizers by inheriting from the base `Optimizer` class

## Current Status

✅ **Fully Working Features:**
- Complete tensor operations (creation, arithmetic, matrix multiplication)
- **Polymorphic Layer System:**
  - Dense layers with multiple activation functions (ReLU, Sigmoid, Tanh, Linear)
  - **Convolutional layers (Conv2D)** with configurable filters, kernels, stride, and padding
  - **Flatten layers** for seamless CNN-to-Dense transitions
  - Unified `addLayer()` interface for all layer types
- Forward and backward propagation with automatic differentiation
- **Advanced Optimizer Integration:**
  - SGD optimizer with configurable learning rates
  - Adam optimizer with adaptive learning rates, momentum, and bias correction
  - Polymorphic optimizer architecture
  - Separate state management for weights and biases
  - Network-level optimizer configuration
- Batch training and epoch-based training
- **CNN Support**: Complete convolutional neural networks for image processing
- XOR problem solver (demonstrates non-linear learning capability)
- Comprehensive error handling and validation

🔧 **Recent Major Updates:**
- **Convolutional Layer Support**: Full Conv2D implementation with forward/backward propagation
- **Polymorphic Layer Architecture**: Unified interface for all layer types with single `addLayer()` method
- **Flatten Layer**: Seamless bridge between convolutional and dense layers
- **Enhanced Activation Support**: Added Linear activation for output layers
- **CNN Examples**: Complete working examples for image classification tasks
- **Optimizer Architecture Overhaul**: Implemented polymorphic base class for optimizers
- **Adam Optimizer**: Full implementation with momentum, adaptive learning rates, and proper state management
- **State Management**: Fixed tensor shape conflicts with map-based approach for parameter-specific states
- **Integration**: Seamless optimizer switching at network level

## Future Enhancements

- [ ] **Pooling layers** (MaxPool, AvgPool) for CNNs
- [ ] **Advanced CNN architectures** (stride > 1, dilated convolutions)
- [ ] LSTM/RNN layers for sequence data
- [ ] GPU support with CUDA
- [ ] More optimizers (RMSprop, AdaGrad)
- [ ] Regularization techniques (Dropout, BatchNorm)
- [ ] Model serialization/loading
- [ ] Advanced loss functions (CrossEntropy, etc.)
- [ ] Batch processing optimization

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.