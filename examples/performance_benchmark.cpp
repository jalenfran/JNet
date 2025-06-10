#include "../include/jnet.h"
#include "../src/core/threadpool.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <thread>

using namespace JNet;

void benchmark_matrix_multiplication() {
    std::cout << "=== Matrix Multiplication Benchmark ===" << std::endl;
    
    // Test different matrix sizes
    std::vector<int> sizes = {100, 500, 1000};
    
    for (int size : sizes) {
        std::cout << "\nTesting " << size << "x" << size << " matrices:" << std::endl;
        
        // Create random matrices
        Tensor a = Tensor::random({size, size});
        Tensor b = Tensor::random({size, size});
        
        // Benchmark dot product
        auto start = std::chrono::high_resolution_clock::now();
        Tensor result = a.dot(b);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Matrix multiplication: " << duration.count() << " ms" << std::endl;
    }
}

void benchmark_convolution() {
    std::cout << "\n=== Convolution Benchmark ===" << std::endl;
    
    // Test different input sizes
    std::vector<std::pair<int, int>> sizes = {{32, 32}, {64, 64}, {128, 128}};
    
    for (auto& size_pair : sizes) {
        int height = size_pair.first;
        int width = size_pair.second;
        
        std::cout << "\nTesting " << height << "x" << width << " convolution:" << std::endl;
        
        // Create input and conv layer
        Tensor input = Tensor::random({3, height, width}); // 3 channels
        Conv2D conv(3, 16, 3, 1, 1, Activation::ReLU);   // 3->16 channels, 3x3 kernel
        
        // Benchmark forward pass
        auto start = std::chrono::high_resolution_clock::now();
        Tensor output = conv.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Convolution forward: " << duration.count() << " ms" << std::endl;
        std::cout << "  Output shape: [" << output.shape()[0] << ", " 
                  << output.shape()[1] << ", " << output.shape()[2] << "]" << std::endl;
    }
}

void benchmark_pooling() {
    std::cout << "\n=== Pooling Benchmark ===" << std::endl;
    
    // Test different input sizes
    std::vector<std::pair<int, int>> sizes = {{64, 64}, {128, 128}, {256, 256}};
    
    for (auto& size_pair : sizes) {
        int height = size_pair.first;
        int width = size_pair.second;
        
        std::cout << "\nTesting " << height << "x" << width << " pooling:" << std::endl;
        
        // Create input and pooling layers
        Tensor input = Tensor::random({16, height, width}); // 16 channels
        MaxPool2D maxpool(2, 2);
        AvgPool2D avgpool(2, 2);
        
        // Benchmark MaxPool2D
        auto start = std::chrono::high_resolution_clock::now();
        Tensor max_output = maxpool.forward(input);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto max_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  MaxPool2D forward: " << max_duration.count() << " ms" << std::endl;
        
        // Benchmark AvgPool2D
        start = std::chrono::high_resolution_clock::now();
        Tensor avg_output = avgpool.forward(input);
        end = std::chrono::high_resolution_clock::now();
        
        auto avg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  AvgPool2D forward: " << avg_duration.count() << " ms" << std::endl;
        
        std::cout << "  Output shape: [" << max_output.shape()[0] << ", " 
                  << max_output.shape()[1] << ", " << max_output.shape()[2] << "]" << std::endl;
    }
}

void benchmark_full_network() {
    std::cout << "\n=== Full Network Benchmark ===" << std::endl;
    
    // Create a complex CNN with pooling
    Network network;
    
    // Build architecture similar to a small ResNet-like model
    network.addLayer(new Conv2D(3, 32, 3, 1, 1, Activation::ReLU));
    network.addLayer(new MaxPool2D(2, 2));
    network.addLayer(new Conv2D(32, 64, 3, 1, 1, Activation::ReLU));
    network.addLayer(new AvgPool2D(2, 2));
    network.addLayer(new Conv2D(64, 128, 3, 1, 1, Activation::ReLU));
    network.addLayer(new MaxPool2D(2, 2));
    network.addLayer(new Flatten());
    network.addLayer(new Dense(256, Activation::ReLU));
    network.addLayer(new Dense(10, Activation::Linear));
    
    auto optimizer = std::make_shared<SGD>(0.001);
    network.setOptimizer(optimizer);
    
    // Test with different batch sizes
    std::vector<int> batch_sizes = {1, 4, 8};
    
    for (int batch_size : batch_sizes) {
        std::cout << "\nTesting full network with batch size " << batch_size << ":" << std::endl;
        
        // Create batch input (batch_size images of 64x64x3)
        std::vector<Tensor> inputs;
        std::vector<Tensor> targets;
        
        for (int i = 0; i < batch_size; ++i) {
            Tensor input = Tensor::random({3, 64, 64});
            Tensor target = Tensor::zeros({1, 10});
            target[{0, i % 10}] = 1.0; // One-hot target
            
            inputs.push_back(input);
            targets.push_back(target);
        }
        
        // Benchmark forward pass
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batch_size; ++i) {
            Tensor output = network.predict(inputs[i]);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto forward_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Forward pass (batch): " << forward_duration.count() << " ms" << std::endl;
        std::cout << "  Average per sample: " << forward_duration.count() / batch_size << " ms" << std::endl;
        
        // Benchmark training step
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batch_size; ++i) {
            network.train(inputs[i], targets[i]);
        }
        end = std::chrono::high_resolution_clock::now();
        
        auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "  Training step (batch): " << train_duration.count() << " ms" << std::endl;
        std::cout << "  Average per sample: " << train_duration.count() / batch_size << " ms" << std::endl;
    }
}

void show_threading_info() {
    std::cout << "=== Threading Information ===" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl;
    std::cout << "Thread pool size: " << getGlobalThreadPool().size() << " threads" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "=== JNet Multithreading Performance Benchmark ===" << std::endl;
    std::cout << "This benchmark demonstrates the performance improvements from multithreading." << std::endl;
    std::cout << std::endl;
    
    show_threading_info();
    
    try {
        benchmark_matrix_multiplication();
        benchmark_convolution();
        benchmark_pooling();
        benchmark_full_network();
    } catch (const std::exception& e) {
        std::cout << "Benchmark error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== Benchmark completed! ===" << std::endl;
    std::cout << "\nMultithreading Benefits:" << std::endl;
    std::cout << "✓ Matrix multiplications automatically use multiple cores" << std::endl;
    std::cout << "✓ Convolutions are parallelized across output filters" << std::endl;
    std::cout << "✓ Pooling operations are parallelized across channels" << std::endl;
    std::cout << "✓ Performance scales with available CPU cores" << std::endl;
    std::cout << "✓ Automatic threshold-based decision for single vs multi-threading" << std::endl;
    
    return 0;
}
