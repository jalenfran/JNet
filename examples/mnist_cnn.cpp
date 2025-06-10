#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <algorithm>
#include "../include/jnet.h"

using namespace JNet;

// MNIST data loader
class MNISTLoader {
public:
    static std::vector<Tensor> loadImages(const std::string& filename, int max_samples = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // Read header
        uint32_t magic_number = 0;
        uint32_t num_images = 0;
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;
        
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&num_rows), 4);
        file.read(reinterpret_cast<char*>(&num_cols), 4);
        
        // Convert from big endian to little endian
        magic_number = reverseInt(magic_number);
        num_images = reverseInt(num_images);
        num_rows = reverseInt(num_rows);
        num_cols = reverseInt(num_cols);
        
        if (magic_number != 2051) {
            throw std::runtime_error("Invalid MNIST image file!");
        }
        
        // Limit number of samples if specified
        if (max_samples > 0 && max_samples < static_cast<int>(num_images)) {
            num_images = max_samples;
        }
        
        std::cout << "Loading " << num_images << " images of size " << num_rows << "x" << num_cols << std::endl;
        
        std::vector<Tensor> images;
        images.reserve(num_images);
        
        for (uint32_t i = 0; i < num_images; ++i) {
            Tensor image({1, static_cast<int>(num_rows), static_cast<int>(num_cols)});
            double* data = image.data_ptr();
            
            for (uint32_t j = 0; j < num_rows * num_cols; ++j) {
                unsigned char temp = 0;
                file.read(reinterpret_cast<char*>(&temp), 1);
                data[j] = static_cast<double>(temp) / 255.0; // Normalize to [0, 1]
            }
            
            images.push_back(image);
        }
        
        file.close();
        return images;
    }
    
    static std::vector<int> loadLabels(const std::string& filename, int max_samples = -1) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // Read header
        uint32_t magic_number = 0;
        uint32_t num_labels = 0;
        
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);
        
        // Convert from big endian to little endian
        magic_number = reverseInt(magic_number);
        num_labels = reverseInt(num_labels);
        
        if (magic_number != 2049) {
            throw std::runtime_error("Invalid MNIST label file!");
        }
        
        // Limit number of samples if specified
        if (max_samples > 0 && max_samples < static_cast<int>(num_labels)) {
            num_labels = max_samples;
        }
        
        std::vector<int> labels;
        labels.reserve(num_labels);
        
        for (uint32_t i = 0; i < num_labels; ++i) {
            unsigned char temp = 0;
            file.read(reinterpret_cast<char*>(&temp), 1);
            labels.push_back(static_cast<int>(temp));
        }
        
        file.close();
        return labels;
    }
    
    static std::vector<Tensor> labelsToOneHot(const std::vector<int>& labels, int num_classes = 10) {
        std::vector<Tensor> one_hot_labels;
        one_hot_labels.reserve(labels.size());
        
        for (int label : labels) {
            Tensor one_hot({1, num_classes});
            one_hot.fill(0.0);
            one_hot[{0, label}] = 1.0;
            one_hot_labels.push_back(one_hot);
        }
        
        return one_hot_labels;
    }
    
private:
    static uint32_t reverseInt(uint32_t i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
    }
};

// Function to print MNIST digit
void printMNISTDigit(const Tensor& image, int actual_digit) {
    std::cout << "Digit " << actual_digit << ":\n";
    const std::vector<int>& shape = image.shape();
    for (int h = 0; h < shape[1]; ++h) {
        for (int w = 0; w < shape[2]; ++w) {
            double val = image[{0, h, w}];
            if (val > 0.7) {
                std::cout << "██";
            } else if (val > 0.4) {
                std::cout << "▓▓";
            } else if (val > 0.1) {
                std::cout << "░░";
            } else {
                std::cout << "  ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Function to download MNIST data if not present
void downloadMNISTIfNeeded() {
    std::vector<std::string> files = {
        "examples/data/train-images.idx3-ubyte",
        "examples/data/train-labels.idx1-ubyte", 
        "examples/data/t10k-images.idx3-ubyte",
        "examples/data/t10k-labels.idx1-ubyte"
    };
    
    bool all_exist = true;
    for (const auto& file : files) {
        std::ifstream f(file);
        if (!f.good()) {
            all_exist = false;
            break;
        }
    }
    
    if (!all_exist) {
        std::cout << "MNIST files not found. Please download them manually:\n\n";
        std::cout << "1. Go to: http://yann.lecun.com/exdb/mnist/\n";
        std::cout << "2. Download these files to the current directory:\n";
        std::cout << "   - train-images-idx3-ubyte.gz\n";
        std::cout << "   - train-labels-idx1-ubyte.gz\n";
        std::cout << "   - t10k-images-idx3-ubyte.gz\n";
        std::cout << "   - t10k-labels-idx1-ubyte.gz\n";
        std::cout << "3. Extract them: gunzip *.gz\n\n";
        
        std::cout << "Or use these commands:\n";
        std::cout << "curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n";
        std::cout << "curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n";
        std::cout << "curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n";
        std::cout << "curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n";
        std::cout << "gunzip *.gz\n\n";
        
        throw std::runtime_error("MNIST data files not found. Please download them first.");
    }
}

int main() {
    std::cout << "=== JNet MNIST Digit Recognition CNN ===\n\n";
    
    try {
        // Check if MNIST files exist
        downloadMNISTIfNeeded();
        
        // Load MNIST data (limit to 1000 samples for faster training)
        std::cout << "Loading MNIST training data...\n";
        // Load MNIST data (limit to 1000 samples for faster training)
        std::cout << "Loading MNIST training data...\n";
        std::vector<Tensor> train_images = MNISTLoader::loadImages("examples/data/train-images.idx3-ubyte", 1000);
        std::vector<int> train_labels_int = MNISTLoader::loadLabels("examples/data/train-labels.idx1-ubyte", 1000);
        std::vector<Tensor> train_labels = MNISTLoader::labelsToOneHot(train_labels_int);
        
        std::cout << "Loading MNIST test data...\n";
        std::vector<Tensor> test_images = MNISTLoader::loadImages("examples/data/t10k-images.idx3-ubyte", 200);
        std::vector<int> test_labels_int = MNISTLoader::loadLabels("examples/data/t10k-labels.idx1-ubyte", 200);
        
        std::cout << "Dataset loaded successfully!\n";
        std::cout << "Training samples: " << train_images.size() << std::endl;
        std::cout << "Test samples: " << test_images.size() << std::endl;
        std::cout << "Image size: " << train_images[0].shape()[1] << "x" << train_images[0].shape()[2] << std::endl << std::endl;
        
        // Show some sample digits
        std::cout << "=== Sample MNIST Digits ===\n";
        for (int i = 0; i < 5; ++i) {
            printMNISTDigit(train_images[i], train_labels_int[i]);
        }
        
        // Create CNN for MNIST (28x28 images)
        Network cnn;
        
        std::cout << "Building CNN architecture for MNIST...\n";
        
        // CNN architecture optimized for 28x28 MNIST images
        cnn.addLayer(new Conv2D(1, 32, 3, 1, 1, Activation::ReLU));  // 28x28 -> 28x28
        std::cout << "Added Conv2D layer: 32 filters, 3x3 kernel\n";
        
        cnn.addLayer(new MaxPool2D(2, 2));  // 28x28 -> 14x14
        std::cout << "Added MaxPool2D layer: 2x2 pooling -> 14x14\n";
        
        cnn.addLayer(new Conv2D(32, 64, 3, 1, 1, Activation::ReLU));  // 14x14 -> 14x14
        std::cout << "Added Conv2D layer: 64 filters, 3x3 kernel\n";
        
        cnn.addLayer(new MaxPool2D(2, 2));  // 14x14 -> 7x7
        std::cout << "Added MaxPool2D layer: 2x2 pooling -> 7x7\n";
        
        cnn.addLayer(new Flatten());
        std::cout << "Added Flatten layer\n";
        
        cnn.addLayer(new Dense(128, Activation::ReLU));  // 7*7*64 = 3136 -> 128
        std::cout << "Added Dense layer: 128 neurons\n";
        
        cnn.addLayer(new Dense(10, Activation::Linear));  // 10 digit classes
        std::cout << "Added Dense layer: 10 neurons (digits 0-9)\n";
        
        // Use SGD optimizer with good learning rate for MNIST
        auto optimizer = std::make_shared<SGD>(0.01);
        cnn.setOptimizer(optimizer);
        std::cout << "Set SGD optimizer with learning rate 0.01\n\n";
        
        // Test forward pass
        std::cout << "Testing forward pass...\n";
        Tensor output = cnn.forward(train_images[0]);
        std::cout << "Forward pass successful!\n";
        std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]\n\n";
        
        // Show initial predictions (before training)
        std::cout << "=== Initial Predictions (before training) ===\n";
        for (int i = 0; i < 5; ++i) {
            Tensor prediction = cnn.predict(train_images[i]);
            int predicted_class = 0;
            for (int j = 1; j < prediction.size(); ++j) {
                if (prediction.at(j) > prediction.at(predicted_class)) {
                    predicted_class = j;
                }
            }
            std::cout << "Sample " << (i+1) << " - Predicted: " << predicted_class 
                      << ", Actual: " << train_labels_int[i] << std::endl;
        }
        std::cout << std::endl;
        
        // Train the CNN
        std::cout << "=== Training CNN on MNIST Data ===\n";
        
        Network::TrainingConfig config;
        config.verbose = true;
        config.show_progress_bar = true;
        config.show_accuracy = true;
        config.print_every = 2;  // Print every 2 epochs
        config.progress_bar_width = 50;
        
        // Train for moderate number of epochs
        cnn.trainEpochsAdvanced(train_images, train_labels, 10, config);
        
        // Test on training set (first 20 samples)
        std::cout << "\n=== Training Set Predictions (after training) ===\n";
        int train_correct = 0;
        for (int i = 0; i < std::min(20, static_cast<int>(train_images.size())); ++i) {
            Tensor prediction = cnn.predict(train_images[i]);
            int predicted_class = 0;
            for (int j = 1; j < prediction.size(); ++j) {
                if (prediction.at(j) > prediction.at(predicted_class)) {
                    predicted_class = j;
                }
            }
            
            bool is_correct = (predicted_class == train_labels_int[i]);
            if (is_correct) train_correct++;
            
            std::cout << "Sample " << (i+1) << " - Predicted: " << predicted_class 
                      << ", Actual: " << train_labels_int[i];
            if (is_correct) {
                std::cout << " ✓";
            } else {
                std::cout << " ✗";
            }
            std::cout << std::endl;
        }
        
        // Test on test set
        std::cout << "\n=== Test Set Predictions ===\n";
        int test_correct = 0;
        int test_samples = std::min(20, static_cast<int>(test_images.size()));
        for (int i = 0; i < test_samples; ++i) {
            Tensor prediction = cnn.predict(test_images[i]);
            int predicted_class = 0;
            for (int j = 1; j < prediction.size(); ++j) {
                if (prediction.at(j) > prediction.at(predicted_class)) {
                    predicted_class = j;
                }
            }
            
            bool is_correct = (predicted_class == test_labels_int[i]);
            if (is_correct) test_correct++;
            
            std::cout << "Test " << (i+1) << " - Predicted: " << predicted_class 
                      << ", Actual: " << test_labels_int[i];
            if (is_correct) {
                std::cout << " ✓";
            } else {
                std::cout << " ✗";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n=== Final Results ===\n";
        std::cout << "Training Accuracy: " << (train_correct * 100.0 / 20) << "% (" 
                  << train_correct << "/20 samples correct)\n";
        std::cout << "Test Accuracy: " << (test_correct * 100.0 / test_samples) << "% (" 
                  << test_correct << "/" << test_samples << " samples correct)\n";
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n=== MNIST CNN completed successfully! ===\n";
    std::cout << "\nThis example demonstrates:\n";
    std::cout << "- Loading real MNIST handwritten digit data\n";
    std::cout << "- Training CNN on actual digit recognition task\n";
    std::cout << "- Evaluation on separate test set\n";
    std::cout << "- Realistic performance metrics on real data\n";
    std::cout << "- Optimized performance with fast tensor operations\n";
    
    return 0;
}
