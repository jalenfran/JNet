#include "network.h"
#include "../layers/conv2d.h"
#include "../layers/flatten.h"
#include "../layers/pooling.h"
#include "../optimizers/sgd.h"
#include "../optimizers/adam.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace JNet {

Network::Network() {
}

Network::~Network() {
    // Smart pointers will automatically clean up
}

void Network::addLayer(Layer* layer) {
    layers.push_back(std::unique_ptr<Layer>(layer));
}

void Network::setOptimizer(std::shared_ptr<Optimizer> opt) {
    this->optimizer = opt;
    for (auto& layer : layers) {
        layer->setOptimizer(opt);
    }
}

Tensor Network::forward(const Tensor& input) {
    Tensor output = input;
    for (auto& layer : layers) {
        output = layer->forward(output);
    }
    last_output = output;
    return output;
}

void Network::backward(const Tensor& target) {
    // Calculate loss gradient (using mean squared error)
    Tensor loss_gradient = calculateLossGradient(last_output, target);
    
    // Backpropagate through layers in reverse order
    Tensor gradient = loss_gradient;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        gradient = (*it)->backward(gradient);
    }
}

void Network::train(const Tensor& input, const Tensor& target) {
    // Forward pass
    forward(input);
    
    // Backward pass
    backward(target);
}

Tensor Network::predict(const Tensor& input) {
    return forward(input);
}

void Network::trainEpochs(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, 
                         int epochs, bool verbose) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    if (verbose) {
        std::cout << "Starting training for " << epochs << " epochs with " << inputs.size() << " samples\n";
        std::cout << "Epoch\tLoss\n";
        std::cout << "-----\t----\n";
    }
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Train on all samples
        trainBatch(inputs, targets);
        
        // Calculate and display loss if verbose
        if (verbose) {
            double loss = calculateBatchLoss(inputs, targets);
            std::cout << epoch + 1 << "\t" << loss << std::endl;
        }
    }
    
    if (verbose) {
        std::cout << "Training completed!\n";
    }
}

void Network::trainEpochsAdvanced(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, 
                                 int epochs, const TrainingConfig& config) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    if (config.verbose) {
        std::cout << "=== JNet Training Started ===" << std::endl;
        std::cout << "Epochs: " << epochs << " | Samples: " << inputs.size() << std::endl;
        if (optimizer) {
            if (auto sgd = std::dynamic_pointer_cast<SGD>(optimizer)) {
                std::cout << "Optimizer: SGD (lr=" << sgd->getLearningRate() << ")" << std::endl;
            } else if (auto adam = std::dynamic_pointer_cast<Adam>(optimizer)) {
                std::cout << "Optimizer: Adam (lr=" << adam->getLearningRate() << ")" << std::endl;
            } else {
                std::cout << "Optimizer: Custom" << std::endl;
            }
        }
        std::cout << std::string(40, '=') << std::endl;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Train on all samples
        auto epoch_start = std::chrono::high_resolution_clock::now();
        trainBatch(inputs, targets);
        auto epoch_end = std::chrono::high_resolution_clock::now();
        
        // Calculate metrics if needed
        bool should_print = config.verbose && ((epoch + 1) % config.print_every == 0 || epoch == epochs - 1);
        
        if (should_print) {
            double loss = calculateBatchLoss(inputs, targets);
            double accuracy = config.show_accuracy ? evaluateAccuracy(inputs, targets) : 0.0;
            
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(epoch_end - start_time);
            
            // Progress bar
            if (config.show_progress_bar) {
                int progress = static_cast<int>((static_cast<double>(epoch + 1) / epochs) * config.progress_bar_width);
                std::cout << "\rEpoch " << std::setw(4) << (epoch + 1) << "/" << epochs << " [";
                for (int i = 0; i < config.progress_bar_width; ++i) {
                    if (i < progress) {
                        std::cout << "█";
                    } else if (i == progress) {
                        std::cout << "▌";
                    } else {
                        std::cout << " ";
                    }
                }
                std::cout << "] ";
                
                // Progress percentage
                double percent = (static_cast<double>(epoch + 1) / epochs) * 100.0;
                std::cout << std::fixed << std::setprecision(1) << percent << "% ";
                
                // Loss
                std::cout << "Loss: " << std::fixed << std::setprecision(6) << loss;
                
                // Accuracy if requested
                if (config.show_accuracy) {
                    std::cout << " Acc: " << std::fixed << std::setprecision(2) << (accuracy * 100) << "%";
                }
                
                // Timing
                std::cout << " (" << epoch_duration.count() << "ms/epoch)";
                
                // ETA
                if (epoch < epochs - 1) {
                    int remaining_epochs = epochs - epoch - 1;
                    auto eta_ms = epoch_duration.count() * remaining_epochs;
                    if (eta_ms > 1000) {
                        std::cout << " ETA: " << (eta_ms / 1000) << "s";
                    } else {
                        std::cout << " ETA: " << eta_ms << "ms";
                    }
                }
                
                std::cout << std::flush;
            } else {
                // Simple line output without progress bar
                std::cout << "Epoch " << (epoch + 1) << "/" << epochs 
                         << " - Loss: " << std::fixed << std::setprecision(6) << loss;
                if (config.show_accuracy) {
                    std::cout << " - Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100) << "%";
                }
                std::cout << std::endl;
            }
        }
    }
    
    if (config.verbose) {
        auto total_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_time - start_time);
        
        std::cout << std::endl << std::string(40, '=') << std::endl;
        std::cout << "Training completed!" << std::endl;
        std::cout << "Total time: " << (duration.count() / 1000.0) << "s" << std::endl;
        std::cout << "Average time per epoch: " << (duration.count() / epochs) << "ms" << std::endl;
        
        // Final metrics
        double final_loss = calculateBatchLoss(inputs, targets);
        double final_accuracy = evaluateAccuracy(inputs, targets);
        std::cout << "Final loss: " << std::fixed << std::setprecision(6) << final_loss << std::endl;
        std::cout << "Final accuracy: " << std::fixed << std::setprecision(2) << (final_accuracy * 100) << "%" << std::endl;
    }
}

void Network::trainBatch(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int batch_size) {
    size_t total = inputs.size();
    for (size_t start = 0; start < total; start += batch_size) {
        size_t end = std::min(start + batch_size, total);
        // Process each sample in the mini-batch
        for (size_t i = start; i < end; ++i) {
            train(inputs[i], targets[i]);
        }
    }
}

double Network::evaluateAccuracy(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
    if (inputs.size() != targets.size()) {
        throw std::invalid_argument("Number of inputs must match number of targets");
    }
    
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction = predict(inputs[i]);
        
        // Find predicted class (highest value)
        int predicted_class = 0;
        int actual_class = 0;
        
        for (int j = 1; j < prediction.size(); ++j) {
            if (prediction.at(j) > prediction.at(predicted_class)) {
                predicted_class = j;
            }
            if (targets[i].at(j) > targets[i].at(actual_class)) {
                actual_class = j;
            }
        }
        
        if (predicted_class == actual_class) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / inputs.size();
}

double Network::calculateBatchLoss(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets) {
    double total_loss = 0.0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Tensor prediction = predict(inputs[i]);
        total_loss += calculateLoss(prediction, targets[i]);
    }
    return total_loss / inputs.size();
}

double Network::calculateLoss(const Tensor& predicted, const Tensor& target) {
    if (predicted.shape() != target.shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }
    
    double loss = 0.0;
    for (int i = 0; i < predicted.size(); ++i) {
        double diff = predicted.at(i) - target.at(i);
        loss += diff * diff;
    }
    return loss / (2.0 * predicted.size()); // Mean squared error
}

Tensor Network::calculateLossGradient(const Tensor& predicted, const Tensor& target) {
    if (predicted.shape() != target.shape()) {
        throw std::invalid_argument("Predicted and target tensors must have the same shape");
    }
    
    Tensor gradient = predicted;
    for (int i = 0; i < gradient.size(); ++i) {
        gradient.at(i) = (predicted.at(i) - target.at(i)) / predicted.size();
    }
    return gradient;
}

void Network::saveModel(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for saving: " + filename);
    }
    
    // Write number of layers
    size_t num_layers = layers.size();
    file.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    
    // Write each layer
    for (const auto& layer : layers) {
        // Write layer type
        std::string layer_type = layer->getLayerType();
        size_t type_length = layer_type.length();
        file.write(reinterpret_cast<const char*>(&type_length), sizeof(type_length));
        file.write(layer_type.c_str(), type_length);
        
        // Write layer parameters
        std::vector<Tensor> parameters = layer->getParameters();
        size_t num_params = parameters.size();
        file.write(reinterpret_cast<const char*>(&num_params), sizeof(num_params));
        
        for (const auto& param : parameters) {
            // Write tensor shape
            std::vector<int> shape = param.shape();
            size_t shape_size = shape.size();
            file.write(reinterpret_cast<const char*>(&shape_size), sizeof(shape_size));
            file.write(reinterpret_cast<const char*>(shape.data()), shape_size * sizeof(int));
            
            // Write tensor data
            size_t data_size = param.size();
            file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
            for (int i = 0; i < data_size; ++i) {
                double value = param.at(i);
                file.write(reinterpret_cast<const char*>(&value), sizeof(value));
            }
        }
    }
    
    file.close();
    std::cout << "Model saved to: " << filename << std::endl;
}

void Network::loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for loading: " + filename);
    }
    
    // Clear existing layers
    layers.clear();
    
    // Read number of layers
    size_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    
    // Read each layer
    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        // Read layer type
        size_t type_length;
        file.read(reinterpret_cast<char*>(&type_length), sizeof(type_length));
        std::string layer_type(type_length, '\0');
        file.read(&layer_type[0], type_length);
        
        // Create layer based on type (simplified - would need more info for full reconstruction)
        std::unique_ptr<Layer> layer;
        if (layer_type == "Dense") {
            layer = std::make_unique<Dense>(1); // Temporary size, will be set by parameters
        } else if (layer_type == "Conv2D") {
            layer = std::make_unique<Conv2D>(1, 3); // Temporary values, will be set by parameters  
        } else if (layer_type == "Flatten") {
            layer = std::make_unique<Flatten>();
        } else if (layer_type == "MaxPool2D") {
            layer = std::make_unique<MaxPool2D>(2); // Temporary value
        } else if (layer_type == "AvgPool2D") {
            layer = std::make_unique<AvgPool2D>(2); // Temporary value
        } else {
            throw std::runtime_error("Unknown layer type: " + layer_type);
        }
        
        // Read layer parameters
        size_t num_params;
        file.read(reinterpret_cast<char*>(&num_params), sizeof(num_params));
        
        std::vector<Tensor> parameters;
        for (size_t param_idx = 0; param_idx < num_params; ++param_idx) {
            // Read tensor shape
            size_t shape_size;
            file.read(reinterpret_cast<char*>(&shape_size), sizeof(shape_size));
            std::vector<int> shape(shape_size);
            file.read(reinterpret_cast<char*>(shape.data()), shape_size * sizeof(int));
            
            // Read tensor data
            size_t data_size;
            file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
            
            Tensor param(shape);
            for (size_t i = 0; i < data_size; ++i) {
                double value;
                file.read(reinterpret_cast<char*>(&value), sizeof(value));
                param.at(i) = value;
            }
            parameters.push_back(param);
        }
        
        // Set parameters to layer
        if (num_params > 0) {
            layer->setParameters(parameters);
        }
        
        layers.push_back(std::move(layer));
    }
    
    file.close();
    std::cout << "Model loaded from: " << filename << std::endl;
}

}