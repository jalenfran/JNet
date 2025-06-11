#ifndef JNET_CORE_NETWORK_H
#define JNET_CORE_NETWORK_H

// JNet Neural Network Framework
// Network class - Neural network architecture and training
//
// Provides functionality for:
// - Building neural networks with multiple layers
// - Forward and backward propagation
// - Training with automatic differentiation
// - Layer management and composition

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include "tensor.h"
#include "../layers/layer.h"
#include "../layers/dense.h"
#include "../layers/conv2d.h"
#include "../layers/flatten.h"
#include "../optimizers/optimizer.h"

namespace JNet {

// Forward declarations
class SGD;
class Adam;

class Network {
public:
    Network();
    ~Network();

    // Polymorphic layer management
    void addLayer(Layer* layer);  // Takes ownership and wraps in unique_ptr
    
    void setOptimizer(std::shared_ptr<Optimizer> optimizer);
    Tensor forward(const Tensor& input);
    void backward(const Tensor& target);
    void train(const Tensor& input, const Tensor& target);
    
    // Epoch-based training methods
    void trainEpochs(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, 
                     int epochs, bool verbose = true);
    
    // Enhanced training with progress bar and detailed statistics
    struct TrainingConfig {
        bool verbose;
        bool show_progress_bar;
        bool show_accuracy;
        bool show_learning_rate;
        int print_every;  // Print stats every N epochs
        int progress_bar_width;
        int batch_size;   // Mini-batch size for training
        
        TrainingConfig() : verbose(true), show_progress_bar(true), show_accuracy(false), 
                          show_learning_rate(false), print_every(1), progress_bar_width(50)
                         , batch_size(1) {}
        TrainingConfig(bool v) : verbose(v), show_progress_bar(true), show_accuracy(false), 
                                show_learning_rate(false), print_every(1), progress_bar_width(50)
                               , batch_size(1) {}
    };
    
    void trainEpochsAdvanced(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, 
                            int epochs, const TrainingConfig& config = TrainingConfig());
    
    void trainBatch(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets, int batch_size = 1);
    double evaluateAccuracy(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);
    
    Tensor predict(const Tensor& input);
    
    // Model persistence
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);

private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::shared_ptr<Optimizer> optimizer;
    Tensor last_output;
    
    double calculateLoss(const Tensor& predicted, const Tensor& target);
    Tensor calculateLossGradient(const Tensor& predicted, const Tensor& target);
    double calculateBatchLoss(const std::vector<Tensor>& inputs, const std::vector<Tensor>& targets);
};

}

#endif // JNET_CORE_NETWORK_H