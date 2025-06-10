#include "../include/jnet.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace JNet;

template<typename T>
void generate_sine_data(std::vector<T>& x, std::vector<Tensor>& inputs, std::vector<Tensor>& targets, int n, T noise=0.0) {
    x.resize(n);
    inputs.clear();
    targets.clear();
    for (int i = 0; i < n; ++i) {
        x[i] = (T)i / n * 2 * M_PI;
        T y = std::sin(x[i]) + noise * (((T)rand() / RAND_MAX) - 0.5);

        Tensor input({1, 1});
        input.at(0) = x[i];
        inputs.push_back(input);

        Tensor target({1, 1});
        target.at(0) = y;
        targets.push_back(target);
    }
}

int main() {
    std::cout << "=== JNet Sine Regression with Advanced Training ===" << std::endl;

    // Generate sine data
    std::vector<float> x_vals;
    std::vector<Tensor> inputs, targets;
    generate_sine_data(x_vals, inputs, targets, 512, 0.05f);

    // Build network
    Network net;
    net.addLayer(new Dense(1, 32, Activation::ReLU));
    net.addLayer(new Dense(32, 16, Activation::ReLU));
    net.addLayer(new Dense(16, 1, Activation::Linear));

    // Set optimizer
    auto optimizer = std::make_shared<Adam>(0.01f);
    net.setOptimizer(optimizer);

    // Configure training
    Network::TrainingConfig config;
    config.show_accuracy = false;
    config.print_every = 10;

    // Train
    net.trainEpochsAdvanced(inputs, targets, 100, config);

    // Evaluate final MSE
    float mse = 0.0f;
    for (int i = 0; i < inputs.size(); ++i) {
        float diff = net.predict(inputs[i]).at(0) - targets[i].at(0);
        mse += diff * diff;
    }
    mse /= inputs.size();

    std::cout << "\nFinal MSE: " << std::fixed << std::setprecision(6) << mse << std::endl;

    return 0;
}
