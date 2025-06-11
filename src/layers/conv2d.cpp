#include "conv2d.h"
#include "activation.h"
#include "../core/threadpool.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <algorithm>
#include "../optimizers/sgd.h"

namespace JNet {

static void conv2d_im2col(const double* data_im, int channels, int padded_h, int padded_w,
                   int ksize, int stride,
                   double* data_col) {
    int output_h = (padded_h - ksize) / stride + 1;
    int output_w = (padded_w - ksize) / stride + 1;
    int col_cols = output_h * output_w;
    size_t num_threads = std::min(static_cast<size_t>(channels), static_cast<size_t>(std::thread::hardware_concurrency()));
    parallel_for(0, channels, num_threads, [&](size_t c) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int kw = 0; kw < ksize; ++kw) {
                int row = c * ksize * ksize + kh * ksize + kw;
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        int im_index = c * (padded_h * padded_w) + ih * padded_w + iw;
                        int col_index = row * col_cols + oh * output_w + ow;
                        data_col[col_index] = data_im[im_index];
                    }
                }
            }
        }
    });
}

// Reverse of im2col: scatter columns back into image gradients
static void conv2d_col2im(const double* data_col, int channels, int padded_h, int padded_w,
                   int ksize, int stride,
                   double* data_im) {
    int output_h = (padded_h - ksize) / stride + 1;
    int output_w = (padded_w - ksize) / stride + 1;
    int col_cols = output_h * output_w;
    size_t num_threads = std::min(static_cast<size_t>(channels), static_cast<size_t>(std::thread::hardware_concurrency()));
    parallel_for(0, channels, num_threads, [&](size_t c) {
        for (int kh = 0; kh < ksize; ++kh) {
            for (int kw = 0; kw < ksize; ++kw) {
                int row = c * ksize * ksize + kh * ksize + kw;
                for (int oh = 0; oh < output_h; ++oh) {
                    for (int ow = 0; ow < output_w; ++ow) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        int col_index = row * col_cols + oh * output_w + ow;
                        int im_index = c * (padded_h * padded_w) + ih * padded_w + iw;
                        data_im[im_index] += data_col[col_index];
                    }
                }
            }
        }
    });
}

Conv2D::Conv2D(int num_filters, int kernel_size, int stride, int padding, Activation activation)
    : input_channels(-1), num_filters(num_filters), kernel_size(kernel_size), 
      stride(stride), padding(padding), activation_func(activation), weights_initialized(false) {
}

Conv2D::Conv2D(int input_channels, int num_filters, int kernel_size, int stride, int padding, Activation activation)
    : input_channels(input_channels), num_filters(num_filters), kernel_size(kernel_size),
      stride(stride), padding(padding), activation_func(activation), weights_initialized(false) {
    initialize_weights(input_channels, 0, 0);
}

Conv2D::~Conv2D() {
    // Smart pointers will automatically clean up
}

void Conv2D::initialize_weights(int input_channels, int height, int width) {
    this->input_channels = input_channels;
    
    // Initialize weights using Xavier/Glorot initialization
    // Shape: [num_filters, input_channels, kernel_size, kernel_size]
    std::vector<int> weight_shape = {num_filters, input_channels, kernel_size, kernel_size};
    weights = Tensor(weight_shape);
    
    // Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
    int fan_in = input_channels * kernel_size * kernel_size;
    int fan_out = num_filters * kernel_size * kernel_size;
    double scale = std::sqrt(2.0 / (fan_in + fan_out));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, scale);
    
    for (int i = 0; i < weights.size(); ++i) {
        weights.at(i) = dis(gen);
    }
    
    // Initialize biases to zero
    std::vector<int> bias_shape = {num_filters};
    biases = Tensor::zeros(bias_shape);
    
    weights_initialized = true;
}

std::vector<int> Conv2D::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("Input must be 3D: [channels, height, width]");
    }
    
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    int output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
    int output_width = (input_width + 2 * padding - kernel_size) / stride + 1;
    
    return {num_filters, output_height, output_width};
}

Tensor Conv2D::apply_padding(const Tensor& input) const {
    if (padding == 0) {
        return input;
    }
    
    std::vector<int> input_shape = input.shape();
    int channels = input_shape[0];
    int height = input_shape[1];
    int width = input_shape[2];
    
    std::vector<int> padded_shape = {channels, height + 2 * padding, width + 2 * padding};
    Tensor padded = Tensor::zeros(padded_shape);
    
    // Use fast indexing for efficient copying
    const double* input_data = input.data_ptr();
    double* padded_data = padded.data_ptr();
    
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int input_idx = input.fast_index_3d(c, h, w);
                int padded_idx = padded.fast_index_3d(c, h + padding, w + padding);
                padded_data[padded_idx] = input_data[input_idx];
            }
        }
    }
    
    return padded;
}

Tensor Conv2D::forward(const Tensor& input) {
    // pad input (reuse buffer)
    std::vector<int> in_shape = input.shape();
    if (padding > 0) {
        padded_input_buffer = apply_padding(input);
    } else {
        padded_input_buffer = input;
    }
    Tensor& padded_input = padded_input_buffer;
    std::vector<int> padded_shape = padded_input.shape();

    int input_channels = padded_shape[0];
    int input_height = padded_shape[1];
    int input_width = padded_shape[2];
    
    // Calculate output dimensions
    std::vector<int> output_shape = calculate_output_shape(input.shape());
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    
    // extract patches (reuse buffer)
    int col_rows = input_channels * kernel_size * kernel_size;
    int col_cols = output_height * output_width;
    if (col_buffer.shape() != std::vector<int>{col_rows, col_cols}) {
        col_buffer = Tensor(std::vector<int>{col_rows, col_cols});
    }
    conv2d_im2col(padded_input.data_ptr(), input_channels, padded_shape[1], padded_shape[2],
                  kernel_size, stride, col_buffer.data_ptr());

    Tensor weight_mat = weights;
    weight_mat.reshape(std::vector<int>{num_filters, col_rows});
    Tensor result = weight_mat.matmul(col_buffer);
    double* result_data = result.data_ptr();
    for (int f = 0; f < num_filters; ++f) {
        for (int i = 0; i < col_cols; ++i) {
            result_data[f * col_cols + i] += biases.at(f);
        }
    }
    result.reshape(std::vector<int>{num_filters, output_height, output_width});
    Tensor output = result;
    
    // Apply activation using centralized activation functions
    output = ActivationFunction::apply(output, activation_func);

    last_input = input;
    last_output = output;
    return output;
}

Tensor Conv2D::backward(const Tensor& output_gradient) {
    if (last_input.size() == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    std::vector<int> input_shape = last_input.shape();
    std::vector<int> output_shape = last_output.shape();
    
    // Apply activation derivative using centralized activation functions
    Tensor activation_grad = ActivationFunction::derivative(last_output, activation_func);
    
    // Element-wise multiply with output gradient
    for (int i = 0; i < output_gradient.size(); ++i) {
        activation_grad.at(i) *= output_gradient.at(i);
    }
    
    // Calculate gradients
    // Prepare shapes
    Tensor padded_input = apply_padding(last_input);
    std::vector<int> padded_shape = padded_input.shape();
    int padded_h = padded_shape[1], padded_w = padded_shape[2];
    int output_h = output_shape[1], output_w = output_shape[2];
    int col_rows = input_channels * kernel_size * kernel_size;
    int col_cols = output_h * output_w;
    // Extract input patches
    Tensor col = Tensor(std::vector<int>{col_rows, col_cols});
    conv2d_im2col(padded_input.data_ptr(), input_channels, padded_h, padded_w,
                 kernel_size, stride, col.data_ptr());
    // Reshape activation gradient to 2D: [filters, cols]
    Tensor act_grad_mat = activation_grad;
    act_grad_mat.reshape(std::vector<int>{num_filters, col_cols});
    // Weight gradients via GEMM: [filters x cols] * [cols x col_rows]
    Tensor weight_grad_2d = act_grad_mat.matmul(col.transpose());
    weight_grad_2d.reshape(weights.shape());
    Tensor weight_gradients = weight_grad_2d;
    // Bias gradients: sum over spatial dims
    Tensor bias_gradients = Tensor::zeros(biases.shape());
    for (int f = 0; f < num_filters; ++f) {
        double sum = 0;
        for (int i = 0; i < col_cols; ++i) sum += act_grad_mat.data_ptr()[f * col_cols + i];
        bias_gradients.at(f) = sum;
    }
    // Input gradients via GEMM: [col_rows x filters] * [filters x cols] => [col_rows x cols]
    Tensor weight_mat = weights;
    weight_mat.reshape(std::vector<int>{num_filters, col_rows});
    Tensor input_col_grad = weight_mat.transpose().matmul(act_grad_mat);
    // Scatter back to padded input gradients
    Tensor padded_input_gradients = Tensor::zeros(padded_shape);
    conv2d_col2im(input_col_grad.data_ptr(), input_channels, padded_h, padded_w,
                 kernel_size, stride, padded_input_gradients.data_ptr());
    // Remove padding
    Tensor input_gradients = Tensor::zeros(input_shape); // existing shape
    if (padding > 0) {
        double* inp = input_gradients.data_ptr();
        const double* padg = padded_input_gradients.data_ptr();
        for (int c = 0; c < input_channels; ++c)
            for (int h = 0; h < input_shape[1]; ++h)
                for (int w = 0; w < input_shape[2]; ++w) {
                    inp[input_gradients.fast_index_3d(c,h,w)] = padg[padded_input_gradients.fast_index_3d(c,h+padding,w+padding)];
                }
    } else input_gradients = padded_input_gradients;
    
    // Update weights and biases using optimizer
    if (weight_optimizer) {
        weight_optimizer->update(weights, weight_gradients);
        bias_optimizer->update(biases, bias_gradients);
    } else {
        // Default SGD with learning rate 0.01
        weights = weights - weight_gradients * 0.01;
        biases = biases - bias_gradients * 0.01;
    }
    
    return input_gradients;
}

void Conv2D::setOptimizer(std::shared_ptr<Optimizer> opt) {
    weight_optimizer = opt;
    bias_optimizer = opt;
}

void Conv2D::setOptimizerType(const std::string& type, double learning_rate) {
    if (type == "sgd") {
        weight_optimizer = std::make_shared<SGD>(learning_rate);
        bias_optimizer = std::make_shared<SGD>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer type: " + type);
    }
}

void Conv2D::set_weights(const Tensor& new_weights) {
    if (new_weights.shape() != weights.shape()) {
        throw std::invalid_argument("Weight shapes do not match");
    }
    weights = new_weights;
}

void Conv2D::set_biases(const Tensor& new_biases) {
    if (new_biases.shape() != biases.shape()) {
        throw std::invalid_argument("Bias shapes do not match");
    }
    biases = new_biases;
}

Tensor Conv2D::get_weights() const {
    return weights;
}

Tensor Conv2D::get_biases() const {
    return biases;
}

std::vector<Tensor> Conv2D::getParameters() const {
    return {weights, biases};
}

void Conv2D::setParameters(const std::vector<Tensor>& params) {
    if (params.size() != 2) {
        throw std::invalid_argument("Conv2D layer expects 2 parameters: weights and biases");
    }
    weights = params[0];
    biases = params[1];
    weights_initialized = true;
}

std::string Conv2D::getLayerType() const {
    return "Conv2D";
}

}
