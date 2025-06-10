#include "pooling.h"
#include "../core/threadpool.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <thread>
#include <limits>

namespace JNet {

// MaxPool2D Implementation
MaxPool2D::MaxPool2D(int pool_size, int stride)
    : pool_size(pool_size), stride(stride == -1 ? pool_size : stride) {
}

MaxPool2D::~MaxPool2D() {
    // Smart pointers will automatically clean up
}

std::vector<int> MaxPool2D::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("MaxPool2D input must be 3D: [channels, height, width]");
    }
    
    int channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    
    return {channels, output_height, output_width};
}

Tensor MaxPool2D::forward(const Tensor& input) {
    std::vector<int> input_shape = input.shape();
    
    if (input_shape.size() != 3) {
        throw std::invalid_argument("MaxPool2D input must be 3D: [channels, height, width]");
    }
    
    // Store input for backpropagation
    last_input = input;
    
    int channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    // Calculate output dimensions
    std::vector<int> output_shape = calculate_output_shape(input_shape);
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    
    // Create output tensor and mask for backpropagation
    Tensor output = Tensor::zeros(output_shape);
    mask = Tensor::zeros(input_shape);
    
    // Perform max pooling with multithreading for larger inputs
    size_t total_operations = static_cast<size_t>(channels) * output_height * output_width;
    size_t thread_threshold = 25; // Threshold for using multithreading
    
    if (total_operations > thread_threshold) {
        // Parallelize over channels
        size_t num_threads = std::min(static_cast<size_t>(channels), 
                                     static_cast<size_t>(std::thread::hardware_concurrency()));
        
        // Get direct data pointers for fast access
        const double* input_data = input.data_ptr();
        double* output_data = output.data_ptr();
        double* mask_data = mask.data_ptr();
        
        parallel_for(0, channels, num_threads, [&](size_t c) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_h = -1, max_w = -1;
                    
                    // Find maximum in pooling window
                    for (int kh = 0; kh < pool_size; ++kh) {
                        for (int kw = 0; kw < pool_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (ih < input_height && iw < input_width) {
                                int input_idx = input.fast_index_3d(static_cast<int>(c), ih, iw);
                                double val = input_data[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_h = ih;
                                    max_w = iw;
                                }
                            }
                        }
                    }
                    
                    // Store result and mask using fast indexing
                    int output_idx = output.fast_index_3d(static_cast<int>(c), oh, ow);
                    output_data[output_idx] = max_val;
                    if (max_h >= 0 && max_w >= 0) {
                        int mask_idx = mask.fast_index_3d(static_cast<int>(c), max_h, max_w);
                        mask_data[mask_idx] = 1.0;
                    }
                }
            }
        });
    } else {
        // Use single-threaded approach for smaller pooling operations with fast access
        const double* input_data = input.data_ptr();
        double* output_data = output.data_ptr();
        double* mask_data = mask.data_ptr();
        
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    double max_val = -std::numeric_limits<double>::infinity();
                    int max_h = -1, max_w = -1;
                    
                    // Find maximum in pooling window
                    for (int kh = 0; kh < pool_size; ++kh) {
                        for (int kw = 0; kw < pool_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (ih < input_height && iw < input_width) {
                                int input_idx = input.fast_index_3d(c, ih, iw);
                                double val = input_data[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_h = ih;
                                    max_w = iw;
                                }
                            }
                        }
                    }
                    
                    // Store result and mask using fast indexing
                    int output_idx = output.fast_index_3d(c, oh, ow);
                    output_data[output_idx] = max_val;
                    if (max_h >= 0 && max_w >= 0) {
                        int mask_idx = mask.fast_index_3d(c, max_h, max_w);
                        mask_data[mask_idx] = 1.0;
                    }
                }
            }
        }
    }
    
    return output;
}

Tensor MaxPool2D::backward(const Tensor& output_gradient) {
    if (last_input.size() == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    // For max pooling, gradient flows only through the maximum elements
    Tensor input_gradient = Tensor::zeros(last_input.shape());
    std::vector<int> output_shape = output_gradient.shape();
    
    int channels = output_shape[0];
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    
    // Distribute gradients based on the mask using fast indexing
    const double* output_grad_data = output_gradient.data_ptr();
    const double* mask_data = mask.data_ptr();
    double* input_grad_data = input_gradient.data_ptr();
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                int output_idx = output_gradient.fast_index_3d(c, oh, ow);
                double grad = output_grad_data[output_idx];
                
                // Find the position that contributed to this output
                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        if (ih < last_input.shape()[1] && iw < last_input.shape()[2]) {
                            int mask_idx = mask.fast_index_3d(c, ih, iw);
                            if (mask_data[mask_idx] > 0.5) { // This was the max element
                                int input_idx = input_gradient.fast_index_3d(c, ih, iw);
                                input_grad_data[input_idx] += grad;
                            }
                        }
                    }
                }
            }
        }
    }
    
    return input_gradient;
}

void MaxPool2D::setOptimizer(std::shared_ptr<Optimizer> opt) {
    // Pooling layers don't have parameters to optimize
}

void MaxPool2D::setOptimizerType(const std::string& type, double learning_rate) {
    // Pooling layers don't have parameters to optimize
}

// AvgPool2D Implementation
AvgPool2D::AvgPool2D(int pool_size, int stride)
    : pool_size(pool_size), stride(stride == -1 ? pool_size : stride) {
}

AvgPool2D::~AvgPool2D() {
    // Smart pointers will automatically clean up
}

std::vector<int> AvgPool2D::calculate_output_shape(const std::vector<int>& input_shape) const {
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AvgPool2D input must be 3D: [channels, height, width]");
    }
    
    int channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    int output_height = (input_height - pool_size) / stride + 1;
    int output_width = (input_width - pool_size) / stride + 1;
    
    return {channels, output_height, output_width};
}

Tensor AvgPool2D::forward(const Tensor& input) {
    std::vector<int> input_shape = input.shape();
    
    if (input_shape.size() != 3) {
        throw std::invalid_argument("AvgPool2D input must be 3D: [channels, height, width]");
    }
    
    // Store input for backpropagation
    last_input = input;
    
    int channels = input_shape[0];
    int input_height = input_shape[1];
    int input_width = input_shape[2];
    
    // Calculate output dimensions
    std::vector<int> output_shape = calculate_output_shape(input_shape);
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    
    // Create output tensor
    Tensor output = Tensor::zeros(output_shape);
    
    // Perform average pooling with multithreading for larger inputs
    size_t total_operations = static_cast<size_t>(channels) * output_height * output_width;
    size_t thread_threshold = 25; // Threshold for using multithreading
    
    if (total_operations > thread_threshold) {
        // Parallelize over channels
        size_t num_threads = std::min(static_cast<size_t>(channels), 
                                     static_cast<size_t>(std::thread::hardware_concurrency()));
        
        // Get direct data pointers for fast access
        const double* input_data = input.data_ptr();
        double* output_data = output.data_ptr();
        
        parallel_for(0, channels, num_threads, [&](size_t c) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    double sum = 0.0;
                    int count = 0;
                    
                    // Sum values in pooling window
                    for (int kh = 0; kh < pool_size; ++kh) {
                        for (int kw = 0; kw < pool_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (ih < input_height && iw < input_width) {
                                int input_idx = input.fast_index_3d(static_cast<int>(c), ih, iw);
                                sum += input_data[input_idx];
                                count++;
                            }
                        }
                    }
                    
                    // Store average using fast indexing
                    int output_idx = output.fast_index_3d(static_cast<int>(c), oh, ow);
                    output_data[output_idx] = sum / count;
                }
            }
        });
    } else {
        // Use single-threaded approach for smaller pooling operations with fast access
        const double* input_data = input.data_ptr();
        double* output_data = output.data_ptr();
        
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < output_height; ++oh) {
                for (int ow = 0; ow < output_width; ++ow) {
                    double sum = 0.0;
                    int count = 0;
                    
                    // Sum values in pooling window
                    for (int kh = 0; kh < pool_size; ++kh) {
                        for (int kw = 0; kw < pool_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            
                            if (ih < input_height && iw < input_width) {
                                int input_idx = input.fast_index_3d(c, ih, iw);
                                sum += input_data[input_idx];
                                count++;
                            }
                        }
                    }
                    
                    // Store average using fast indexing
                    int output_idx = output.fast_index_3d(c, oh, ow);
                    output_data[output_idx] = sum / count;
                }
            }
        }
    }
    
    return output;
}

Tensor AvgPool2D::backward(const Tensor& output_gradient) {
    if (last_input.size() == 0) {
        throw std::runtime_error("Must call forward() before backward()");
    }
    
    // For average pooling, gradient is distributed evenly across the pooling window
    Tensor input_gradient = Tensor::zeros(last_input.shape());
    std::vector<int> output_shape = output_gradient.shape();
    
    int channels = output_shape[0];
    int output_height = output_shape[1];
    int output_width = output_shape[2];
    int input_height = last_input.shape()[1];
    int input_width = last_input.shape()[2];
    
    // Distribute gradients evenly using fast indexing
    const double* output_grad_data = output_gradient.data_ptr();
    double* input_grad_data = input_gradient.data_ptr();
    
    for (int c = 0; c < channels; ++c) {
        for (int oh = 0; oh < output_height; ++oh) {
            for (int ow = 0; ow < output_width; ++ow) {
                int output_idx = output_gradient.fast_index_3d(c, oh, ow);
                double grad = output_grad_data[output_idx];
                int count = 0;
                
                // Count valid positions in pooling window
                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        if (ih < input_height && iw < input_width) {
                            count++;
                        }
                    }
                }
                
                // Distribute gradient evenly
                double avg_grad = grad / count;
                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        
                        if (ih < input_height && iw < input_width) {
                            int input_idx = input_gradient.fast_index_3d(c, ih, iw);
                            input_grad_data[input_idx] += avg_grad;
                        }
                    }
                }
            }
        }
    }
    
    return input_gradient;
}

void AvgPool2D::setOptimizer(std::shared_ptr<Optimizer> opt) {
    // Pooling layers don't have parameters to optimize
}

void AvgPool2D::setOptimizerType(const std::string& type, double learning_rate) {
    // Pooling layers don't have parameters to optimize
}

}
