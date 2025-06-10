cmake_minimum_required(VERSION 3.10)

project(JNet)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

file(GLOB_RECURSE SOURCES 
    src/core/*.cpp 
    src/layers/*.cpp 
    src/optimizers/*.cpp
)

# Add pthread support for multithreading
find_package(Threads REQUIRED)

add_library(jnet ${SOURCES})
target_link_libraries(jnet Threads::Threads)

add_executable(cnn_example examples/cnn_example.cpp)
target_link_libraries(cnn_example jnet)

add_executable(training_progress_demo examples/training_progress_demo.cpp)
target_link_libraries(training_progress_demo jnet)

add_executable(optimizer_comparison examples/optimizer_comparison.cpp)
target_link_libraries(optimizer_comparison jnet)

add_executable(performance_benchmark examples/performance_benchmark.cpp)
target_link_libraries(performance_benchmark jnet)

add_executable(mnist_cnn examples/mnist_cnn.cpp)
target_link_libraries(mnist_cnn jnet)

add_executable(sine_regression examples/sine_regression.cpp)
target_link_libraries(sine_regression jnet)