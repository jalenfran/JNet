cmake_minimum_required(VERSION 3.10)

project(JNet)

set(CMAKE_CXX_STANDARD 14)

include_directories(include)

file(GLOB_RECURSE SOURCES 
    src/core/*.cpp 
    src/layers/*.cpp 
    src/optimizers/*.cpp
)

add_library(jnet ${SOURCES})

add_executable(example examples/basic_example.cpp)
target_link_libraries(example jnet)

add_executable(simple_test examples/simple_test.cpp)
target_link_libraries(simple_test jnet)

add_executable(epoch_example examples/epoch_training_example.cpp)
target_link_libraries(epoch_example jnet)

add_executable(optimizer_comparison examples/optimizer_comparison.cpp)
target_link_libraries(optimizer_comparison jnet)

add_executable(cnn_example examples/cnn_example.cpp)
target_link_libraries(cnn_example jnet)

enable_testing()