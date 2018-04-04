cmake_minimum_required(VERSION 3.10)
project(learning_CNN)

set(CMAKE_CXX_STANDARD 11)

add_executable(learning_CNN main.cpp Net.cpp Net.h Neuron.cpp Neuron.h TrainingData.cpp TrainingData.h)