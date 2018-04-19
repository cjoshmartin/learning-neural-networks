cmake_minimum_required(VERSION 3.10)
project(learning_neural_networks)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(learning_neural_networks
        main.cpp
        Net.cpp
        Net.h
        Neuron.cpp
        Neuron.h
        string_formater.cpp
        string_formater.h
        TrainingData.cpp
        TrainingData.h)
