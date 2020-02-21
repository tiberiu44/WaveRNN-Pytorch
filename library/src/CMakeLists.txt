cmake_minimum_required (VERSION 3.0)

project(WaveRNN)

set(CMAKE_CXX_STANDARD 11)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -ffast-math -march=native -fopenmp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -ffast-math -march=native -fopenmp")

add_executable (vocoder vocoder.cpp)
add_library(wavernn wavernn.cpp net_impl.cpp)
target_link_libraries(vocoder wavernn Eigen3::Eigen)

add_subdirectory(pybind11)
pybind11_add_module(WaveRNNVocoder WaveRNNVocoder.cpp wavernn.cpp net_impl.cpp)

