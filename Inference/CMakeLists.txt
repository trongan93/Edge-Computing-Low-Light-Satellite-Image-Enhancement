cmake_minimum_required(VERSION 3.27)
project(Inference)

set(CMAKE_CXX_STANDARD 17)

find_package( OpenCV REQUIRED ) # OpenCV config
include_directories( ${OpenCV_INCLUDE_DIRS} ) # OpenCV config

add_executable(Inference main.cpp test_1.cpp test_1.h q_Train_Model_Weight.h memAlloc.h zerodce_algorithm.cpp zerodce_algorithm.h
        ll_enhancement_algorithm.cpp
        ll_enhancement_algorithm.h)
target_link_libraries( QuantizedZeroDCE_CPP ${OpenCV_LIBS} ) # OpenCV config
