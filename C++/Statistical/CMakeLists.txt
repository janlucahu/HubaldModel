cmake_minimum_required(VERSION 3.24)
project(Statistical)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_executable(Statistical main.cpp statistical_hubald_model.cpp statistical_hubald_model.h)
