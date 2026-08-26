cmake_minimum_required(VERSION 3.18)
project(aoti_cpp_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(aoti_cpp_inference aoti_cpp_inference.cpp)
target_link_libraries(aoti_cpp_inference "${TORCH_LIBRARIES}")
