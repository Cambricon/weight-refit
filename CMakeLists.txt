cmake_minimum_required(VERSION 2.8)

project(resnet50_pytorch)

message("-- ABI_MODE=$ENV{ABI_MODE}")
message("-- NEUWARE_HOME=$ENV{NEUWARE_HOME}")
message("-- TOOLCHAIN_DIR=$ENV{TOOLCHAIN_DIR}")

set(CMAKE_SYSTEM_NAME Linux)
set(DEP_DIR ${CMAKE_SOURCE_DIR}/../3rdparty)
set(ABI_MODE $ENV{ABI_MODE})
set(CMAKE_CXX_FLAGS "-Werror -g -fPIC -std=c++11 -pthread")
SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)

include_directories(include $ENV{NEUWARE_HOME}/include)
link_directories($ENV{NEUWARE_HOME}/lib64)

add_executable(infer src/infer.cpp src/pre_process.cpp)
target_link_libraries(infer magicmind_runtime cnrt opencv_core opencv_imgcodecs opencv_imgproc)

add_executable(gen_model src/gen_model.cpp src/calibrate.cpp src/pre_process.cpp)
target_link_libraries(gen_model magicmind magicmind_parser opencv_core opencv_imgcodecs opencv_imgproc)
