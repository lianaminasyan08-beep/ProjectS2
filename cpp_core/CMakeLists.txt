cmake_minimum_required(VERSION 3.10)
project(core)

set(OpenCV_DIR "C:/opencv/opencv/build")

find_package(OpenCV REQUIRED)

add_executable(core main.cpp)

target_link_libraries(core ${OpenCV_LIBS})