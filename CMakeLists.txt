cmake_minimum_required(VERSION 3.10)
project(optim_proj)

set(CMAKE_BUILD_TYPE "DEBUG")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

add_executable(hello_world hello_world.cpp)
target_link_libraries(hello_world ${CERES_LIBRARIES})


add_executable(powell powell.cpp)
target_link_libraries(powell ${CERES_LIBRARIES} gflags)