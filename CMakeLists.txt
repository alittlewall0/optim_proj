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

add_executable(curve_fitting curve_fitting.cpp)
target_link_libraries(curve_fitting ${CERES_LIBRARIES} gflags)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS} "/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

add_executable(g2o_curve_fit g2o_curve_fit.cpp)
target_link_libraries(g2o_curve_fit
                    ${OpenCV_LIBS}
                    g2o_core g2o_stuff)