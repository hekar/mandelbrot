cmake_minimum_required(VERSION 3.9)

project(mandelbrot LANGUAGES C)

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

find_package(OpenMP)

add_executable(mandelbrot main.c)
target_link_libraries(mandelbrot ${SDL2_LIBRARIES})
target_link_libraries(mandelbrot OpenMP::OpenMP_C)