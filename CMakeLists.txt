cmake_minimum_required(VERSION 3.9)

project(mandelbrot LANGUAGES C)

SET(AVX_FLAG "-mavx")
add_definitions(${AVX_FLAG})

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

find_package(OpenMP)

SET(SOURCES main.c
            include/mandelbrot.h
            include/mandelbrot_avx.h
            include/mandelbrot_sse.h
            src/mandelbrot.c
            src/mandelbrot_avx.c
            src/mandelbrot_sse.c
)

add_executable(mandelbrot ${SOURCES})
target_link_libraries(mandelbrot ${SDL2_LIBRARIES})
target_link_libraries(mandelbrot OpenMP::OpenMP_C)
target_include_directories(mandelbrot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)