cmake_minimum_required(VERSION 3.16)
project(jacobi_openacc LANGUAGES CXX)

# OpenACC backend selector
set(CMAKE_CXX_COMPILER nvc++)
set(ACC_BACKEND "gpu" CACHE STRING "OpenACC target backend")

# Use NVIDIA HPC compiler
if (NOT CMAKE_CXX_COMPILER MATCHES "nvc\\+\\+")
    message(FATAL_ERROR "You must use NVIDIA HPC compiler (nvc++) for OpenACC")
endif()

# Enable OpenACC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minfo=accel -fast -acc=gpu")


find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(jacobi optimized.cpp)

find_package(CUDAToolkit REQUIRED)
target_compile_definitions(jacobi PRIVATE CUBLAS)
target_include_directories(jacobi PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(jacobi PRIVATE CUDA::cublas)

target_link_libraries(jacobi PRIVATE ${Boost_LIBRARIES})
