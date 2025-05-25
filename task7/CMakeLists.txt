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

include_directories(/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/include)
link_directories(/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/math_libs/lib64)

add_executable(jacobi optimized.cpp)

target_link_libraries(jacobi PRIVATE ${Boost_LIBRARIES} cublas)
