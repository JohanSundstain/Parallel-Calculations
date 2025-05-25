cmake_minimum_required(VERSION 3.16)
project(jacobi_openacc LANGUAGES CXX)

# OpenACC backend selector
set(CMAKE_CXX_COMPILER nvc++)
set(ACC_BACKEND "gpu" CACHE STRING "OpenACC target backend (host, multicore, gpu)")
set_property(CACHE ACC_BACKEND PROPERTY STRINGS host multicore gpu)

# Use NVIDIA HPC compiler
if (NOT CMAKE_CXX_COMPILER MATCHES "nvc\\+\\+")
    message(FATAL_ERROR "You must use NVIDIA HPC compiler (nvc++) for OpenACC")
endif()

# Enable OpenACC
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minfo=accel -fast -lm")

# Select backend
if (ACC_BACKEND STREQUAL "gpu")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=gpu")
elseif (ACC_BACKEND STREQUAL "multicore")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=multicore")
elseif (ACC_BACKEND STREQUAL "host")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -acc=host")
else()
    message(FATAL_ERROR "Unknown ACC_BACKEND value: ${ACC_BACKEND}")
endif()


find_package(Boost REQUIRED COMPONENTS program_options)


add_executable(jacobi optimized.cpp)


target_include_directories(jacobi PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(jacobi PRIVATE ${Boost_LIBRARIES})
