cmake_minimum_required(VERSION 3.2)

project(Lab1)

set(source main.cpp)
set(target_name main)
option(double OFF)

add_executable(${target_name} ${source})

if (double)
	target_compile_options(${target_name} PRIVATE -D DOUBLE)
endif()