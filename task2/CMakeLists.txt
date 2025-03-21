cmake_minimum_required(VERSION 3.2)

project(Lab2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

if(part STREQUAL "one")
    message("TASK ONE")

    add_executable(task1 part1/main.cpp)
elseif(part STREQUAL "two")
    message("TASK TWO")

    add_executable(task2 part2/main.cpp)
elseif(part STREQUAL "three")
    message("TASK THREE")

    add_executable(task3_m1 part2/main_m1.cpp)
	add_executable(task3_m2 part3/main_m2.cpp)
else()
    message(FATAL_ERROR "Неверное значение part: ${part}")
endif()
