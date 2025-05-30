cmake_minimum_required(VERSION 3.8)

project(jacobi_project)

project(MyCUDAProject LANGUAGES CXX CUDA)

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(main optimized.cu)

target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O2>)

target_link_libraries(main PRIVATE 
    Boost::program_options
)

set_target_properties(main PROPERTIES
    CUDA_STANDARD 14
    CXX_STANDARD 14
)