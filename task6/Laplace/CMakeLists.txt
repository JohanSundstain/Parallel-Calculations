cmake_minimum_required(VERSION 3.22)

project(jacobi VERSION 1.0 LANGUAGES C)

set(CMAKE_C_COMPILER pgcc)
message("-!-Compile C: " ${CMAKE_C_COMPILER})

find_package(CUDAToolkit REQUIRED)

list(APPEND option_compile -acc=host -Minfo=all)
list(APPEND option_link -acc=host)

# Добавляем исходники
set(SRC
    src/laplace2d.c
    main_jacobi.c
)

# Создаём папку для бинарников
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${OUTPUT_DIR})

add_executable(${PROJECT_NAME} ${SRC})
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
# openACC
target_compile_options(${PROJECT_NAME} PRIVATE ${option_compile}) 
target_link_options(${PROJECT_NAME} PRIVATE ${option_link})
# cuda and nvtx
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::nvToolsExt)

