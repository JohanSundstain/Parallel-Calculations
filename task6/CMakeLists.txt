cmake_minimum_required(VERSION 3.20)
project(Laplace2D LANGUAGES C CXX)

# Настройка компиляторов
set(CMAKE_C_COMPILER pgcc)
set(CMAKE_CXX_COMPILER pgc++)

# Флаги компиляции
set(CMAKE_C_FLAGS "-fast -ta=tesla")
set(CMAKE_CXX_FLAGS "-fast -ta=tesla")
set(CMAKE_Fortran_FLAGS "-fast -ta=tesla")
set(CMAKE_EXE_LINKER_FLAGS "-fast -ta=tesla")

# Путь к NVIDIA HPC SDK
set(NVTXLIB_DIR "/opt/nvidia/hpc_sdk/Linux_x86_64/20.9/cuda/11.0/include")

# Исходные файлы
set(SOURCES laplace2d.c jacobi.c)  # Или .cpp/.F90 в зависимости от ваших исходников

# Создание исполняемого файла
add_executable(exe ${SOURCES})
target_include_directories(exe PRIVATE ${NVTXLIB_DIR})

# Добавление цели для запуска
add_custom_target(run ALL
    DEPENDS exe
    COMMAND ./exe
)

# Цель для очистки
add_custom_target(clean_cmake
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}
)