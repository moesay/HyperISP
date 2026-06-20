cmake_minimum_required(VERSION 3.20)
project(hyperISP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMake can't set cuda standards to 23 eventho for nvcc 13.x, c++23 is supported
# to bypass this, i will not ask cmake for cuda 23 and pass the flag manually

# set(CMAKE_CUDA_STANDARD 23)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

add_executable(hyperISP
    src/main.cpp
    src/config.cpp
    src/blocks/dpc.cu
    src/blocks/blc.cu
    src/blocks/aaf.cu
    src/blocks/awb.cu
    src/blocks/cnf.cu
    src/blocks/cfa.cu
)

target_compile_options(hyperISP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++23>
)

target_include_directories(hyperISP PRIVATE include)
target_link_libraries(hyperISP PRIVATE tomlplusplus::tomlplusplus)

set_target_properties(hyperISP PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
