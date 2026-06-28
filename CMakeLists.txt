cmake_minimum_required(VERSION 3.20)
project(hyperISP LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

add_library(hyperisp STATIC
    src/config.cpp
    src/pipeline.cpp
    src/blocks/dpc.cu
    src/blocks/blc.cu
    src/blocks/aaf.cu
    src/blocks/awb.cu
    src/blocks/cnf.cu
    src/blocks/cfa.cu
    src/blocks/ccm.cu
    src/blocks/gac.cu
    src/blocks/csc.cu
    src/blocks/nlm.cu
    src/blocks/bnf.cu
)

target_compile_options(hyperisp PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-g -G>
)

target_include_directories(hyperisp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hyperisp PUBLIC tomlplusplus::tomlplusplus)

set_target_properties(hyperisp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
