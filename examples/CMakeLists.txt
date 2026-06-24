cmake_minimum_required(VERSION 3.20)
project(hyperisp_examples LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(FetchContent)

FetchContent_Declare(
    cmdparser
    GIT_REPOSITORY https://github.com/FlorianRappl/CmdParser
    GIT_TAG v1.1.0
)

FetchContent_MakeAvailable(cmdparser)

# Because its just a demo, im pulling the static lib from the parent project.
# If you are a real consumer, you would have to FetchContent/add_subdirectory the repo as a dependency
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_BINARY_DIR}/hyperisp)

add_executable(hyperisp_example main.cpp)
target_link_libraries(hyperisp_example PRIVATE hyperisp)

# A glue fix to an issue that is not mine, CMDParser's CMakeLists only builds its test dir.
# it doesn't export a target for the header itself thats why im adding the header manually
target_include_directories(hyperisp_example PRIVATE ${cmdparser_SOURCE_DIR})
