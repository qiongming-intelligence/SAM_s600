cmake_minimum_required(VERSION 3.22)

project(sam_s600 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SAM_S600_BUILD_APPS "Build SAM_s600 command line applications" ON)
option(SAM_S600_BUILD_TESTS "Build SAM_s600 tests" OFF)
option(SAM_S600_ENABLE_HOBOT "Enable D-Robotics/Hobot integration hooks" ON)

add_subdirectory(src/csrc)

if(SAM_S600_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/csrc)
endif()
