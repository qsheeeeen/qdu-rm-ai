cmake_minimum_required(VERSION 3.5)

project(qdu_rm_ai_node)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)