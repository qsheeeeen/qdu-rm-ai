if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

cmake_minimum_required(VERSION 3.5)

project(qdu_rm_ai
    DESCRIPTION "AI for Robomaster"
    VERSION 0.1.0
    LANGUAGES CXX
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

#---------------------------------------------------------------------------------------
# Platform
#---------------------------------------------------------------------------------------

message(STATUS "------------------------- Platform ------------------------------")

string(TIMESTAMP CONFIG_TIMESTAMP "" UTC)
message(STATUS "Timestamp: ${CONFIG_TIMESTAMP}")
message(STATUS "Host: ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_CROSSCOMPILING)
    message(STATUS "Target: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
message(STATUS "CMake: ${CMAKE_VERSION}")
message(STATUS "CMake generator: ${CMAKE_GENERATOR}")
message(STATUS "CMake build tool: ${CMAKE_BUILD_TOOL}")
if(MSVC)
    message(STATUS "MSVC: ${MSVC_VERSION}")
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
    message(STATUS "Xcode: ${XCODE_VERSION}")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
    message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}")
endif()

#---------------------------------------------------------------------------------------
# Compiler Options
#---------------------------------------------------------------------------------------

message(STATUS "-------------------------- Compiler ------------------------------")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags (Release): ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug): ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
if(WIN32)
    message(STATUS "Linker flags (Release): ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
    message(STATUS "Linker flags (Debug): ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
    message(STATUS "Linker flags (Release): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
    message(STATUS "Linker flags (Debug): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()

#---------------------------------------------------------------------------------------
# Building Options
#---------------------------------------------------------------------------------------

message(STATUS "----------------------- Building Options ------------------------")

find_package(BehaviorTreeV3 REQUIRED)
find_package(CUDA)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

message(STATUS "BehaviorTreeV3 version: ${BehaviorTreeV3_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "spdlog version: ${spdlog_VERSION}")
message(STATUS "TBB version: ${TBB_VERSION}")

option(BUILD_NN "Build nn" OFF)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Realase")
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
endif ()

#---------------------------------------------------------------------------------------
# Source
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)

#---------------------------------------------------------------------------------------
# Test
#---------------------------------------------------------------------------------------

message(STATUS "---------------------------- Test -------------------------------")

enable_testing()
find_package(GTest REQUIRED)
message(STATUS "Google test version: ${GTest_VERSION}")
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)

#---------------------------------------------------------------------------------------
# Install
#---------------------------------------------------------------------------------------
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "------------------------ Install --------------------------------")
    message(STATUS "Generating install")
    message(STATUS "Install to: ${CMAKE_INSTALL_PREFIX}")

    install(
        TARGETS auto-aim
        EXPORT auto-aim
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        TARGETS sentry
        EXPORT sentry
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

endif()

message(STATUS "-----------------------------------------------------------------")
