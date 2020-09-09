cmake_minimum_required(VERSION 3.5)

project(qdu_rm_ai
  DESCRIPTION "AI for Robomaster"
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "System arch: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMake:  ${CMAKE_VERSION}")
message(STATUS "CMake generator: ${CMAKE_GENERATOR}")
message(STATUS "CMake build tool: ${CMAKE_BUILD_TOOL}")

find_package(BehaviorTreeV3 REQUIRED) 
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

message(STATUS "BehaviorTreeV3: ${BehaviorTreeV3_VERSION}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "spdlog: ${spdlog_VERSION}")
message(STATUS "Threads: ${Threads_VERSION}")

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG__)  
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)

elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(RELEASE__)  
    add_compile_definitions(SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)

endif()

add_compile_options(-Wall -Wextra -O2)

if(DEFINED CMAKE_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
endif()
message(STATUS "C++ flags (Release): ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug): ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision)

#---------------------------------------------------------------------------------------
# Applications
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/auto_aim)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/full_auto)
