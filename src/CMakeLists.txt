cmake_minimum_required(VERSION 3.5)

project(qdu_rm_ai
  DESCRIPTION "AI for Robomaster"
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("-- System arch:  ${CMAKE_SYSTEM_PROCESSOR}")

find_package(BehaviorTreeV3 REQUIRED) 
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

message("-- BehaviorTreeV3 version: ${BehaviorTreeV3_VERSION}")
message("-- CUDA version: ${CUDA_VERSION}")
message("-- OpenCV version: ${OpenCV_VERSION}")
message("-- spdlog version: ${spdlog_VERSION}")
message("-- Threads version: ${Threads_VERSION}")

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
