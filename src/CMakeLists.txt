cmake_minimum_required(VERSION 3.5)

project(qdu_rm_ai
  DESCRIPTION "AI for Robomaster"
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
    message(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there.")
endif()

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

# Allow switching of Applications
set(APPLICATION_LIST auto_aim full_auto)
set(APPLICATION auto_aim CACHE STRING "Default Application is auto_aim")
set_property(CACHE APPLICATION PROPERTY STRINGS ${APPLICATION_LIST})


# Allow switching of camera API
set(CAMERA_API_LIST OPENCV MVS)
set(CAMERA_API auto_aim CACHE STRING "Default camera API is OPENCV")
set_property(CACHE CAMERA_API PROPERTY STRINGS ${APPLICATION_LIST})

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/camera)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/robot)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision)

#---------------------------------------------------------------------------------------
# Applications
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/${APPLICATION})
