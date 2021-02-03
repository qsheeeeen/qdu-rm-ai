cmake_minimum_required(VERSION 3.5)

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)

if(BUILD_NN)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/nn)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision)

#---------------------------------------------------------------------------------------
# Demo
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/demo/armor)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/demo/camera)

#---------------------------------------------------------------------------------------
# Applications
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/auto_aim)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/sentry)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/radar)

