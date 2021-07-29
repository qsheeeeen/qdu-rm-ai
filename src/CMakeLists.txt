cmake_minimum_required(VERSION 3.5)

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/component)

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
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/buff)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/dart)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/radar)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/sentry)
