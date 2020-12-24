cmake_minimum_required(VERSION 3.5)

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)

if(CUDA_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cuda)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision)

