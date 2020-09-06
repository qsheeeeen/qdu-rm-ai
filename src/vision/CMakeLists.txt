cmake_minimum_required(VERSION 3.5)
project(vision)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})


add_dependencies(${PROJECT_NAME} 
    device
)

target_link_libraries(${PROJECT_NAME} PUBLIC 
    ${OpenCV_LIBS}
    spdlog::spdlog
    nvinfer
    nvonnxparser
    cudart
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:device,INTERFACE_INCLUDE_DIRECTORIES>
)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    target_include_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/targets/aarch64-linux/include)
    target_link_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/targets/aarch64-linux/lib)
else()
    target_include_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/include)
    target_link_directories(${PROJECT_NAME} PUBLIC /usr/local/cuda/lib64)
endif()

