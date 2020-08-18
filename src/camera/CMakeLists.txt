cmake_minimum_required(VERSION 3.5)
project(camera)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

find_package(OpenCV REQUIRED) 

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    MvCameraControl
)

target_link_directories(${PROJECT_NAME} PUBLIC
    /opt/MVS/lib/64/
    /opt/MVS/lib/aarch64/
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    /opt/MVS/include
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
