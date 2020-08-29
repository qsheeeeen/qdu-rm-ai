cmake_minimum_required(VERSION 3.5)
project(vision)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

find_package(OpenCV REQUIRED) 
find_package(spdlog REQUIRED)

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
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:device,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
