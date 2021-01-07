cmake_minimum_required(VERSION 3.5)
project(vision)

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME} PUBLIC 
    ${OpenCV_LIBS}
    spdlog::spdlog
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:device,INTERFACE_INCLUDE_DIRECTORIES>
)
