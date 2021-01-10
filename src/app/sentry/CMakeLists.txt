cmake_minimum_required(VERSION 3.5)
project(sentry)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
    behavior
    device
    vision
    spdlog::spdlog
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_dependencies(${PROJECT_NAME} 
    behavior
    device
    vision
)
