cmake_minimum_required(VERSION 3.5)
project(armor_demo)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE
    vision
    device
    spdlog::spdlog
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:vision,INTERFACE_INCLUDE_DIRECTORIES>
)
