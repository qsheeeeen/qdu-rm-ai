cmake_minimum_required(VERSION 3.5)
project(auto_aim)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

add_executable(${PROJECT_NAME} main.cpp)

find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} PUBLIC
    behavior
    vision
    camera
    robot
    Threads::Threads
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(${PROJECT_NAME} PUBLIC -Wall -Wextra)
