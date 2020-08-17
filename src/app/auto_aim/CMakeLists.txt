cmake_minimum_required(VERSION 3.5)
project(auto_aim)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC
    behavior
    vision
    camera
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra)
