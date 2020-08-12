cmake_minimum_required(VERSION 3.5)
project(full_auto)

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
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:behavior,INTERFACE_INCLUDE_DIRECTORIES>>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:vision,INTERFACE_INCLUDE_DIRECTORIES>>
)
