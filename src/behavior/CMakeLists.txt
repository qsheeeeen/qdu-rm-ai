cmake_minimum_required(VERSION 3.5)
project(behavior)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

find_package(BehaviorTreeV3)

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${PROJECT_NAME} OBJECT EXCLUDE_FROM_ALL ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME} BT::behaviortree_cpp_v3)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
