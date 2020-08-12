cmake_minimum_required(VERSION 3.5)
project(behavior)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

find_package(BehaviorTreeV3 REQUIRED) 

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME} behaviortree_cpp_v3)

add_dependencies(${PROJECT_NAME} camera)
add_dependencies(${PROJECT_NAME} com)
add_dependencies(${PROJECT_NAME} vision)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
