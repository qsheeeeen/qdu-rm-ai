cmake_minimum_required(VERSION 3.5)
project(auto_aim)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# Target
#---------------------------------------------------------------------------------------

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

add_executable(${PROJECT_NAME}
    ${PROJECT_NAME}_SOURCES
    $<TARGET_OBJECTS:behavior>
    $<TARGET_OBJECTS:vision>
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:behavior,INTERFACE_INCLUDE_DIRECTORIES>>
    $<BUILD_INTERFACE:$<TARGET_PROPERTY:vision,INTERFACE_INCLUDE_DIRECTORIES>>
)
