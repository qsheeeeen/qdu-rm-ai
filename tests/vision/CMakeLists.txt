cmake_minimum_required(VERSION 3.5)
project(test_vision)

file(GLOB ${PROJECT_NAME}_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_SOURCES})

target_link_libraries(${PROJECT_NAME} PRIVATE 
    vision
    gtest
    gtest_main
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<TARGET_PROPERTY:vision,INTERFACE_INCLUDE_DIRECTORIES>
)


add_test(test_vision test_vision)
