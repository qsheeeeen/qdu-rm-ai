cmake_minimum_required(VERSION 3.5)

project(qdu_rm_ai
  DESCRIPTION "AI for Robomaster"
  VERSION 0.1.0
  LANGUAGES CXX
)


#---------------------------------------------------------------------------------------
# Platform
#---------------------------------------------------------------------------------------

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS "-----------------------------------------------------------------")

string(TIMESTAMP CONFIG_TIMESTAMP "" UTC)
message(STATUS "Timestamp: ${CONFIG_TIMESTAMP}")
message(STATUS "Host: ${CMAKE_HOST_SYSTEM_NAME} ${CMAKE_HOST_SYSTEM_VERSION} ${CMAKE_HOST_SYSTEM_PROCESSOR}")
if(CMAKE_CROSSCOMPILING)
  message(STATUS "Target: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION} ${CMAKE_SYSTEM_PROCESSOR}")
endif()
message(STATUS "CMake: ${CMAKE_VERSION}")
message(STATUS "CMake generator: ${CMAKE_GENERATOR}")
message(STATUS "CMake build tool: ${CMAKE_BUILD_TOOL}")
if(MSVC)
  message(STATUS "MSVC: ${MSVC_VERSION}")
endif()
if(CMAKE_GENERATOR MATCHES Xcode)
  message(STATUS "Xcode: ${XCODE_VERSION}")
endif()
if(NOT CMAKE_GENERATOR MATCHES "Xcode|Visual Studio")
  message(STATUS "Configuration: ${CMAKE_BUILD_TYPE}")
endif()

message(STATUS "-----------------------------------------------------------------")

find_package(BehaviorTreeV3 REQUIRED) 
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Threads REQUIRED)

message(STATUS "BehaviorTreeV3 version: ${BehaviorTreeV3_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "spdlog version: ${spdlog_VERSION}")
message(STATUS "Threads version: ${Threads_VERSION}")

#---------------------------------------------------------------------------------------
# Compiler Options
#---------------------------------------------------------------------------------------

message(STATUS "-----------------------------------------------------------------")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-Wall -Wextra -O2")

message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags (Release): ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags (Debug): ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
if(WIN32)
  message(STATUS "Linker flags (Release): ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")
  message(STATUS "Linker flags (Debug): ${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_DEBUG}")
else()
  message(STATUS "Linker flags (Release): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_RELEASE}")
  message(STATUS "Linker flags (Debug): ${CMAKE_SHARED_LINKER_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS_DEBUG}")
endif()

#---------------------------------------------------------------------------------------
# Building Options
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/device)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision)

#---------------------------------------------------------------------------------------
# Applications
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/auto_aim)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/full_auto)

#---------------------------------------------------------------------------------------
# Install
#---------------------------------------------------------------------------------------

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "-----------------------------------------------------------------")
    message(STATUS "Generating install")
    message(STATUS "Install to: ${CMAKE_INSTALL_PREFIX}")

    install(
        TARGETS auto_aim
        EXPORT auto_aim
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        TARGETS full_auto
        EXPORT full_auto
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

endif()

message(STATUS "-----------------------------------------------------------------")