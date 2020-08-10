cmake_minimum_required(VERSION 3.5)
project(qdu_rm_ai VERSION 0.1.0)

#---------------------------------------------------------------------------------------
# Options
#---------------------------------------------------------------------------------------

# Allow switching of Applications
set(APPLICATION_LIST auto_aim full_auto)
set(APPLICATION auto_aim CACHE STRING "Default Application is auto_aim")
set_property(CACHE APPLICATION PROPERTY STRINGS ${APPLICATION_LIST})

#---------------------------------------------------------------------------------------
# General Components
#---------------------------------------------------------------------------------------

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/behavior)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/vision)

#---------------------------------------------------------------------------------------
# Applications
#---------------------------------------------------------------------------------------

if(APPLICATION STREQUAL auto_aim)

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/auto_aim)

elseif(APPLICATION STREQUAL full_auto)

    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/app/full_auto)

endif()


