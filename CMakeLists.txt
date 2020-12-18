cmake_minimum_required(VERSION 3.12)

project(vulkan-practice LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set(GLFW_INCLUDEDIR "" CACHE STRING "path to GLFW include directory.")
set(GLFW_LIBRARYDIR "" CACHE STRING "path to GLFW library directory.")
set(GLM_INCLUDEDIR "" CACHE STRING "path to glm include directory.")

find_package(Vulkan REQUIRED)

find_library(GLFW_LIB NAMES glfw3 HINTS ${GLFW_LIBRARYDIR})

if(NOT GLFW_LIB)
    message(FATAL "GLFW library not found!")
endif()

message(STATUS "GLFW library found: ${GLFW_LIB}")

find_path(GLFW_INCLUDE_DIR NAMES "GLFW/glfw3.h" HINTS ${GLFW_INCLUDEDIR})

if(NOT GLFW_INCLUDE_DIR)
    message(FATAL "GLFW headers not found!")
endif()

message(STATUS "GLFW header path: ${GLFW_INCLUDE_DIR}")

find_path(GLM_INCLUDE_DIR NAMES "glm/vec4.hpp" HINTS ${GLM_INCLUDEDIR})

if(NOT GLM_INCLUDE_DIR)
    message(FATAL "glm headers not found!")
endif()

message(STATUS "glm header path: ${GLM_INCLUDE_DIR}")

include_directories(
    ${GLFW_INCLUDE_DIR}
    ${GLM_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/external/stb
    ${PROJECT_SOURCE_DIR}/external/tinyobjloader
)

add_executable(vulkan-test ${PROJECT_SOURCE_DIR}/test.cpp)
add_executable(vulkan-triangle ${PROJECT_SOURCE_DIR}/triangle.cpp)

target_link_libraries(vulkan-test ${Vulkan_LIBRARIES} ${GLFW_LIB})
target_link_libraries(vulkan-triangle ${Vulkan_LIBRARIES} ${GLFW_LIB})
