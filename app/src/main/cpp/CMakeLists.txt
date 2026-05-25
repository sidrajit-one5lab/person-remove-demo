# Native engine for PersonRemoval.
# Step 6a: minimal scaffold — one function returning OpenCV's version string.
# Real engine code (buffer, aligner, stitcher, blender) is added in Step 6b.
cmake_minimum_required(VERSION 3.22.1)
project(personremoval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Point CMake at OpenCV's bundled config inside our :opencv module.
# CMAKE_SOURCE_DIR is app/src/main/cpp; the opencv module lives 4 levels up.
# OpenCV's CMake config walks UP 4 dirs from its location to compute _IMPORT_PREFIX,
# then expects 3rdparty libs at "$prefix/sdk/native/3rdparty/..." — hence the sdk/ level.
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../../../opencv/sdk/native/jni)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

add_library(personremoval SHARED
        jni_bridge.cpp
        buffer.cpp
        aligner.cpp
        stitcher.cpp
        blender.cpp
)

target_include_directories(personremoval PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(personremoval
        ${OpenCV_LIBS}
        log
        android
        jnigraphics
)
