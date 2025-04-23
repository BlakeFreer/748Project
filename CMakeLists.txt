cmake_minimum_required(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

project(proj748)
enable_language(C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

add_library(resources)
target_include_directories(resources PUBLIC inc)

# Add the compiler's library and include paths to CMake search paths
set(CMAKE_LIBRARY_PATH "${CMAKE_LIBRARY_PATH};${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
set(CMAKE_INCLUDE_PATH "${CMAKE_INCLUDE_PATH};${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}")
set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})

# Find FFTW using the extended search paths
find_library(FFTW_LIB fftw3)
find_path(FFTW_INCLUDE_DIR fftw3.h)

if(FFTW_LIB AND FFTW_INCLUDE_DIR)
    message(STATUS "Found FFTW library: ${FFTW_LIB}")
    message(STATUS "Found FFTW include directory: ${FFTW_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "FFTW was not found on your system!")
endif()

target_link_libraries(resources PUBLIC ${FFTW_LIB})

# libsndfile
find_library(LIBSNDFILE NAMES sndfile PATHS third-party/libsndfile/build)
target_link_libraries(resources PUBLIC ${LIBSNDFILE})
target_include_directories(resources PUBLIC third-party/libsndfile/include)

add_subdirectory(src)

add_executable(extract extract.cpp)
target_link_libraries(extract resources)

add_executable(basis basis.cpp)
target_link_libraries(basis resources)

add_executable(reduce reduce.cpp)
target_link_libraries(reduce resources)

add_executable(prep-svm prep_svm.cpp)
target_link_libraries(prep-svm resources)