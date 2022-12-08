cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cpptorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

aux_source_directory(. SOURCES)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(test_model ${SOURCES})

target_link_libraries(test_model "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

set_property(TARGET test_model PROPERTY CXX_STANDARD 14)

# message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")
# message(STATUS "TORCH_LIBRARIES = ${TORCH_LIBRARIES}")