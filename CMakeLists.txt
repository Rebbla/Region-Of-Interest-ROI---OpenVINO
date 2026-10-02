cmake_minimum_required(VERSION 3.16)
project(trash_sort_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(yaml-cpp REQUIRED)

add_executable(trash_sort
  src/main.cpp
  src/yolo_ov.cpp
  src/roi.cpp
)

target_include_directories(trash_sort PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(trash_sort PRIVATE
  ${OpenCV_LIBS}
  openvino::runtime
  yaml-cpp
)

install(TARGETS trash_sort RUNTIME DESTINATION bin)
