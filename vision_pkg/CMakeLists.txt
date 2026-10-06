cmake_minimum_required(VERSION 3.8)
project(vision_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(ament_index_cpp REQUIRED)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(yaml-cpp REQUIRED)

add_executable(vision_node
  src/vision_node.cpp
  src/yolo_ov.cpp
  src/roi.cpp
)

target_include_directories(vision_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include/vision_pkg
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vision_node
  ${OpenCV_LIBS}
  openvino::runtime
  yaml-cpp
)

ament_target_dependencies(vision_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  ament_index_cpp
)

install(TARGETS vision_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
