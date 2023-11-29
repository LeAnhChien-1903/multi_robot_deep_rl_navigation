cmake_minimum_required(VERSION 3.0.2)
project(multi_robot_environment)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  stage_ros
  
)
find_package(Eigen3 REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES multi_robot_environment
#  CATKIN_DEPENDS roscpp stage_ros
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${EIGEN3_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  stage
)
add_executable(random_scenarios src/random_scenarios.cpp)
target_link_libraries(random_scenarios ${catkin_LIBRARIES})
