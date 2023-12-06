cmake_minimum_required(VERSION 3.0.2)
project(deep_rl_navigation_2021)

find_package(catkin REQUIRED COMPONENTS
    dynamic_reconfigure
    roscpp
    rospy
)

include_directories(
# include
    ${catkin_INCLUDE_DIRS}
    include/${PROJECT_NAME}
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS 
    src/train_policy.py
    src/test_policy.py
    src/train_policy_ppo.py
    src/test_policy_ppo.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)