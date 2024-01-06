#!/usr/bin/env python3
import os
import math
import numpy as np
world_dir = "collision_avoidance/world"
launch_dir = "collision_avoidance/launch"
goal_dir = "collision_avoidance/goal_point"
pose_dir = "collision_avoidance/init_pose"
map_name = "multi_env"
scale = 10
map_size = [100.0, 80.0]
window_size = [1050.0, 850.0]
num_of_robot = 20
robot_width = 0.38
robot_length = 0.44
robot_height = 0.22
laser_max_range = 6.0
laser_fov = 180
laser_samples = 512
laser_pose = [0.0, 0.0, 0.0, 0.0]

world_file = open(os.path.join(world_dir, "multi_env_{}.world".format(num_of_robot)), 'w')

world_file.write(
"""
show_clock 0
show_clock_interval 10000
resolution 0.025
threads 4
speedup 1

define laser ranger
(
    sensor(
        pose [ 0 0 0.1 0 ]
        fov {}
        range [ 0.0 {} ]
        samples {}
    )
    color "random"
    block( 
        points 4
        point[0] [0 0]
        point[1] [0 1]
        point[2] [1 1]
        point[3] [1 0]
        z [0 0.21]
    )
)


define floor model
(
    color "gray30"
    boundary 1

    gui_nose 0
    gui_grid 0
    gui_move 0
    gui_outline 0
    gripper_return 0
    fiducial_return 0
    ranger_return 1
    obstacle_return 1
)

floor
(
    name "{}"
    bitmap "../maps/{}.png"
    size [{} {} 2.00]
    pose [0.000 0.000 0.000 0.000]
)

window
(
    size [{} {}]
    center [0.000000 0.000000] # Camera options 
    rotate [0.000000 0.000000] # Camera options 
    scale {}
    show_data 1
    show_grid 1
    show_trailarrows 1
)
define agent position
(
    # actual size
    size [{} {} {}] # sizes from MobileRobots' web site

    # the pioneer's center of rotation is offset from its center of area
    origin [0 0 0 0]

    # draw a nose on the robot so we can see which way it points
    gui_nose 1

    color "random"
    drive "diff"		 	# Differential steering model.
    obstacle_return 1           	# Can hit things.
    ranger_return 0.5            	# reflects sonar beams
    blob_return 1               	# Seen by blobfinders  
    fiducial_return 1           	# Seen as "1" fiducial finders
    laser
    (
        pose [ {} {} {} {} ] 
    )
)
""".format(laser_fov, laser_max_range, laser_samples, map_name, map_name, 
            map_size[0], map_size[1], window_size[0], window_size[1], scale, 
            robot_length, robot_width, robot_height, laser_pose[0], laser_pose[1], 
            laser_pose[2], laser_pose[3]))

step = 2 * math.pi / num_of_robot
goal_point = []
init_pose = []

world_file.write("agent( pose [{} {} 0.0 {}])\n".format(-31.0, 37.0, -90))
init_pose.append([-31.0, 37.0, -math.pi/2])
goal_point.append([-31.0, 5.0])
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(-31.0, 5.0, 90))
init_pose.append([-31.0, 5.0, math.pi/2])
goal_point.append([-31.0, 37.0])
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(-15.0, 21.0, 180))
init_pose.append([-15.0, 21.0, math.pi])
goal_point.append([-47.0, 21.0])
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(-47.0, 21.0, 0))
init_pose.append([-47.0, 21.0, 0.0])
goal_point.append([-15.0, 21.0])

# agent( pose [-6.139 31.382 0.000 -90.000])
# agent( pose [13.912 5.763 0.000 90.000])
# agent( pose [13.839 31.433 0.000 180.000])
# agent( pose [-5.951 6.110 0.000 0.000])

world_file.write("agent( pose [{} {} 0.0 {}])\n".format(-6.0, 31.5, 0.0))
init_pose.append([-6.0, 31.5, 0.0])
goal_point.append([14.0, 6.5])
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(14.0, 31.5, 180))
init_pose.append([14.0, 31.5, math.pi])
goal_point.append([-6.0, 6.5])
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(-6.0, 6.5, 0.0))
init_pose.append([-6.0, 6.5, 0.0])
goal_point.append([14.0, 31.5])
world_file.write("agent( pose [{} {} 0.0 {}])\n".format(14.0, 6.5, 180))
init_pose.append([14.0, 6.5, math.pi])
goal_point.append([-6.0, 31.5])
world_file.close()

np.savetxt(os.path.join(goal_dir, "multi_env_{}.txt".format(num_of_robot)), np.array(goal_point), fmt= '%.2f')
np.savetxt(os.path.join(pose_dir, "multi_env_{}.txt".format(num_of_robot)), np.array(init_pose), fmt= '%.5f')

launch_file =open(os.path.join(launch_dir, "multi_env_{}.launch".format(num_of_robot)), 'w')
launch_file.write(
"""
<launch>

    <!--  ************** Global Parameters ***************  -->
    <param name="/use_sim_time" value="true"/>
    <arg name="world_file" default="$(find collision_avoidance)/world/multi_env_{}.world" />
    <node name="stageros" type="stageros" pkg="stage_ros_add_pose_and_crash" args=" $(arg world_file)"/>
""".format(num_of_robot))
for i in range(8):
    launch_file.write('\t<node pkg="collision_avoidance" type="test_policy.py" name="robot_{}" output="screen"/>\n'.format(i))
launch_file.write("</launch>")
launch_file.close()
