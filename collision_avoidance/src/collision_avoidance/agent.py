#!/usr/bin/env python3
import time
import rospy
import copy
import tf
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque
import math
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from rosgraph_msgs.msg import Clock
from std_srvs.srv import Empty
from std_msgs.msg import Int8
from collision_avoidance.ultis import test_init_pose, test_goal_point

from collision_avoidance.network import CNNPolicy
from collision_avoidance.ppo import generate_action_no_sampling


MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 200
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 3
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 50
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5

# policy_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/collision_avoidance/parameters"
# # policy = MLPPolicy(obs_size, act_size)
# policy = CNNPolicy(frames=LASER_HIST, action_space=2)
# policy.cuda()

# file = policy_path + '/stage2.pth'
# print ('############Loading Model###########')
# state_dict = torch.load(file)
# policy.load_state_dict(state_dict)
action_bound = [[0, -1], [1, 1]]
class Agent():
    def __init__(self, beam_num, index, world_name):
        self.index = index
        self.beam_num = beam_num
        self.world_name = world_name
        self.laser_cb_num = 0
        self.scan = None
        
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.

        # used in generate goal point
        self.map_size = np.array([8., 8.], dtype=np.float32)  # 20x20m
        self.goal_size = 0.5

        self.robot_value = 10.
        self.goal_value = 0.

        # -----------Publisher and Subscriber-------------
        cmd_vel_topic = 'robot_' + str(index) + '/cmd_vel'
        self.cmd_vel = rospy.Publisher(cmd_vel_topic, Twist, queue_size=10)

        cmd_pose_topic = 'robot_' + str(index) + '/cmd_pose'
        self.cmd_pose = rospy.Publisher(cmd_pose_topic, Pose, queue_size=10)

        object_state_topic = 'robot_' + str(index) + '/base_pose_ground_truth'
        self.object_state_sub = rospy.Subscriber(object_state_topic, Odometry, self.ground_truth_callback)

        laser_topic = 'robot_' + str(index) + '/base_scan'

        self.laser_sub = rospy.Subscriber(laser_topic, LaserScan, self.laser_scan_callback)

        odom_topic = 'robot_' + str(index) + '/odom'
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odometry_callback)

        crash_topic = 'robot_' + str(index) + '/is_crashed'
        self.check_crash = rospy.Subscriber(crash_topic, Int8, self.crash_callback)


        self.sim_clock = rospy.Subscriber('clock', Clock, self.sim_clock_callback)

        # -----------Service-------------------
        self.reset_stage = rospy.ServiceProxy('reset_positions', Empty)
        # # Wait until the first callback
        self.speed = None
        self.state = None
        self.speed_GT = None
        self.state_GT = None
        self.is_crashed = None
        while self.scan is None or self.speed is None or self.state is None\
                or self.speed_GT is None or self.state_GT is None or self.is_crashed is None:
            pass
        rospy.sleep(1.)
        if index == 0:
            self.reset_world()
        self.reset_pose()
        self.generate_goal_point()
        obs = self.get_laser_observation()
        self.obs_stack = deque([obs, obs, obs])
        goal = np.asarray(self.get_local_goal())
        speed = np.asarray(self.get_self_speed())
        self.state_obs = [self.obs_stack, goal, speed]
        
        # timer = rospy.Timer(rospy.Duration(0.1), self.timerCallback)
    
    def timerCallback(self, event):
        pass
        # ex = self.goal_point[0] - self.state_GT[0]
        # ey = self.goal_point[1] - self.state_GT[1]
        # if math.hypot(ex, ey) < 0.1:
        #     self.control_vel([[0.0, 0.0]])
        # else:
        #     laser = np.asarray(self.state_obs[0])
        #     goal = np.asarray(self.state_obs[1])
        #     speed = np.asarray(self.state_obs[2])
        #     laser_obs = Variable(torch.from_numpy(laser.reshape(1, 3, self.beam_num))).float().cuda()
        #     goal_obs = Variable(torch.from_numpy(goal.reshape(1, 2))).float().cuda()
        #     speed_obs = Variable(torch.from_numpy(speed.reshape(1, 2))).float().cuda()
            
        #     _, _, _, mean = policy(laser_obs, goal_obs, speed_obs)
        #     mean = mean.data.cpu().numpy()
        #     scaled_action = np.clip(mean, a_min=action_bound[0], a_max=action_bound[1])

        #     self.control_vel(scaled_action)
        #     s_next = self.get_laser_observation()
        #     left = self.obs_stack.popleft()
        #     self.obs_stack.append(s_next)
        #     goal_next = np.asarray(self.get_local_goal())
        #     speed_next = np.asarray(self.get_self_speed())
        #     state_next = [self.obs_stack, goal_next, speed_next]


        #     self.state_obs = state_next
        
    def ground_truth_callback(self, GT_odometry):
        quaternion = GT_odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        self.state_GT = [GT_odometry.pose.pose.position.x, GT_odometry.pose.pose.position.y, Euler[2]]
        v_x = GT_odometry.twist.twist.linear.x
        v_y = GT_odometry.twist.twist.linear.y
        v = np.sqrt(v_x**2 + v_y**2)
        self.speed_GT = [v, GT_odometry.twist.twist.angular.z]

    def laser_scan_callback(self, scan):
        self.scan_param = [scan.angle_min, scan.angle_max, scan.angle_increment, scan.time_increment,
                            scan.scan_time, scan.range_min, scan.range_max]
        self.scan = np.array(scan.ranges)
        self.laser_cb_num += 1


    def odometry_callback(self, odometry):
        Quaternions = odometry.pose.pose.orientation
        Euler = tf.transformations.euler_from_quaternion([Quaternions.x, Quaternions.y, Quaternions.z, Quaternions.w])
        self.state = [odometry.pose.pose.position.x, odometry.pose.pose.position.y, Euler[2]]
        self.speed = [odometry.twist.twist.linear.x, odometry.twist.twist.angular.z]

    def sim_clock_callback(self, clock):
        self.sim_time = clock.clock.secs + clock.clock.nsecs / 1000000000.

    def crash_callback(self, flag):
        self.is_crashed = flag.data

    def get_self_stateGT(self):
        return self.state_GT

    def get_self_speedGT(self):
        return self.speed_GT

    def get_laser_observation(self):
        scan = copy.deepcopy(self.scan)
        scan[np.isnan(scan)] = 6.0
        scan[np.isinf(scan)] = 6.0
        raw_beam_num = len(scan)
        sparse_beam_num = self.beam_num
        step = float(raw_beam_num) / sparse_beam_num
        sparse_scan_left = []
        index = 0.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_left.append(scan[int(index)])
            index += step
        sparse_scan_right = []
        index = raw_beam_num - 1.
        for x in range(int(sparse_beam_num / 2)):
            sparse_scan_right.append(scan[int(index)])
            index -= step
        scan_sparse = np.concatenate((sparse_scan_left, sparse_scan_right[::-1]), axis=0)
        return scan_sparse / 6.0 - 0.5


    def get_self_speed(self):
        return self.speed

    def get_self_state(self):
        return self.state

    def get_crash_state(self):
        return self.is_crashed

    def get_sim_time(self):
        return self.sim_time

    def get_local_goal(self):
        [x, y, theta] = self.get_self_stateGT()
        [goal_x, goal_y] = self.goal_point
        local_x = (goal_x - x) * np.cos(theta) + (goal_y - y) * np.sin(theta)
        local_y = -(goal_x - x) * np.sin(theta) + (goal_y - y) * np.cos(theta)
        return [local_x, local_y]

    def reset_world(self):
        self.reset_stage()
        self.self_speed = [0.0, 0.0]
        self.step_goal = [0., 0.]
        self.step_r_cnt = 0.
        self.start_time = time.time()
        rospy.sleep(0.5)


    def generate_goal_point(self):
        goal_point_list = np.loadtxt("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/collision_avoidance/goal_point/{}.txt".format(self.world_name))
        self.goal_point = goal_point_list[self.index]
        self.pre_distance = 0
        self.distance = copy.deepcopy(self.pre_distance)



    def get_reward_and_terminate(self, t):
        terminate = False
        laser_scan = self.get_laser_observation()
        [x, y, theta] = self.get_self_stateGT()
        [v, w] = self.get_self_speedGT()
        self.pre_distance = copy.deepcopy(self.distance)
        self.distance = np.sqrt((self.goal_point[0] - x) ** 2 + (self.goal_point[1] - y) ** 2)
        reward_g = (self.pre_distance - self.distance) * 2.5
        reward_c = 0
        reward_w = 0
        result = 0

        is_crash = self.get_crash_state()

        if self.distance < self.goal_size:
            terminate = True
            reward_g = 15
            result = 'Reach Goal'

        if is_crash == 1:
            terminate = True
            reward_c = -15.
            result = 'Crashed'

        if np.abs(w) >  0.7:
            reward_w = -0.1 * np.abs(w)

        if t > 10000:
            terminate = True
            result = 'Time out'
        reward = reward_g + reward_c + reward_w

        return reward, terminate, result

    def reset_pose(self):
        init_pose = np.loadtxt("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/collision_avoidance/init_pose/{}.txt".format(self.world_name))
        reset_pose = init_pose[self.index]
        self.control_pose(reset_pose)


    def control_vel(self, action):
        move_cmd = Twist()
        move_cmd.linear.x = action[0][0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = action[0][1]
        self.cmd_vel.publish(move_cmd)


    def control_pose(self, pose):
        pose_cmd = Pose()
        assert len(pose)==3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        self.cmd_pose.publish(pose_cmd)


    def generate_random_pose(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y * 10 + 1)
        else:
            y = -(y * 10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        theta = np.random.uniform(0, 2*np.pi)
        return [x, y, theta]

    def generate_random_goal(self):
        [x_robot, y_robot, theta] = self.get_self_stateGT()
        x = np.random.uniform(9, 19)
        y = np.random.uniform(0, 1)
        if y <= 0.4:
            y = -(y*10 + 1)
        else:
            y = -(y*10 + 9)
        dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        while (dis_goal < 7) and not rospy.is_shutdown():
            x = np.random.uniform(9, 19)
            y = np.random.uniform(0, 1)
            if y <= 0.4:
                y = -(y * 10 + 1)
            else:
                y = -(y * 10 + 9)
            dis_goal = np.sqrt((x - x_robot) ** 2 + (y - y_robot) ** 2)
        return [x, y]