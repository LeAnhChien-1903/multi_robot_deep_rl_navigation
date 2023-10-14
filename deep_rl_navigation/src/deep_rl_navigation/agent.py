#!/usr/bin/env python

import rospy
from tf.transformations import euler_from_quaternion

import numpy as np
import math

import torch
import random
import torch.nn as neuron_network

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

from deep_rl_navigation.ultis import *
from deep_rl_navigation.actor_critic import *

# Seeding
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, name: str):
        self.robot_name = name
        self.setup_params = Setup() # setup parameters for navigation
        self.ppo_params = PPO() # ppo parameters for training
        self.hybrid_params = Hybrid() # hybrid parameters for hybrid control
        self.reward = Reward()
        # Initialize hyper parameters
        self.initializeParameters()
        # Initialize observation
        self.observation = Observation(self.setup_params.num_observations, self.setup_params.num_laser_ray)
        # Robot pose and goal pose
        self.robot_pose = np.zeros(3, dtype=np.float32)
        self.goal_pose = self.randomPose()
        self.current_vel = np.zeros(2, dtype=np.float32)
        # Laser data
        self.laser_data = np.zeros(self.setup_params.num_laser_ray, dtype=np.float32)
        # Actor 
        self.actor = Actor(self.actor_parameters_path).to(device)
        # Critic
        self.critic = Critic(self.critic_parameters_path).to(device)
        # Initialize Publisher and Subscriber
        self.initializePublisherAndSubscriber()
    
    def timerCallback(self, event):
        '''
            Timer callback function for implement navigation
        '''
        # Set the current observation
        self.observation.setObservation(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel)
        
        reward = Float32()
        reward.data = self.reward.calculateReward(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel, self.setup_params.robot_radius)
        self.reward_pub.publish(reward)
        
        # Reshape the observation
        laser_obs, goal_obs, vel_obs = self.transformObservationCPUToGPU()
        action, _, _ = self.actor.get_action(laser_obs, goal_obs, vel_obs, device)
        cmd_vel = Twist()
        cmd_vel.linear.x = action.cpu().numpy()[0][0]
        cmd_vel.angular.z = action.cpu().numpy()[0][1]
        self.cmd_vel_pub.publish(cmd_vel)
    
    def initializePublisherAndSubscriber(self):
        '''
            Initialize the publisher and subscriber
        '''
        self.odometry_sub = rospy.Subscriber(self.robot_name + "/" + self.setup_params.odometry_topic, Odometry, self.odometryCallback)
        self.laser_scan_sub = rospy.Subscriber(self.robot_name + "/" + self.setup_params.laser_scan_topic, LaserScan, self.getLaserData)
        self.cmd_vel_pub = rospy.Publisher(self.robot_name + "/" + self.setup_params.cmd_vel_topic, Twist, queue_size= 10)
        self.reward_pub = rospy.Publisher(self.robot_name + "/reward", Float32, queue_size= 10)
        self.timer = rospy.Timer(rospy.Duration(self.setup_params.sample_time), self.timerCallback)

    def odometryCallback(self, odom: Odometry):
        '''
            Get robot pose and robot velocity from odometry ground truth
            ### Parameter
            odom: odometry message
        '''
        # Get current robot pose
        self.robot_pose[0] = odom.pose.pose.position.x
        self.robot_pose[1] = odom.pose.pose.position.y
        
        rqy = euler_from_quaternion([odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w])
        self.robot_pose[2] = rqy[2]
        # Get current velocity
        self.current_vel[0] = odom.twist.twist.linear.x
        self.current_vel[1] = odom.twist.twist.angular.z
    
    def getLaserData(self, laser_scan: LaserScan):
        '''
            Get data from lidar 2D
            ### Parameters
            laser_scan: laser scan topic
        '''
        # Convert laser scan to center of robot
        extra_x = self.setup_params.lidar_x * math.cos(self.robot_pose[2]) - self.setup_params.lidar_y * math.sin(self.robot_pose[2])
        extra_y = self.setup_params.lidar_x * math.sin(self.robot_pose[2]) + self.setup_params.lidar_y * math.cos(self.robot_pose[2])
        extra_theta = normalize_angle(self.robot_pose[2] + self.setup_params.lidar_theta)
        for i in range(len(laser_scan.ranges)):
            if (laser_scan.ranges[i] < laser_scan.range_min and laser_scan.ranges[i] > laser_scan.range_max):
                self.laser_data[i] = 500.0
            else:
                angle = normalize_angle(laser_scan.angle_min + i * laser_scan.angle_increment + extra_theta)
                range_x = laser_scan.ranges[i] * math.cos(angle) + extra_x
                range_y = laser_scan.ranges[i] * math.sin(angle) + extra_y
                self.laser_data[i] = math.hypot(range_x, range_y)

    def transformObservationCPUToGPU(self):
        laser_obs = torch.from_numpy(self.observation.laser_data.reshape((1, self.setup_params.num_observations, self.setup_params.num_laser_ray))).to(device)
        goal_obs = torch.from_numpy(self.observation.goal_relation).to(device)
        vel_obs = torch.from_numpy(self.observation.current_velocity).to(device)
        
        return laser_obs, goal_obs, vel_obs
    
    def randomPose(self):        
        return np.array([random.uniform(-self.setup_params.map_width/2, self.setup_params.map_width/2), 
                            random.uniform(-self.setup_params.map_length/2, self.setup_params.map_length/2),
                            random.uniform(-math.pi, math.pi)])

    def initializeParameters(self):
        '''
            Initialize the parameters from ros param
        '''
        # Setup parameters
        self.setup_params.sample_time = rospy.get_param("/sample_time")
        
        self.setup_params.laser_scan_topic = rospy.get_param("/laser_scan_topic")
        self.setup_params.odometry_topic = rospy.get_param("/odometry_topic")
        self.setup_params.cmd_vel_topic = rospy.get_param("/cmd_vel_topic")
        
        self.setup_params.min_linear_velocity = rospy.get_param("/min_linear_velocity")
        self.setup_params.max_linear_velocity = rospy.get_param("/max_linear_velocity")
        self.setup_params.min_angular_velocity = rospy.get_param("/min_angular_velocity")
        self.setup_params.max_angular_velocity = rospy.get_param("/max_angular_velocity")
        self.setup_params.max_linear_acceleration = rospy.get_param("/max_linear_acceleration")
        self.setup_params.max_angular_acceleration = rospy.get_param("/max_angular_acceleration")
        self.setup_params.num_observations = rospy.get_param("/num_observations")
        self.setup_params.num_laser_ray = rospy.get_param("/num_laser_ray")
        self.setup_params.goal_tolerance = rospy.get_param("/goal_tolerance")
        
        self.setup_params.robot_length = rospy.get_param("/robot_length")
        self.setup_params.robot_width = rospy.get_param("/robot_width")
        
        self.setup_params.lidar_x = rospy.get_param("/lidar_x")
        self.setup_params.lidar_y = rospy.get_param("/lidar_y")
        self.setup_params.lidar_theta = rospy.get_param("/lidar_theta")
        
        self.setup_params.map_length = rospy.get_param("/map_length")
        self.setup_params.map_width = rospy.get_param("/map_width")
        
        self.setup_params.robot_radius = math.hypot(self.setup_params.robot_width/2, self.setup_params.robot_length/2) + 0.05
        # Reward parameters
        self.reward.r_arrival = rospy.get_param("/reward/r_arrival")
        self.reward.r_collision = rospy.get_param("/reward/r_collision")
        self.reward.omega_g = rospy.get_param("/reward/omega_g")
        self.reward.omega_w = rospy.get_param("/reward/omega_w")
        self.reward.large_angular_velocity = rospy.get_param("/reward/large_angular_velocity")
        self.reward.goal_tolerance = rospy.get_param("/goal_tolerance")

        # PPO hyper parameters
        self.ppo_params.lambda_ = rospy.get_param("/ppo/lambda")
        self.ppo_params.gamma = rospy.get_param("/ppo/gamma")
        self.ppo_params.T_max = rospy.get_param("/ppo/T_max")
        self.ppo_params.E_phi = rospy.get_param("/ppo/E_phi")
        self.ppo_params.beta = rospy.get_param("/ppo/beta")
        self.ppo_params.KL_target = rospy.get_param("/ppo/KL_target")
        self.ppo_params.xi = rospy.get_param("/ppo/xi")
        self.ppo_params.lr_theta_1st = rospy.get_param("/ppo/lr_theta_1st")
        self.ppo_params.lr_theta_2nd = rospy.get_param("/ppo/lr_theta_2nd")
        self.ppo_params.E_v = rospy.get_param("/ppo/E_v")
        self.ppo_params.lr_phi = rospy.get_param("/ppo/lr_phi")
        self.ppo_params.beta_high = rospy.get_param("/ppo/beta_high")
        self.ppo_params.alpha = rospy.get_param("/ppo/alpha")
        self.ppo_params.beta_low = rospy.get_param("/ppo/beta_low")
        # Hybrid control hyper parameters
        self.hybrid_params.r_safe = rospy.get_param("/hybrid/r_safe")
        self.hybrid_params.r_risk = rospy.get_param("/hybrid/r_risk")
        self.hybrid_params.p_scale = rospy.get_param("/hybrid/p_scale")
        self.hybrid_params.v_max = rospy.get_param("/hybrid/v_max")
        
        # Path to save parameters
        self.actor_parameters_path = rospy.get_param("/actor_parameters_path")
        self.critic_parameters_path = rospy.get_param("/critic_parameters_path")