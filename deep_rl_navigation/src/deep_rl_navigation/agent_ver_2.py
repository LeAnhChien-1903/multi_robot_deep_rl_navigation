#!/usr/bin/env python

import rospy
from tf.transformations import euler_from_quaternion

import numpy as np
import math
import os

import torch
import random
import torch.nn as neuron_network

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32

from deep_rl_navigation.ultis import *
from deep_rl_navigation.actor_critic import *


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
        # Actor and Critic
        self.actor = Actor(self.actor_parameters_path)
        # self.actor.save_parameters(self.actor_parameters_path)
        self.critic = Critic(self.critic_parameters_path)
        # self.critic.save_parameters(self.critic_parameters_path)
        # Step initialization
        self.step: int = 0
        # Initialize the storage
        self.initializeStorage()
        self.folder_name = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/data" + self.robot_name
        if not os.path.exists(self.folder_name):
            os.mkdir(self.folder_name)
        # Initialize Publisher and Subscriber
        self.initializePublisherAndSubscriber()

    def timerCallback(self, event):
        '''
            Timer callback function for implement navigation
        '''
        if self.step < self.time_step_size:
            if (self.robot_name == "/robot_0"):
                print(self.step + 1)
            done = self.goalReached()
            # Set the current observation
            self.observation.setObservation(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel)
            laser_obs, goal_obs, vel_obs = self.transformObservation()
            
            # Get the reward
            reward = self.reward.calculateReward(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel, self.setup_params.robot_radius)
            self.reward_storage[self.step] = torch.tensor(reward)
            # Store the current observation
            self.laser_obs_storage[self.step] = laser_obs
            self.goal_obs_storage[self.step] = goal_obs
            self.vel_obs_storage[self.step] = vel_obs
            
            # Get action and value function
            action, log_prob, entropy, mean, std = self.actor.get_action(laser_obs, goal_obs, vel_obs)
            value = self.critic.get_value(laser_obs, goal_obs, vel_obs)
            self.action_storage[self.step] = action
            self.mean_storage[self.step] = mean
            self.std_storage[self.step] = std
            self.log_prob_storage[self.step] = log_prob
            self.entropy_storage[self.step] = entropy
            self.value_storage[self.step] = value
            self.done_storage[self.step] = torch.tensor(done)
            
            reward_msg = Float32()
            reward_msg.data = reward
            self.reward_pub.publish(reward_msg)
            
            cmd_vel = Twist()
            cmd_vel.linear.x = action.numpy()[0]
            cmd_vel.angular.z = action.numpy()[1]
            self.cmd_vel_pub.publish(cmd_vel)
            self.step += 1
        else:
            cmd_vel = Twist()
            self.cmd_vel_pub.publish(cmd_vel)
            # Store the data to files
            torch.save(self.laser_obs_storage, self.folder_name + "/" + "laser_obs.pt")
            torch.save(self.goal_obs_storage, self.folder_name + "/" + "goal_obs.pt")
            torch.save(self.vel_obs_storage, self.folder_name + "/" + "vel_obs.pt")
            torch.save(self.reward_storage, self.folder_name + "/" + "reward.pt")
            torch.save(self.action_storage, self.folder_name + "/" + "actions.pt")
            torch.save(self.log_prob_storage, self.folder_name + "/" + "log_prob.pt")
            torch.save(self.value_storage, self.folder_name + "/" + "value.pt")
            torch.save(self.done_storage, self.folder_name + "/" + "done.pt")
            torch.save(self.entropy_storage, self.folder_name + "/" + "entropy.pt")
            torch.save(self.mean_storage, self.folder_name + "/" + "mean.pt")
            torch.save(self.std_storage, self.folder_name + "/" + "std.pt")

    def initializeStorage(self):
        """
            Setup the storage for observation, action, log probability, reward, value, done state
        """
        self.laser_obs_storage = torch.zeros((self.time_step_size, 1, self.setup_params.num_observations, self.setup_params.num_laser_ray))
        self.goal_obs_storage = torch.zeros((self.time_step_size, 2))
        self.vel_obs_storage = torch.zeros((self.time_step_size, 2))
        self.action_storage = torch.zeros((self.time_step_size, 2))
        self.mean_storage = torch.zeros((self.time_step_size, 2))
        self.std_storage = torch.zeros((self.time_step_size, 2))
        self.log_prob_storage = torch.zeros(self.time_step_size)
        self.reward_storage = torch.zeros(self.time_step_size)
        self.value_storage = torch.zeros(self.time_step_size)
        self.done_storage = torch.zeros(self.time_step_size)
        self.entropy_storage = torch.zeros(self.time_step_size)
        
        
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

    def transformObservation(self):
        
        laser_obs = torch.from_numpy(self.observation.laser_data.reshape((1, self.setup_params.num_observations, self.setup_params.num_laser_ray)))
        goal_obs = torch.from_numpy(self.observation.goal_relation)
        vel_obs = torch.from_numpy(self.observation.current_velocity)
        
        return laser_obs, goal_obs, vel_obs
    
    def goalReached(self):
        distance = calculatedDistance(self.robot_pose[0:2], self.goal_pose[0:2])
        if distance < self.setup_params.goal_tolerance:
            return True
        
        return False
    
    def randomPose(self):        
        return np.array([random.uniform(-self.setup_params.map_width/2 + 0.5, self.setup_params.map_width/2 - 0.5), 
                        random.uniform(-self.setup_params.map_length/2 + 0.5, self.setup_params.map_length/2 - 0.5),
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
        self.number_of_robot = rospy.get_param("/number_of_robot")
        
        self.time_step_size = math.ceil(self.ppo_params.T_max / self.number_of_robot) + 1