#!/usr/bin/env python

import rospy
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import torch
import torch.nn as neurons
import torchvision 

import random

from deep_rl_navigation.ultis import *

class DeepReinforcementLearningNavigator:
    def __init__(self, name: str):
        self.robot_name = name
        self.setup_params = Setup() # setup params for navigation
        self.ppo_params = PPO()
        self.hybrid_params = Hybrid()
        self.reward = Reward()
        # Init parameters
        self.initializeParameters()
        self.observation = Observation(self.setup_params.num_observations, self.setup_params.num_laser_ray)
        # Robot pose and goal pose
        self.robot_pose = np.zeros(3)
        self.randomGoalPose()
        self.current_vel = np.zeros(2)
        # Laser data
        self.laser_data = np.zeros(self.setup_params.num_laser_ray)
        # Initialize publishers and subscribers
        self.initializePublisherAndSubscriber()
        
    def timerCallback(self, event):
        '''
            Timer callback function for implement navigation
        '''
        # Set the current observation
        self.observation.setLaserObservation(self.laser_data)
        self.observation.setCurrentVelocityObservation(self.current_vel)
        self.observation.setGoalRelationObservation(self.robot_pose, self.goal_pose)
        
        reward = Float32()
        reward.data = self.reward.calculateReward(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel, self.setup_params.robot_radius)
        self.reward_pub.publish(reward)
        print(reward)
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

    def randomGoalPose(self):
        self.goal_pose = np.zeros(3)
        self.goal_pose[0] = random.uniform(-self.setup_params.map_width/2, self.setup_params.map_width/2)
        self.goal_pose[1] = random.uniform(-self.setup_params.map_length/2, self.setup_params.map_length/2)
        self.goal_pose[2] = random.uniform(-math.pi, math.pi)
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