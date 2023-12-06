#!/usr/bin/env python

import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler

import numpy as np
import math
import os

import torch
import random

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import MarkerArray, Marker

from deep_rl_navigation.ultis import *
from deep_rl_navigation.actor_critic import *

class Agent:
    def __init__(self, name: str, hyper_param: HyperParameters):
        self.robot_name = name
        self.setup: Setup = hyper_param.setup
        self.reward: Reward = Reward(hyper_param.reward, hyper_param.setup)
        # Initialize observation
        self.observation = Observation(hyper_param.setup.num_observations, hyper_param.setup.num_laser_ray)
        # Robot pose and goal pose
        self.robot_pose = np.zeros(3, dtype=np.float32)
        self.setPoseInMapPlus()
        # Random color
        self.color = np.array([random.uniform(0.0, 1.0), random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)])
        self.setColorInMapPlus()
        # self.randomPose()
        self.current_vel = np.zeros(2, dtype=np.float32)
        # Laser data
        self.laser_data = np.zeros(hyper_param.setup.num_laser_ray, dtype=np.float32)
        # Subscriber
        self.odometry_sub = rospy.Subscriber(self.robot_name + "/" + hyper_param.setup.odometry_topic, Odometry, self.odometryCallback, queue_size= 10)
        self.laser_scan_sub = rospy.Subscriber(self.robot_name + "/" + hyper_param.setup.laser_scan_topic, LaserScan, self.getLaserData, queue_size= 10)
        self.cmd_vel_pub = rospy.Publisher(self.robot_name + "/" + hyper_param.setup.cmd_vel_topic, Twist, queue_size= 10)
        self.markers_pub = rospy.Publisher(self.robot_name + "/robot_visualization", MarkerArray, queue_size= 10)
        
        self.path_marker = Marker()
        self.path_marker.header.stamp = rospy.Time.now()
        self.path_marker.header.frame_id = "map"
        self.path_marker.ns = "robot_path"
        self.path_marker.action = self.path_marker.ADD
        self.path_marker.type = self.path_marker.LINE_STRIP
        
        self.path_marker.pose.orientation.x = 0.0
        self.path_marker.pose.orientation.y = 0.0
        self.path_marker.pose.orientation.z = 0.0
        self.path_marker.pose.orientation.w = 1.0
    
        self.path_marker.scale.x = 0.05
        self.path_marker.scale.y = 0.05
        self.path_marker.scale.z = 0.05
        
        self.path_marker.color.r = self.color[0]
        self.path_marker.color.g = self.color[1]
        self.path_marker.color.b = self.color[2]
        self.path_marker.color.a = 1.0

    def step(self, actor: Actor, critic: Critic, device):
        '''
            Timer callback function for implement navigation
        '''
        # Get done
        self.robot_visualization()
        done = self.goalReached()
        
        # Get the current state
        laser_data, goal_data, vel_data = self.observation.setObservation(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel[0], self.current_vel[1])
        # Calculate the reward at state
        reward = self.reward.calculateReward(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel)
        laser_obs = torch.from_numpy(laser_data.reshape((1, self.setup.num_observations, self.setup.num_laser_ray))).to(device)
        goal_obs = torch.from_numpy(goal_data.reshape(1, 2)).to(device)
        vel_obs = torch.from_numpy(vel_data.reshape(1, 2)).to(device)
        
        # Get action and value function
        linear_vel, angular_vel, log_prob, vel_probs = actor.get_action(laser_obs, goal_obs, vel_obs)
        value = critic.get_value(laser_obs, goal_obs, vel_obs)
        
        sample: dict = {'laser_obs': laser_obs, 'goal_obs': goal_obs, 'vel_obs': vel_obs,
                        'linear': linear_vel, 'angular': angular_vel, 'log_prob': log_prob, 
                        'vel_probs': vel_probs, 'value': value,'reward': reward, 'done': done}
        return sample
    
    def run_policy(self, actor: Actor, device):
        self.robot_visualization()
        done = self.goalReached()
        
        # Get the current state
        laser_data, goal_data, vel_data = self.observation.setObservation(self.laser_data, self.robot_pose, self.goal_pose, self.current_vel[0], self.current_vel[1])
        # Calculate the reward at state
        reward = self.reward.calculateReward(self.laser_data, self.robot_pose, self.goal_pose)
        laser_obs = torch.from_numpy(laser_data.reshape((1, self.setup.num_observations, self.setup.num_laser_ray))).to(device)
        goal_obs = torch.from_numpy(goal_data.reshape(1, 2)).to(device)
        vel_obs = torch.from_numpy(vel_data.reshape(1, 2)).to(device)
        
        # Get action and value function
        linear_vel, angular_vel = actor.exploit_policy(laser_obs, goal_obs, vel_obs)
        
        return linear_vel, angular_vel, reward, done

    def robot_visualization(self):
        visual_markers = MarkerArray()
        robot_marker = Marker()
        robot_marker.header.stamp = rospy.Time.now()
        robot_marker.header.frame_id = "map"
        robot_marker.ns = "robot_position"
        robot_marker.action = robot_marker.ADD
        robot_marker.type = robot_marker.CUBE
        
        robot_marker.pose.position.x = self.robot_pose[0]
        robot_marker.pose.position.y = self.robot_pose[1]
        robot_marker.pose.position.z = 0.2
        
        q = quaternion_from_euler(0, 0, self.robot_pose[2])
        robot_marker.pose.orientation.x = q[0]
        robot_marker.pose.orientation.y = q[1]
        robot_marker.pose.orientation.z = q[2]
        robot_marker.pose.orientation.w = q[3]
        
        robot_marker.scale.x = 0.9
        robot_marker.scale.y = 0.6
        robot_marker.scale.z = 0.4
        
        robot_marker.color.r = self.color[0]
        robot_marker.color.g = self.color[1]
        robot_marker.color.b = self.color[2]
        robot_marker.color.a = 1.0
        
        visual_markers.markers.append(robot_marker)
        
        goal_marker = Marker()
        goal_marker.header.stamp = rospy.Time.now()
        goal_marker.header.frame_id = "map"
        goal_marker.ns = "robot_goal"
        goal_marker.action = goal_marker.ADD
        goal_marker.type = goal_marker.CUBE
        
        goal_marker.pose.position.x = self.goal_pose[0]
        goal_marker.pose.position.y = self.goal_pose[1]
        goal_marker.pose.position.z = 0.2
        
        q = quaternion_from_euler(0, 0, self.goal_pose[2])
        goal_marker.pose.orientation.x = q[0]
        goal_marker.pose.orientation.y = q[1]
        goal_marker.pose.orientation.z = q[2]
        goal_marker.pose.orientation.w = q[3]
        
        goal_marker.scale.x = 0.9
        goal_marker.scale.y = 0.6
        goal_marker.scale.z = 0.4
        
        goal_marker.color.r = self.color[0]
        goal_marker.color.g = self.color[1]
        goal_marker.color.b = self.color[2]
        goal_marker.color.a = 1.0
        
        visual_markers.markers.append(goal_marker)

        p = Point()
        p.x = self.robot_pose[0]
        p.y = self.robot_pose[1]
        self.path_marker.points.append(p)
        visual_markers.markers.append(self.path_marker)
        
        self.markers_pub.publish(visual_markers)
        
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
        extra_theta = normalize_angle(self.robot_pose[2] + self.setup.lidar_theta)
        for i in range(len(laser_scan.ranges)):
            angle = normalize_angle(laser_scan.angle_min + i * laser_scan.angle_increment + extra_theta)
            if laser_scan.ranges[i] > laser_scan.range_max:
                range_x = laser_scan.range_max * math.cos(angle) + self.setup.lidar_x
                range_y = laser_scan.range_max * math.sin(angle) + self.setup.lidar_y
                self.laser_data[i] = math.hypot(range_x, range_y)
            elif (laser_scan.ranges[i] < laser_scan.range_min):
                range_x = math.cos(angle) + self.setup.lidar_x
                range_y = math.sin(angle) + self.setup.lidar_y
                self.laser_data[i] = math.hypot(range_x, range_y)
            else:
                range_x = laser_scan.ranges[i] * math.cos(angle) + self.setup.lidar_x
                range_y = laser_scan.ranges[i] * math.sin(angle) + self.setup.lidar_y
                self.laser_data[i] = math.hypot(range_x, range_y)
    
    def goalReached(self):
        distance = calculatedDistance(self.robot_pose[0:2], self.goal_pose[0:2])
        if distance < self.setup.goal_tolerance:
            return True
        return False
    
    def randomPose(self):        
        self.goal_pose = np.array([random.uniform(-self.setup.map_width/2 + 1.0, self.setup.map_width/2 - 1.0), random.uniform(-self.setup.map_length/2 + 1.0, self.setup.map_length/2 - 1.0), random.uniform(-math.pi, math.pi)])
    
    def setPoseInMapPlus(self):
        id = int(self.robot_name.split("_")[1])
        if id == 0:
            self.goal_pose = np.array([-13.0, 0.0, math.pi])
        elif id == 1:
            self.goal_pose = np.array([-12.36, -4.02, -0.9*math.pi])
        elif id == 2:
            self.goal_pose = np.array([-10.52, -7.64, -0.8*math.pi])
        elif id == 3:
            self.goal_pose = np.array([-7.64, -10.52, -0.7*math.pi])
        elif id == 4:
            self.goal_pose = np.array([-4.02, -12.36, -0.6*math.pi])
        elif id == 5:
            self.goal_pose = np.array([0.0, -13.0, -0.5*math.pi])
        elif id == 6:
            self.goal_pose = np.array([4.02, -12.36, -0.4*math.pi])
        elif id == 7:
            self.goal_pose = np.array([7.64, -10.52, -0.3*math.pi])
        elif id == 8:
            self.goal_pose = np.array([10.52, -7.64, -0.2*math.pi])
        elif id == 9:
            self.goal_pose = np.array([12.36, -4.02, -0.1*math.pi])
        elif id == 10:
            self.goal_pose = np.array([13.0, 0.0, 0.0])
        elif id == 11:
            self.goal_pose = np.array([12.36, 4.02, 0.1*math.pi])
        elif id == 12:
            self.goal_pose = np.array([10.52, 7.64, 0.2*math.pi])
        elif id == 13:
            self.goal_pose = np.array([7.64, 10.52, 0.3*math.pi])
        elif id == 14:
            self.goal_pose = np.array([4.02, 12.36, 0.4*math.pi])
        elif id == 15:
            self.goal_pose = np.array([0.0, 13.0, 0.5*math.pi])
        elif id == 16:
            self.goal_pose = np.array([-4.02, 12.36, 0.6*math.pi])
        elif id == 17:
            self.goal_pose = np.array([-7.64, 10.52, 0.7*math.pi])
        elif id == 18:
            self.goal_pose = np.array([-10.52, 7.64, 0.8*math.pi])
        elif id == 19:
            self.goal_pose = np.array([-12.36, 4.02, 0.9*math.pi])
    def setColorInMapPlus(self):
        id = int(self.robot_name.split("_")[1])
        if id == 0:
            self.color = np.array([1.0, 0.0, 0.0])
        elif id == 1:
            self.color = np.array([0.0, 1.0, 0.0])
        elif id == 2:
            self.color = np.array([0.0, 0.0, 1.0])
        elif id == 3:
            self.color = np.array([1.0, 1.0, 0.0])
        elif id == 4:
            self.color = np.array([1.0, 0.0, 1.0])
        elif id == 5:
            self.color = np.array([0.0, 1.0, 1.0])
        elif id == 6:
            self.color = np.array([0.0, 0.0, 0.0])
        elif id == 7:
            self.color = np.array([0.5, 0.0, 0.0])
        elif id == 8:
            self.color = np.array([0.0, 0.5, 0.0])
        elif id == 9:
            self.color = np.array([0.0, 0.0, 0.5])
        elif id == 10:
            self.color = np.array([0.5, 0.5, 0.0])
        elif id == 11:
            self.color = np.array([0.5, 0.0, 0.5])
        elif id == 12:
            self.color = np.array([0.0, 0.5, 0.5])
        elif id == 13:
            self.color = np.array([0.5, 0.5, 0.5])
        elif id == 14:
            self.color = np.array([0.8, 0.0, 0.0])
        elif id == 15:
            self.color = np.array([0.0, 0.8, 0.0])
        elif id == 16:
            self.color = np.array([0.0, 0.0, 0.8])
        elif id == 17:
            self.color = np.array([0.8, 0.8, 0.0])
        elif id == 18:
            self.color = np.array([0.8, 0.0, 0.8])
        elif id == 19:
            self.color = np.array([0.0, 0.8, 0.8])