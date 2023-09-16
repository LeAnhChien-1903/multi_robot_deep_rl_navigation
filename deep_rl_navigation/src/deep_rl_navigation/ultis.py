#!/usr/bin/env python

import numpy as np
import math

def normalize_angle(angle: float):
    '''
        Normalizes the angle to -pi to pi 
    '''
    return math.atan2(math.sin(angle), math.cos(angle))

def calculatedDistance(point1: np.ndarray, point2: np.ndarray):
    '''
        Calculates the Euclidean distance between two points
        ### Parameters
        - point1: the coordinate of first point
        - point2: the coordinate of second point
    '''
    return math.sqrt(np.sum(np.square(point1 - point2)))
    
class Setup:
    def __init__(self):
        # Time constraint
        self.sample_time: float = 0.1 # Sample time [s]
        # Robot constraints
        self.min_linear_velocity: float = 0.0 # Min linear velocity [m/s]
        self.max_linear_velocity: float = 0.0 # Max linear velocity [m/s]
        self.min_angular_velocity: float = 0.0 # Min angular velocity [rad/s]
        self.max_angular_velocity: float = 0.0 # Max angular velocity [rad/s]
        self.max_linear_acceleration: float = 0.0 # Max angular acceleration [m/s^2]
        self.max_angular_acceleration: float = 0.0 # Max angular acceleration [rad/s^2]
        self.robot_width: float = 0.0 # Width of robot [m]
        self.robot_length: float = 0.0 # Length of robot [m]
        self.robot_radius: float = 0.0 # Radius of robot [m]
        self.num_observations: int = 0 # Number of observations
        self.num_laser_ray: int = 0 # Number of layer ray
        self.goal_tolerance: float = 0.0 # Goal tolerance [m]
        # Lidar pose
        self.lidar_x: float = 0.0 # lidar position in x direction with base link [m]
        self.lidar_y: float = 0.0 # lidar position in y direction with base link [m]
        self.lidar_theta: float = 0.0 # lidar orientation in angle direction with base link [rad]
        # Map size
        self.map_length: float = 0.0 # Length of the map [m]
        self.map_width: float = 0.0 # Width of the map [m]
        # Topic name
        self.laser_scan_topic = "" # Laser scan topic name
        self.odometry_topic = "" # Odometry topic name
        self.cmd_vel_topic = "" # Command velocity topic name

class Observation:
    def __init__(self, num_observations: int, num_laser_ray: int):
        self.laser_data: np.ndarray = np.zeros((num_observations, num_laser_ray), dtype=float) # Matrix of laser data 
        self.goal_relation: np.ndarray = np.zeros((2, 1), dtype=float).transpose() # 2D vector representing the goal in polar coordinate (distance and angle) with respect to the robot’s current position.
        self.current_velocity: np.ndarray = np.zeros((2, 1), dtype=float).transpose() # he current linear and angular velocity of the differential-driven robot

    def setLaserObservation(self, laser_vec: np.ndarray):
        '''
            Set laser scan observation of the robot
            ### Parameters
            - laser_vec: Vector of the current laser data
        '''
        # Set the laser data
        for i in range(self.laser_data.shape[0] - 1, 1, -1):
            self.laser_data[[i, i - 1]] = self.laser_data[[i-1, i]]
        for i in range(laser_vec.size):
            self.laser_data[0, i] = laser_vec[i]
    def setGoalRelationObservation(self, current_pose: np.ndarray, goal_pose: np.ndarray):
        '''
            Set the goal relation observation of the robot
            ##### Parameters
            - current_pose: current robot pose [x, y, theta]
            - goal_pose: goal pose of robot [x, y, theta]
        '''
        self.goal_relation[0] = calculatedDistance(goal_pose[0:2], current_pose[0:2])
        self.goal_relation[1] = normalize_angle(goal_pose[2] - current_pose[2])
    
    def setCurrentVelocityObservation(self, linear_velocity: float, angular_velocity: float):
        '''
            Set the current velocity observation of the robot
            ### Parameters
            - linear_velocity: current linear velocity of the robot
            - angular_velocity: current angular velocity of the robot
        '''
        self.current_velocity[0] = linear_velocity
        self.current_velocity[1] = angular_velocity
    def setCurrentVelocityObservation(self, current_vel: np.ndarray):
        '''
            Set the current velocity observation of the robot
            ### Parameters
            - current_vel: the current velocity of the robot
        '''
        self.current_velocity = current_vel
class Reward:
    def __init__(self):
        self.r_arrival = 0.0 # reward when robot reach the goal
        self.r_collision = 0.0 # reward when robot colliding with obstacle
        self.omega_g = 0.0 # factor for goal reward
        self.omega_w = 0.0 # factor for rotational velocity
        self.large_angular_velocity = 0.0 # angular velocity for punish robot
        self.goal_tolerance = 0.0 # tolerance for goal reach
        self.prev_pose = np.zeros((3, 1))
    def calculateReward(self, observation: Observation, current:np.ndarray, goal: np.ndarray, min_distance: float, large_angular_vel: float):
        '''
            observation: observation of robot
            current: current pose of robot
            goal: goal pose of robot
            min_distance: minimum distance that robot collides with other objects
            large_angular_vel: large angular velocity to punish robot
        '''
        reward = 0.0 # reset reward value
        # Calculate reward relative to reaching the goal
        if observation.goal_relation[0] < self.goal_tolerance:
            reward += self.r_arrival
        else:
            # Calculate the difference distance between two pose
            diff_distance = calculatedDistance(self.prev_pose[0:2], goal[0:2]) - observation.goal_relation[0]
            reward += self.omega_g * diff_distance
        # Update previous robot pose
        self.prev_pose = current
        # Calculate reward relative with collision
        for i in range(observation.laser_data[0, :].size):
            if observation.laser_data[0, i] < min_distance:
                reward += self.r_collision
                break
        # Calculate reward relative with large rotational velocities
        if abs(observation.current_velocity[1]) > large_angular_vel:
            reward += self.omega_w * abs(observation.current_velocity[1])
            
        return reward

class PPO:
    '''
        Hyperparameters of PPO algorithm
    '''
    def __init__(self):
        self.lambda_: float = 0.0
        self.gamma: float = 0.0
        self.T_max: int = 0
        self.E_phi: int = 0
        self.beta: float = 0.0
        self.KL_target: float= 0.0
        self.xi: float = 0.0
        self.lr_theta_1st: float = 0.0
        self.lr_theta_2nd: float = 0.0
        self.E_v: int = 0
        self.lr_phi: float = 0.0
        self.beta_high: float = 0.0
        self.alpha: float = 0.0
        self.beta_low: float = 0.0

class Hybrid:
    '''
        Hyperparameters of hybrid control mode
    '''
    def __init__(self):
        self.r_safe: float = 0.0
        self.r_risk: float = 0.0
        self.p_scale: float = 0.0
        self.v_max: float = 0.0