#!/usr/bin/env python

import numpy as np
import math
import torch

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

def calculateMultiKLDivergence(p1: torch.Tensor, p2: torch.Tensor, q1: torch.Tensor, q2: torch.Tensor)-> torch.Tensor:
    # Calculate the probability list
    p:torch.Tensor = torch.zeros(p1.shape[0] * p2.shape[0])
    q:torch.Tensor = torch.zeros(q2.shape[0] * q1.shape[0])
    counter = 0
    for i in range(p1.shape[0]):
        for j in range(p2.shape[0]):
            p[counter] = p1[i] * p2[j]
            q[counter] = q1[i] * p2[j]
            counter += 1

    return torch.sum(p * (p/q).log())
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
        self.laser_data: np.ndarray = np.zeros((num_observations, num_laser_ray), np.float32) # Matrix of laser data 
        self.goal_relation: np.ndarray = np.zeros(2, np.float32) # 2D vector representing the goal in polar coordinate (distance and angle) with respect to the robot’s current position.
        self.current_velocity: np.ndarray = np.zeros(2, np.float32) # he current linear and angular velocity of the differential-driven robot

    def setLaserObservation(self, laser_vec: np.ndarray):
        '''
            Set laser scan observation of the robot
            ### Parameters
            - laser_vec: Vector of the current laser data
        '''
        # Set the laser data
        for i in range(self.laser_data.shape[0] - 1, 0, -1):
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
        self.current_velocity = current_vel.copy()
    def setObservation(self, laser_vec: np.ndarray, current_pose: np.ndarray, goal_pose: np.ndarray, current_vel: np.ndarray):
        self.setLaserObservation(laser_vec)
        self.setGoalRelationObservation(current_pose, goal_pose)
        self.setCurrentVelocityObservation(current_vel)
        self.normalizeObservation()
    
    def normalizeObservation(self):
        observation = np.concatenate((self.laser_data.flatten(), self.goal_relation))
        mean = observation.mean()
        std_deviation = observation.std()
        
        self.laser_data = (self.laser_data - mean) / std_deviation
        self.goal_relation = (self.goal_relation - mean) / std_deviation
        # velocity_data = (velocity_data - mean) / std_deviation

class Reward:
    def __init__(self):
        self.r_arrival: float = 15.0 # reward when robot reach the goal
        self.r_collision: float = -15.0 # reward when robot colliding with obstacle
        self.omega_g: float = 2.5 # factor for goal reward
        self.omega_w: float = - 0.1 # factor for rotational velocity
        self.large_angular_velocity: float = 0.7 # angular velocity for punish robot
        self.goal_tolerance: float = 0.1 # tolerance for goal reach
        self.prev_pose = np.zeros(3, dtype=np.float32)
    def calculateReward(self, laser_data:np.ndarray, current:np.ndarray, goal: np.ndarray, current_vel: np.ndarray, min_distance: float):
        '''
            observation: observation of robot
            current: current pose of robot
            goal: goal pose of robot
            min_distance: minimum distance that robot collides with other objects
            large_angular_vel: large angular velocity to punish robot
        '''
        r_g = r_c = r_w = 0.0 
        if (self.prev_pose[0] == 0.0 and self.prev_pose[1] == 0.0 and self.prev_pose[2] == 0.0):
            self.prev_pose = current.copy()
            return r_g + r_c + r_w
        else:
            # Calculate reward relative to reaching the goal
            distance_to_goal = calculatedDistance(current[0:2], goal[0:2])

            if distance_to_goal < self.goal_tolerance:
                r_g = self.r_arrival
            else:
                # Calculate the difference distance between two pose
                diff_distance = calculatedDistance(self.prev_pose[0:2], goal[0:2]) - distance_to_goal
                r_g = self.omega_g * diff_distance
            # Calculate reward relative with collision
            if min(laser_data) < min_distance:
                r_c = self.r_collision
            # Calculate reward relative with large rotational velocities
            if abs(current_vel[1]) > self.large_angular_velocity:
                r_w = self.omega_w * abs(current_vel[1])
            # Update previous robot pose
            self.prev_pose = current.copy()
        return r_g + r_c + r_w

class PPO:
    '''
        Hyperparameters of PPO algorithm
    '''
    def __init__(self):
        self.lambda_: float = 0.95
        self.gamma: float = 0.99
        self.T_max: int = 1000
        self.E_phi: int = 20
        self.beta: float = 1.0
        self.KL_target: float= 0.0015
        self.xi: float = 50.0
        self.lr_theta_1st: float = 0.00005
        self.lr_theta_2nd: float = 0.00002
        self.E_v: int = 10
        self.lr_phi: float = 0.001
        self.beta_high: float = 2.0
        self.alpha: float = 1.5
        self.beta_low: float = 0.5

class Hybrid:
    '''
        Hyperparameters of hybrid control mode
    '''
    def __init__(self):
        self.r_safe: float = 0.0
        self.r_risk: float = 0.0
        self.p_scale: float = 0.0
        self.v_max: float = 0.0