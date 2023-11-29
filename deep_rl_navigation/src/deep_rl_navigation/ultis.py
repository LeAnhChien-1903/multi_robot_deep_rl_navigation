#!/usr/bin/env python

import rospy
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
    q:torch.Tensor = torch.zeros(q1.shape[0] * q2.shape[0])
    for i in range(p1.shape[0]):
        p[i*p1.shape[0]: i*p1.shape[0] + p2.shape[0]] = p1[i] * p2
        q[i*q1.shape[0]: i*q1.shape[0] + q2.shape[0]] = q1[i] * q2

    return (p * (p/q).log()).sum()
class Setup:
    def __init__(self):
        # Time constraint
        self.sample_time: float = 0.1 # Sample time [s]
        # Robot constraints
        self.min_linear_velocity: float = 0.0 # Min linear velocity [m/s]
        self.max_linear_velocity: float = 1.0 # Max linear velocity [m/s]
        self.min_angular_velocity: float = -1.0 # Min angular velocity [rad/s]
        self.max_angular_velocity: float = 1.0 # Max angular velocity [rad/s]
        self.max_linear_acceleration: float = 2.0 # Max angular acceleration [m/s^2]
        self.max_angular_acceleration: float = 2.0 # Max angular acceleration [rad/s^2]
        self.robot_width: float = 0.6 # Width of robot [m]
        self.robot_length: float = 0.9 # Length of robot [m]
        self.robot_radius: float = math.hypot(self.robot_width/2, self.robot_length/2) + 0.02 # Radius of robot [m]
        self.num_observations: int = 3 # Number of observations
        self.num_laser_ray: int = 541 # Number of layer ray
        self.goal_tolerance: float = 0.05 # Goal tolerance [m]
        # Lidar pose
        self.lidar_x: float = 0.0 # lidar position in x direction with base link [m]
        self.lidar_y: float = 0.0 # lidar position in y direction with base link [m]
        self.lidar_theta: float = 0.0 # lidar orientation in angle direction with base link [rad]
        # Map size
        self.map_length: float = 10.0 # Length of the map [m]
        self.map_width: float = 10.0 # Width of the map [m]
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
        self.goal_relation[1] = math.atan2(goal_pose[1] - current_pose[1], goal_pose[0] - current_pose[0])
    
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

class HyperParameters:
    def __init__(self):
        self.number_of_robot: int = 5
        self.number_of_action: int = 21
        self.mini_batch_size: int = 400
        self.batch_size: int = 4000
        self.setup = Setup() # setup parameters for navigation
        self.ppo = PPO() # ppo parameters for training
        self.hybrid = Hybrid() # hybrid parameters for hybrid control
        self.initialize()
    
    def initialize(self):
        '''
            Initialize the parameters from ros param
        '''
        # Setup parameters
        self.setup.sample_time = rospy.get_param("/sample_time")
        
        self.setup.laser_scan_topic = rospy.get_param("/laser_scan_topic")
        self.setup.odometry_topic = rospy.get_param("/odometry_topic")
        self.setup.cmd_vel_topic = rospy.get_param("/cmd_vel_topic")
        
        self.setup.min_linear_velocity = rospy.get_param("/min_linear_velocity")
        self.setup.max_linear_velocity = rospy.get_param("/max_linear_velocity")
        self.setup.min_angular_velocity = rospy.get_param("/min_angular_velocity")
        self.setup.max_angular_velocity = rospy.get_param("/max_angular_velocity")
        self.setup.max_linear_acceleration = rospy.get_param("/max_linear_acceleration")
        self.setup.max_angular_acceleration = rospy.get_param("/max_angular_acceleration")
        self.setup.num_observations = rospy.get_param("/num_observations")
        self.setup.num_laser_ray = rospy.get_param("/num_laser_ray")
        self.setup.goal_tolerance = rospy.get_param("/goal_tolerance")
        
        self.setup.robot_length = rospy.get_param("/robot_length")
        self.setup.robot_width = rospy.get_param("/robot_width")
        
        self.setup.lidar_x = rospy.get_param("/lidar_x")
        self.setup.lidar_y = rospy.get_param("/lidar_y")
        self.setup.lidar_theta = rospy.get_param("/lidar_theta")
        
        self.setup.map_length = rospy.get_param("/map_length")
        self.setup.map_width = rospy.get_param("/map_width")
        
        self.setup.robot_radius = math.hypot(self.setup.robot_width/2, self.setup.robot_length/2) + 0.02

        # PPO hyper parameters
        self.ppo.lambda_ = rospy.get_param("/ppo/lambda")
        self.ppo.gamma = rospy.get_param("/ppo/gamma")
        self.ppo.T_max = rospy.get_param("/ppo/T_max")
        self.ppo.E_phi = rospy.get_param("/ppo/E_phi")
        self.ppo.beta = rospy.get_param("/ppo/beta")
        self.ppo.KL_target = rospy.get_param("/ppo/KL_target")
        self.ppo.xi = rospy.get_param("/ppo/xi")
        self.ppo.lr_theta_1st = rospy.get_param("/ppo/lr_theta_1st")
        self.ppo.lr_theta_2nd = rospy.get_param("/ppo/lr_theta_2nd")
        self.ppo.E_v = rospy.get_param("/ppo/E_v")
        self.ppo.lr_phi = rospy.get_param("/ppo/lr_phi")
        self.ppo.beta_high = rospy.get_param("/ppo/beta_high")
        self.ppo.alpha = rospy.get_param("/ppo/alpha")
        self.ppo.beta_low = rospy.get_param("/ppo/beta_low")
        # Hybrid control hyper parameters
        self.hybrid.r_safe = rospy.get_param("/hybrid/r_safe")
        self.hybrid.r_risk = rospy.get_param("/hybrid/r_risk")
        self.hybrid.p_scale = rospy.get_param("/hybrid/p_scale")
        self.hybrid.v_max = rospy.get_param("/hybrid/v_max")
        
        self.number_of_robot = rospy.get_param("/number_of_robot")
        self.number_of_action = rospy.get_param("/number_of_action")
        
        self.mini_batch_size = math.ceil(self.ppo.T_max / self.number_of_robot)
        self.batch_size = self.mini_batch_size * self.number_of_robot
        
class SingleBuffer:
    def __init__(self, mini_batch_size: int, setup: Setup, num_of_actions: int):
        self.laser_obs_mini_batch = torch.zeros(mini_batch_size, 1, setup.num_observations, setup.num_laser_ray)
        self.goal_obs_mini_batch = torch.zeros((mini_batch_size, 2))
        self.vel_obs_mini_batch = torch.zeros((mini_batch_size, 2))
        self.linear_vel_mini_batch = torch.zeros((mini_batch_size, 1))
        self.angular_vel_mini_batch= torch.zeros((mini_batch_size, 1))
        self.linear_probs_mini_batch = torch.zeros((mini_batch_size, num_of_actions))
        self.angular_probs_mini_batch= torch.zeros((mini_batch_size, num_of_actions))
        self.done_mini_batch = torch.zeros(mini_batch_size)
        self.log_prob_mini_batch = torch.zeros(mini_batch_size)
        self.reward_mini_batch = torch.zeros(mini_batch_size)
        self.value_mini_batch = torch.zeros(mini_batch_size + 1)
        self.advantage_mini_batch = torch.zeros(mini_batch_size)
    
    def update(self, index: int, sample: dict)->None:
        self.value_mini_batch[index] = sample['value']
        if index > 0:
            self.reward_mini_batch[index-1] = sample['reward']
            self.done_mini_batch[index-1] = sample['done']
        if index < self.laser_obs_mini_batch.shape[0]:
            self.laser_obs_mini_batch[index] = sample['laser_obs']
            self.goal_obs_mini_batch[index] = sample['goal_obs']
            self.vel_obs_mini_batch[index] = sample['vel_obs']
            
            self.linear_vel_mini_batch[index] = sample['linear']
            self.angular_vel_mini_batch[index] = sample['angular']
            self.log_prob_mini_batch[index] = sample['log_prob']
            
            self.linear_probs_mini_batch[index] = sample['linear_probs']
            self.angular_probs_mini_batch[index] = sample['angular_probs']
    
    def advantageEstimator(self, gamma: float = 0.99, lambda_: float = 0.95):
        last_advantage = torch.zeros(1)
        last_value = self.value_mini_batch[-1]
        
        for t in reversed(range(self.advantage_mini_batch.shape[0])):
            mask = 1.0 - self.done_mini_batch[t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            
            delta = self.reward_mini_batch[t] + gamma * last_value - self.value_mini_batch[t]
            
            last_advantage = delta + gamma * lambda_ * last_advantage
            
            self.advantage_mini_batch[t] = last_advantage
            
            last_value = self.value_mini_batch[t]

class Buffer:
    def __init__(self, single_buffer_list: list):
        with torch.no_grad():
            self.laser_obs_batch = single_buffer_list[0].laser_obs_mini_batch
            self.goal_obs_batch = single_buffer_list[0].goal_obs_mini_batch
            self.vel_obs_batch = single_buffer_list[0].vel_obs_mini_batch
            self.linear_vel_batch = single_buffer_list[0].linear_vel_mini_batch
            self.linear_probs_batch = single_buffer_list[0].linear_probs_mini_batch
            self.angular_vel_batch = single_buffer_list[0].angular_vel_mini_batch
            self.angular_probs_batch = single_buffer_list[0].angular_probs_mini_batch
            self.log_prob_batch = single_buffer_list[0].log_prob_mini_batch
            self.value_batch = single_buffer_list[0].value_mini_batch[0:-1]
            self.done_batch = single_buffer_list[0].done_mini_batch
            self.reward_batch = single_buffer_list[0].reward_mini_batch
            self.advantage_batch = single_buffer_list[0].advantage_mini_batch

            for i in range(1, len(single_buffer_list)):
                self.laser_obs_batch = torch.concatenate((self.laser_obs_batch, single_buffer_list[i].laser_obs_mini_batch))
                self.goal_obs_batch = torch.concatenate((self.goal_obs_batch, single_buffer_list[i].goal_obs_mini_batch))
                self.vel_obs_batch = torch.concatenate((self.vel_obs_batch, single_buffer_list[i].vel_obs_mini_batch))
                self.linear_vel_batch = torch.concatenate((self.linear_vel_batch, single_buffer_list[i].linear_vel_mini_batch))
                self.linear_probs_batch = torch.concatenate((self.linear_probs_batch, single_buffer_list[i].linear_probs_mini_batch))
                self.angular_vel_batch = torch.concatenate((self.angular_vel_batch, single_buffer_list[i].angular_vel_mini_batch))
                self.angular_probs_batch = torch.concatenate((self.angular_probs_batch, single_buffer_list[i].angular_probs_mini_batch))
                self.log_prob_batch = torch.concatenate((self.log_prob_batch, single_buffer_list[i].log_prob_mini_batch))
                self.value_batch = torch.concatenate((self.value_batch, single_buffer_list[i].value_mini_batch[0:-1]))
                self.reward_batch = torch.concatenate((self.reward_batch, single_buffer_list[i].reward_mini_batch))
                self.done_batch = torch.concatenate((self.done_batch, single_buffer_list[i].done_mini_batch))
                self.advantage_batch = torch.concatenate((self.advantage_batch, single_buffer_list[i].advantage_mini_batch))


def AdvantageFunctionEstimation(reward_batch: torch.Tensor, value_batch: torch.Tensor, done_batch: torch.Tensor, 
                                gamma: float = 0.99, lambda_: float= 0.95):
    batch_size = reward_batch.shape[0]
    advantage_batch = torch.zeros(batch_size)
    
    last_advantage = torch.zeros(1)
    last_value = value_batch[-1]
    
    for t in reversed(range(batch_size)):
        mask = 1.0 - done_batch[t]
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        
        delta = reward_batch[t] + gamma * last_value - value_batch[t]
        
        last_advantage = delta + gamma * lambda_ * last_advantage
        
        advantage_batch[t] = last_advantage
        
        last_value = value_batch[t]
        
    
    return advantage_batch