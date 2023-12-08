#!/usr/bin/env python

import rospy
import numpy as np
import math
import torch
from deep_rl_navigation.actor_critic import *

def normalize_angle(angle: float):
    '''
        Normalizes the angle to -pi to pi 
    '''
    return math.atan2(math.sin(angle), math.cos(angle))

def find_difference_orientation(angle1: float, angle2: float):

    if 0 <= angle1 <= math.pi and 0 <= angle2 <= math.pi:
        return angle2 - angle1
    elif -math.pi < angle1 < 0 and -math.pi < angle2 < 0:
        return angle2 - angle1
    elif 0 <= angle1 <= math.pi and -math.pi < angle2 < 0:
        turn = angle2 - angle1
        if turn < -math.pi:
            turn += 2 * math.pi
        return turn
    elif -math.pi < angle1 < 0 and 0 <= angle2 <= math.pi:
        turn = angle2 - angle1
        if turn > math.pi:
            turn -= 2 * math.pi
        return turn

    return angle2 - angle1

def calculatedDistance(point1: np.ndarray, point2: np.ndarray):
    '''
        Calculates the Euclidean distance between two points
        ### Parameters
        - point1: the coordinate of first point
        - point2: the coordinate of second point
    '''
    return math.sqrt(np.sum(np.square(point1 - point2)))

def calculateCollisionSpace(width: float, length: float, angle_start: float, angle_end: float, angle_increment: float):
    angle_thresh = math.atan2(width/2, length/2)
    angle_range = np.arange(angle_start, angle_end, angle_increment)
    collision_space = np.zeros(angle_range.shape).astype(np.float32)
    for i in range(angle_range.shape[0]):
        if math.pi - angle_thresh <= abs(angle_range[i]) <= math.pi:
            collision_space[i] = (length + 0.1)/2 / math.cos(math.pi - abs(angle_range[i]))
        if math.pi/2 <= abs(angle_range[i]) < math.pi - angle_thresh:
            collision_space[i] = (width + 0.1)/2 / math.cos(abs(angle_range[i]) - math.pi/2)
        elif angle_thresh <= abs(angle_range[i]) <= math.pi/2:
            collision_space[i] = (width + 0.1)/2 / math.cos(math.pi/2 - abs(angle_range[i]))
        else:
            collision_space[i] = (length + 0.1)/2 / math.cos(abs(angle_range[i]))

    return collision_space
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
        self.angle_start: float = -3* math.pi/4 # Start angle
        self.angle_end: float = 3 * math.pi/4 # End angle
        self.angle_increment: float = math.pi/360 # Increment angle
        self.num_observations: int = 3 # Number of observations
        self.num_laser_ray: int = 541 # Number of layer ray
        self.goal_tolerance: float = 0.02 # Goal tolerance [m]
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
        self.position_topic = "" 
        self.velocity_topic = "" 

class Observation:
    def __init__(self, num_observations: int = 3, num_laser_ray: int = 541):
        self.laser_data: np.ndarray = np.zeros((num_observations, num_laser_ray), np.float32) # Matrix of laser data 
        self.goal_data: np.ndarray = np.zeros(2, np.float32) # 2D vector representing the goal in polar coordinate (distance and angle) with respect to the robot’s current position.
        self.vel_data: np.ndarray = np.zeros(2, np.float32) # the current linear and angular velocity of the differential-driven robot
    def setObservation(self, laser_vec: np.ndarray, current_pose: np.ndarray, goal_pose: np.ndarray, linear_vel: float, angular_vel: float):
        # Set the laser data
        flag = False
        for i in range(self.laser_data.shape[0]):
            if self.laser_data[i].nonzero()[0].shape[0] == 0:
                flag = True
                break
        if flag == False:
            self.laser_data = np.roll(self.laser_data, shift=-1, axis=0)
            self.laser_data[-1] = laser_vec.copy()
        else:
            for i in range(self.laser_data.shape[0]):
                self.laser_data[i] = laser_vec.copy()
        
        mean = self.laser_data.mean()
        std = self.laser_data.std()
        laser_data = (self.laser_data - mean)/ std 
        # Set the goal data
        self.goal_data[0] = calculatedDistance(current_pose[0:-1], goal_pose[0:-1]) / 30
        self.goal_data[1] = math.atan2(goal_pose[1] - current_pose[1], goal_pose[0] - current_pose[0])/ math.pi
        # Set the current velocity
        self.vel_data[0] = linear_vel
        self.vel_data[1] = angular_vel

        return laser_data, self.goal_data, self.vel_data

class RewardParams:
    '''
        Hyperparameter of reward
    '''
    def __init__(self):
        self.r_arrival: float = 15.0
        self.r_collision: float = -15.0
        self.omega_g: float = 2.5
        self.omega_w: float = -0.1
        self.large_angular_velocity: float = 0.7

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
        self.kl_target: float= 0.0015
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
        self.number_of_robot: int = 8
        self.number_of_action: int = 21
        self.mini_batch_size: int = 400
        self.batch_size: int = 3200
        self.setup = Setup() # setup parameters for navigation
        self.ppo = PPO() # ppo parameters for training
        self.hybrid = Hybrid() # hybrid parameters for hybrid control
        self.reward = RewardParams() # reward parameters for compute reward
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
        self.setup.position_topic = rospy.get_param("/position_topic")
        self.setup.velocity_topic = rospy.get_param("/velocity_topic")
        
        self.setup.min_linear_velocity = rospy.get_param("/min_linear_velocity")
        self.setup.max_linear_velocity = rospy.get_param("/max_linear_velocity")
        self.setup.min_angular_velocity = rospy.get_param("/min_angular_velocity")
        self.setup.max_angular_velocity = rospy.get_param("/max_angular_velocity")
        self.setup.max_linear_acceleration = rospy.get_param("/max_linear_acceleration")
        self.setup.max_angular_acceleration = rospy.get_param("/max_angular_acceleration")
        self.setup.num_observations = rospy.get_param("/num_observations")
        self.setup.num_laser_ray = rospy.get_param("/num_laser_ray")
        self.setup.goal_tolerance = rospy.get_param("/goal_tolerance")
        self.setup.angle_start = rospy.get_param("/angle_start")
        self.setup.angle_end = rospy.get_param("/angle_end")
        self.setup.angle_increment = rospy.get_param("/angle_increment")
        
        self.setup.robot_length = rospy.get_param("/robot_length")
        self.setup.robot_width = rospy.get_param("/robot_width")
        
        self.setup.lidar_x = rospy.get_param("/lidar_x")
        self.setup.lidar_y = rospy.get_param("/lidar_y")
        self.setup.lidar_theta = rospy.get_param("/lidar_theta")
        
        self.setup.map_length = rospy.get_param("/map_length")
        self.setup.map_width = rospy.get_param("/map_width")
        # Reward hyperparameters
        self.reward.r_arrival = rospy.get_param("/reward/r_arrival")
        self.reward.r_collision = rospy.get_param("/reward/r_collision")
        self.reward.omega_g = rospy.get_param("/reward/omega_g")
        self.reward.omega_w = rospy.get_param("/reward/omega_w")
        self.reward.large_angular_velocity = rospy.get_param("/reward/large_angular_velocity")
        # PPO hyper parameters
        self.ppo.lambda_ = rospy.get_param("/ppo/lambda")
        self.ppo.gamma = rospy.get_param("/ppo/gamma")
        self.ppo.T_max = rospy.get_param("/ppo/T_max")
        self.ppo.E_phi = rospy.get_param("/ppo/E_phi")
        self.ppo.beta = rospy.get_param("/ppo/beta")
        self.ppo.kl_target = rospy.get_param("/ppo/kl_target")
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
        
        self.mini_batch_size = math.ceil(self.ppo.T_max / self.number_of_robot)
        self.batch_size = self.mini_batch_size * self.number_of_robot

class Reward:
    def __init__(self, reward_params: RewardParams, setup_params: Setup):
        self.params = reward_params
        self.goal_tolerance = setup_params.goal_tolerance
        self.collision_space = calculateCollisionSpace(setup_params.robot_width, setup_params.robot_length,
                                                        setup_params.angle_start, setup_params.angle_end, 
                                                        setup_params.angle_increment)
        self.prev_pose = np.zeros(3, dtype=np.float32)
    def calculateReward(self, laser_data:np.ndarray, current:np.ndarray, goal: np.ndarray, current_vel: np.ndarray):
        '''
            observation: observation of robot
            current: current pose of robot
            goal: goal pose of robot
            r_collision: minimum distance that robot collides with other objects
            large_angular_vel: large angular velocity to punish robot
        '''
        r_g = r_c = r_w = 0.0 
        if (self.prev_pose[0] == 0.0 and self.prev_pose[1] == 0.0 and self.prev_pose[2] == 0.0):
            self.prev_pose = current.copy()
            return 0.0
        
        curr_dist_to_goal = calculatedDistance(current[0:-1], goal[0:-1])
        prev_dist_to_goal = calculatedDistance(self.prev_pose[0:-1], goal[0:-1])      
        
        if curr_dist_to_goal < self.goal_tolerance:
            r_g = self.params.r_arrival
        else:
            r_g = self.params.omega_g * (prev_dist_to_goal - curr_dist_to_goal)
            
        if (laser_data <= self.collision_space).any() == True:
            r_c = self.params.r_collision
            
        if abs(current_vel[1]) > self.params.large_angular_velocity:
            r_w = self.params.omega_w * abs(current_vel[1])

        self.prev_pose = current.copy()
        return r_g + r_c + r_w

class SingleBuffer:
    def __init__(self, mini_batch_size: int, hyper_params: HyperParameters, device):
        self.device = device
        self.laser_obs_mini_batch = torch.zeros(mini_batch_size, hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray).to(device)
        self.goal_obs_mini_batch = torch.zeros(mini_batch_size, 2).to(device)
        self.vel_obs_mini_batch = torch.zeros(mini_batch_size, 2).to(device)
        self.linear_mini_batch = torch.zeros(mini_batch_size).to(device)
        self.angular_mini_batch = torch.zeros(mini_batch_size).to(device)
        self.vel_probs_mini_batch = torch.zeros(mini_batch_size, hyper_params.number_of_action**2)
        self.done_mini_batch = torch.zeros(mini_batch_size).to(device)
        self.log_prob_mini_batch = torch.zeros(mini_batch_size).to(device)
        self.reward_mini_batch = torch.zeros(mini_batch_size).to(device)
        self.value_mini_batch = torch.zeros(mini_batch_size + 1).to(device)
        self.advantage_mini_batch = torch.zeros(mini_batch_size).to(device)
    
    def update(self, index: int, sample: dict)->None:
        self.value_mini_batch[index] = sample['value']
        if index > 0:
            self.reward_mini_batch[index-1] = sample['reward']
            self.done_mini_batch[index-1] = sample['done']
        if index < self.laser_obs_mini_batch.shape[0]:
            self.linear_mini_batch[index] = sample['linear']
            self.angular_mini_batch[index] = sample['angular']
            self.laser_obs_mini_batch[index] = sample['laser_obs']
            self.goal_obs_mini_batch[index] = sample['goal_obs']
            self.vel_obs_mini_batch[index] = sample['vel_obs']
            
            self.vel_probs_mini_batch[index] = sample['vel_probs']
            self.log_prob_mini_batch[index] = sample['log_prob']
    
    def advantageEstimator(self, gamma: float = 0.99, lambda_: float = 0.95):
        last_advantage = torch.zeros(1).to(self.device)
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
    def __init__(self, single_buffer_list: list, device):
        with torch.no_grad():
            self.laser_obs_batch = single_buffer_list[0].laser_obs_mini_batch
            self.goal_obs_batch = single_buffer_list[0].goal_obs_mini_batch
            self.vel_obs_batch = single_buffer_list[0].vel_obs_mini_batch
            self.linear_batch = single_buffer_list[0].linear_mini_batch
            self.angular_batch = single_buffer_list[0].angular_mini_batch
            self.vel_probs_batch = single_buffer_list[0].vel_probs_mini_batch
            self.log_prob_batch = single_buffer_list[0].log_prob_mini_batch
            self.value_batch = single_buffer_list[0].value_mini_batch[0:-1]
            self.reward_batch = single_buffer_list[0].reward_mini_batch
            self.done_batch = single_buffer_list[0].done_mini_batch
            self.advantage_batch = single_buffer_list[0].advantage_mini_batch

            for i in range(1, len(single_buffer_list)):
                self.laser_obs_batch = torch.concatenate((self.laser_obs_batch, single_buffer_list[i].laser_obs_mini_batch))
                self.goal_obs_batch = torch.concatenate((self.goal_obs_batch, single_buffer_list[i].goal_obs_mini_batch))
                self.vel_obs_batch = torch.concatenate((self.vel_obs_batch, single_buffer_list[i].vel_obs_mini_batch))
                self.linear_batch = torch.concatenate((self.linear_batch, single_buffer_list[i].linear_mini_batch))
                self.angular_batch = torch.concatenate((self.angular_batch, single_buffer_list[i].angular_mini_batch))
                self.vel_probs_batch = torch.concatenate((self.vel_probs_batch, single_buffer_list[i].vel_probs_mini_batch))
                self.log_prob_batch = torch.concatenate((self.log_prob_batch, single_buffer_list[i].log_prob_mini_batch))
                self.value_batch = torch.concatenate((self.value_batch, single_buffer_list[i].value_mini_batch[0:-1]))
                self.reward_batch = torch.concatenate((self.reward_batch, single_buffer_list[i].reward_mini_batch))
                self.done_batch = torch.concatenate((self.done_batch, single_buffer_list[i].done_mini_batch))
                self.advantage_batch = torch.concatenate((self.advantage_batch, single_buffer_list[i].advantage_mini_batch))

            self.laser_obs_batch.to(device)
            self.goal_obs_batch.to(device)
            self.vel_obs_batch.to(device)
            self.linear_batch.to(device)
            self.angular_batch.to(device)
            self.vel_probs_batch.to(device)
            self.log_prob_batch.to(device)
            self.value_batch.to(device)
            self.reward_batch.to(device)
            self.done_batch.to(device)
            self.advantage_batch.to(device)
            
def calculatePolicyLoss(buffer: Buffer, actor: Actor, ppo: PPO,  device):
    new_log_prob_batch, new_vel_probs_batch = actor.evaluate(buffer.laser_obs_batch, 
                                                            buffer.goal_obs_batch, 
                                                            buffer.vel_obs_batch,
                                                            buffer.linear_batch,
                                                            buffer.angular_batch)
    
    log_ratio = new_log_prob_batch - buffer.log_prob_batch
    ratio = log_ratio.exp()
    
    old_vel_probs_batch = buffer.vel_probs_batch.to(device)
    kl_div = ((old_vel_probs_batch * (old_vel_probs_batch / new_vel_probs_batch).log()).sum(dim = 1)).mean()
    kl_divergence_final = kl_div.item()
    
    loss1 = - buffer.advantage_batch * ratio
    loss2 = ppo.beta * kl_div
    loss3 = -  ppo.xi * torch.pow(torch.max(torch.zeros(1).to(device), kl_div - 2 * ppo.kl_target), 2)
    loss = loss1 + loss2 + loss3
    actor_loss = loss.sum()
    
    return actor_loss, kl_divergence_final

def calculateSingleValueLoss(buffer: SingleBuffer, critic: Critic, gamma:float, device):
    mini_batch_size = buffer.reward_mini_batch.shape[0]
    new_value = critic.get_value(buffer.laser_obs_mini_batch, 
                                buffer.goal_obs_mini_batch, 
                                buffer.vel_obs_mini_batch)
    
    value_loss = torch.zeros(mini_batch_size).to(device)
    last_reward = 0.0
    
    for t in reversed(range(mini_batch_size)):
        value_loss[t] = (last_reward - new_value[t])**2             
        last_reward = gamma * (buffer.reward_mini_batch[t] + last_reward)
    return value_loss.sum()

def calculateValueLoss(buffer_list: list, critic: Critic, gamma: float, device):
    value_loss = 0.0
    num_of_robots = len(buffer_list)
    
    for i in range(num_of_robots):
        value_loss += calculateSingleValueLoss(buffer_list[i], critic, gamma, device)
    
    return value_loss