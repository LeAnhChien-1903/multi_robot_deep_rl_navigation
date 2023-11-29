#!/usr/bin/env python

import torch
import torch.nn as neuron_network
from torch.distributions import Categorical
class ActorDiscrete(neuron_network.Module):
    def __init__(self, num_input_channels = 3, num_of_actions = 20):
        super(ActorDiscrete, self).__init__()
        # Actor network architecture
        self.convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= num_input_channels, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Flatten(),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.ReLU()
        )
        self.fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.ReLU(),
        )
        self.linear_vel_network = neuron_network.Sequential(
            neuron_network.Linear(in_features= 128, out_features= num_of_actions),
            neuron_network.Softmax(-1)
        )
        self.angular_vel_network = neuron_network.Sequential(
            neuron_network.Linear(in_features= 128, out_features= num_of_actions),
            neuron_network.Softmax(-1)
        )
    
    def get_action(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        # Get output from convolution and fully-connected
        conv_out = self.convolution(laser_obs).reshape(256)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs))
        fully_out = self.fully_connected(fully_in)
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.linear_vel_network(fully_out)
        angular_probs: torch.Tensor = self.angular_vel_network(fully_out)
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        # Get action
        linear_vel = linear_distribution.sample()
        angular_vel = angular_distribution.sample()
        # Get action log prob
        action_log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        return linear_vel, angular_vel, action_log_prob, linear_probs, angular_probs
    
    def exploit_policy(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        # Get output from convolution and fully-connected
        conv_out = self.convolution(laser_obs).reshape(256)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs))
        fully_out = self.fully_connected(fully_in)
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.linear_vel_network(fully_out)
        angular_probs: torch.Tensor = self.angular_vel_network(fully_out)
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        # Get action
        linear_vel = linear_probs.argmax(-1)
        angular_vel = angular_probs.argmax(-1)
        log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        return linear_vel, angular_vel, log_prob

    def evaluate(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, linear_vel: torch.Tensor, angular_vel: torch.Tensor):
        # Get output from convolution and fully-connected
        laser_obs_shape = laser_obs.shape
        goal_obs_shape = goal_obs.shape
        vel_obs_shape = vel_obs.shape
        linear_vel_shape = linear_vel.shape
        angular_vel_shape = angular_vel.shape

        if len(laser_obs_shape) == 2: 
            laser_obs = laser_obs[None, ...]

        if len(goal_obs_shape) == 1: 
            goal_obs = goal_obs[None, ...]

        if len(vel_obs_shape) == 1: 
            vel_obs = vel_obs[None, ...]

        if len(linear_vel_shape) == 0: 
            linear_vel = linear_vel[None, ...]

        if len(angular_vel_shape) == 0: 
            angular_vel = angular_vel[None, ...]

        batch_size = laser_obs.shape[0]

        conv_out = self.convolution(laser_obs).reshape(batch_size, 256)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs), dim = 1)
        fully_out = self.fully_connected(fully_in)
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.linear_vel_network(fully_out)
        angular_probs: torch.Tensor = self.angular_vel_network(fully_out)
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)

        action_log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        if len(angular_vel_shape) == 0: 
            return action_log_prob[0], linear_probs[0], angular_probs[0]

        return action_log_prob, linear_probs, angular_probs
            
class CriticDiscrete(neuron_network.Module):
    def __init__(self, num_input_channels = 3):
        super(CriticDiscrete, self).__init__()
        # Actor network architecture
        self.convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= num_input_channels, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Flatten(),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.ReLU()
        )
        self.fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.ReLU(),
            neuron_network.Linear(in_features= 128, out_features= 1),
        )
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):

        laser_obs_shape = laser_obs.shape
        goal_obs_shape = goal_obs.shape
        vel_obs_shape = vel_obs.shape

        if len(laser_obs_shape) == 2: 
            laser_obs = laser_obs[None, ...]

        if len(goal_obs_shape) == 1: 
            goal_obs = goal_obs[None, ...]

        if len(vel_obs_shape) == 1: 
            vel_obs = vel_obs[None, ...]

        # Get output from convolution and fully-connected
        batch_size = laser_obs.shape[0]

        conv_out = self.convolution(laser_obs)
        conv_out = conv_out.reshape(batch_size, 256)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs), dim = 1)
        value = self.fully_connected(fully_in)

        if len(goal_obs_shape) == 1:
            return value[0]

        return value