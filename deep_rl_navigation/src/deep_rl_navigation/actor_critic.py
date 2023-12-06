#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import Categorical
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
class Actor(nn.Module):
    def __init__(self, num_input_channels = 3, num_of_actions = 21):
        super(Actor, self).__init__()
        # Actor network architecture
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels= num_input_channels, out_channels= 32, kernel_size= 5, stride= 2),
            nn.ReLU(),
            nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features= 4288, out_features= 256),
            nn.ReLU()
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features= 260, out_features= 128),
            nn.ReLU(),
        )
        self.linear_vel_network = nn.Sequential(
            nn.Linear(in_features= 128, out_features= num_of_actions),
            nn.Softmax(dim = 1)
        )
        self.angular_vel_network = nn.Sequential(
            nn.Linear(in_features= 128, out_features= num_of_actions),
            nn.Softmax(dim = 1)
        )
    
    def get_action(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.convolution(laser_obs)
    
        fully_in = torch.cat((laser_out, goal_obs, vel_obs), dim = 1)
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
        batch_size = linear_probs.shape[0]
        num_of_actions = linear_probs.shape[1]
        
        log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        vel_probs = torch.matmul(linear_probs.reshape(batch_size, num_of_actions, 1),
                                angular_probs.reshape(batch_size, 1, num_of_actions)).reshape(batch_size, -1)
        
        return linear_vel, angular_vel, log_prob, vel_probs
    
    def exploit_policy(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.convolution(laser_obs)
    
        fully_in = torch.cat((laser_out, goal_obs, vel_obs), dim = 1)
        fully_out = self.fully_connected(fully_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs: torch.Tensor = self.linear_vel_network(fully_out)
        
        angular_probs: torch.Tensor = self.angular_vel_network(fully_out)
        
        # Get action
        linear_vel = linear_probs.argmax(dim=1)
        angular_vel = angular_probs.argmax(dim=1)
        
        return linear_vel, angular_vel

    def evaluate(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, linear_vel: torch.Tensor, angular_vel: torch.Tensor):
        laser_out = self.convolution(laser_obs)
    
        fully_in = torch.cat((laser_out, goal_obs, vel_obs), dim = 1)
        fully_out = self.fully_connected(fully_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.linear_vel_network(fully_out)
        angular_probs: torch.Tensor = self.angular_vel_network(fully_out)
        
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)

        # Get action log prob
        batch_size = linear_probs.shape[0]
        num_of_actions = linear_probs.shape[1]
        
        log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        vel_probs = torch.matmul(linear_probs.reshape(batch_size, num_of_actions, 1),
                                angular_probs.reshape(batch_size, 1, num_of_actions)).reshape(batch_size, -1)
        
        return log_prob, vel_probs
            
class Critic(nn.Module):
    def __init__(self, num_input_channels = 3):
        super(Critic, self).__init__()
        # Actor network architecture
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels= num_input_channels, out_channels= 32, kernel_size= 5, stride= 2),
            nn.ReLU(),
            nn.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features= 4288, out_features= 256),
            nn.ReLU()
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features= 260, out_features= 128),
            nn.ReLU(),
            nn.Linear(in_features= 128, out_features= 1),
        )
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.convolution(laser_obs)
    
        fully_in = torch.cat((laser_out, goal_obs, vel_obs), dim = 1)
        value = self.fully_connected(fully_in)
        
        return value
