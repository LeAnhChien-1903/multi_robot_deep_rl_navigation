#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # Actor network architecture
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels= 3, out_channels= 32, kernel_size= 5, stride= 2),
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
            nn.Linear(in_features= 128, out_features= 2)
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        # Actor standard deviation
        self.log_std = nn.Parameter(torch.ones(2) * 0.0)
        
        self.apply(init_weights)

    def get_action_mean(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        conv_out = self.convolution(laser_obs)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs), dim = 1)
        mean = self.fully_connected(fully_in)
        linear_mean = self.sigmoid(mean[:, 0]).reshape((-1, 1))
        angular_mean = self.tanh(mean[:, 1]).reshape((-1, 1))

        return torch.cat((linear_mean, angular_mean), dim = 1)
    
    def get_action(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        mean = self.get_action_mean(laser_obs, goal_obs, vel_obs)
        std = self.log_std.exp()
        cov = torch.diag(std)
        dist = MultivariateNormal(mean, cov)
        
        action = dist.sample()
        
        return action, dist.log_prob(action), mean
    def evaluate(self,  laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, action: torch.Tensor):
        mean = self.get_action_mean(laser_obs, goal_obs, vel_obs)
        std = self.log_std.exp()
        cov = torch.diag(std)
        dist = MultivariateNormal(mean, cov)
        
        return dist.log_prob(action), dist.entropy(), mean

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels= 3, out_channels= 32, kernel_size= 5, stride= 2),
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
            nn.Linear(in_features= 128, out_features= 1)
        )
        self.apply(init_weights)
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        conv_out = self.convolution(laser_obs)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs), dim = 1)
        value = self.fully_connected(fully_in)
        return value
