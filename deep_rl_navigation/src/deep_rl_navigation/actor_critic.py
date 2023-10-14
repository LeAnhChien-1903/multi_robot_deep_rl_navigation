#!/usr/bin/env python

import torch
import torch.nn as neuron_network
from torch.distributions.normal import Normal

class Actor(neuron_network.Module):
    def __init__(self, path: str):
        super(Actor, self).__init__()
        self.convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= 3, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Flatten(0, -1),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.LeakyReLU()
        )
        self.fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.LeakyReLU(),
            neuron_network.Linear(in_features= 128, out_features= 2)
        )
        self.sigmoid =  neuron_network.Sigmoid()
        self.tanh = neuron_network.Tanh()
        
        self.actor_logstd = neuron_network.Parameter(torch.zeros(1, 2))

        self.load_state_dict(torch.load(path))
    
    def get_action_mean(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, device):
        action_mean = self.convolution(laser_obs)
        action_mean = torch.cat((action_mean, goal_obs, vel_obs))
        action_mean = self.fully_connected(action_mean)
        linear_mean = self.sigmoid(action_mean[0])
        angular_mean = self.tanh(action_mean[1])

        return torch.tensor([linear_mean, angular_mean]).to(device)
    
    def get_action(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, device, action = None):
        action_mean = self.get_action_mean(laser_obs, goal_obs, vel_obs, device)
        action_mean = torch.reshape(action_mean, (1, 2))
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

class Critic(neuron_network.Module):
    def __init__(self, path):
        super(Critic, self).__init__()
        self.convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= 3, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Flatten(0, -1),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.LeakyReLU()
        )
        self.fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.LeakyReLU(),
            neuron_network.Linear(in_features= 128, out_features= 1)
        )
        
        self.load_state_dict(torch.load(path))
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        value = self.convolution(laser_obs)
        value = torch.cat((value, goal_obs, vel_obs))
        value = self.fully_connected(value)
        
        return value

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)
