#!/usr/bin/env python

import torch
import torch.nn as neuron_network
from torch.distributions import MultivariateNormal, Categorical

class ActorCritic(neuron_network.Module):
    def __init__(self, path: str):
        super(ActorCritic, self).__init__()
        # Actor network architecture
        self.actor_convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= 3, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Flatten(0, -1),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.LeakyReLU()
        )
        self.actor_fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.LeakyReLU(),
            neuron_network.Linear(in_features= 128, out_features= 2)
        )
        # Critic network architecture
        self.critic_convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= 3, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.LeakyReLU(),
            neuron_network.Flatten(0, -1),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.LeakyReLU()
        )
        self.critic_fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.LeakyReLU(),
            neuron_network.Linear(in_features= 128, out_features= 1)
        )
        self.sigmoid = neuron_network.Sigmoid()
        self.tanh = neuron_network.Tanh()
        
        # Actor standard deviation
        self.actor_std = neuron_network.Parameter(torch.tensor([0.5, 0.5]))

        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def get_action_mean(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        action_mean = self.actor_convolution(laser_obs)
        action_mean = torch.cat((action_mean, goal_obs, vel_obs))
        action_mean = self.actor_fully_connected(action_mean)
        linear_mean = self.sigmoid(action_mean[0])
        angular_mean = self.tanh(action_mean[1])

        return torch.tensor([linear_mean, angular_mean])
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        value = self.critic_convolution(laser_obs)
        value = torch.cat((value, goal_obs, vel_obs))
        value = self.critic_fully_connected(value)
        
        return value
    
    def get_action_and_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, action = None):
        value = self.get_value(laser_obs, goal_obs, vel_obs)
        action_mean = self.get_action_mean(laser_obs, goal_obs, vel_obs)
        cov_mat = torch.diag(self.actor_std)
        distribution = MultivariateNormal(action_mean, cov_mat)
        
        if action is None:
            action = distribution.sample()
        
        return action, value, distribution.log_prob(action), distribution.entropy()

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

class Actor(neuron_network.Module):
    def __init__(self, path: str):
        super(Actor, self).__init__()
        # Actor network architecture
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
        self.sigmoid = neuron_network.Sigmoid()
        self.tanh = neuron_network.Tanh()
        
        # Actor standard deviation
        self.actor_log_std = neuron_network.Parameter(torch.zeros(2))

        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def get_action_mean(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        action_mean = self.convolution(laser_obs)
        action_mean = torch.cat((action_mean, goal_obs, vel_obs))
        action_mean = self.fully_connected(action_mean)
        linear_mean = self.sigmoid(action_mean[0])
        angular_mean = self.tanh(action_mean[1])

        return torch.tensor([linear_mean, angular_mean])
    
    def get_action(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, action = None):
        action_mean = self.get_action_mean(laser_obs, goal_obs, vel_obs)
        action_std = self.actor_log_std.exp()
        cov_mat = torch.diag(action_std)
        distribution = MultivariateNormal(action_mean, cov_mat)
        
        if action is None:
            action = distribution.sample()
        
        return action, distribution.log_prob(action), distribution.entropy(), action_mean, action_std

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
        
        self.load_state_dict(torch.load(path,  map_location=torch.device('cpu')))
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        value = self.convolution(laser_obs)
        value = torch.cat((value, goal_obs, vel_obs))
        value = self.fully_connected(value)
        
        return value

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

class ActorCuda(neuron_network.Module):
    def __init__(self, path: str):
        super(ActorCuda, self).__init__()
        # Actor network architecture
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
        self.sigmoid = neuron_network.Sigmoid()
        self.tanh = neuron_network.Tanh()
        
        # Actor standard deviation
        self.actor_std = neuron_network.Parameter(torch.tensor([0.5, 0.5]))

        self.load_state_dict(torch.load(path, map_location=torch.device('cuda')))
    
    def get_action_mean(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, device):
        action_mean = self.convolution(laser_obs)
        action_mean = torch.cat((action_mean, goal_obs, vel_obs))
        action_mean = self.fully_connected(action_mean)
        linear_mean = self.sigmoid(action_mean[0])
        angular_mean = self.tanh(action_mean[1])

        return torch.tensor([linear_mean, angular_mean]).to(device)
    
    def get_action(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, device, action = None):
        action_mean = self.get_action_mean(laser_obs, goal_obs, vel_obs, device)
        cov_mat = torch.diag(self.actor_std)
        distribution = MultivariateNormal(action_mean, cov_mat)
        
        if action is None:
            action = distribution.sample()
        
        return action.to(device), distribution.log_prob(action).to(device), distribution.entropy().to(device)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)
class CriticCuda(neuron_network.Module):
    def __init__(self, path):
        super(CriticCuda, self).__init__()
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
        
        self.load_state_dict(torch.load(path,  map_location=torch.device('cuda')))
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, device):
        value = self.convolution(laser_obs)
        value = torch.cat((value, goal_obs, vel_obs))
        value = self.fully_connected(value)
        
        return value.to(device)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)
        
class ActorDiscrete(neuron_network.Module):
    def __init__(self, num_input_channels = 3, num_of_actions = 20):
        super(ActorDiscrete, self).__init__()
        # Actor network architecture
        self.convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= num_input_channels, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Flatten(0, -1),
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
        conv_out = self.convolution(laser_obs)
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
        
        return linear_vel, angular_vel, action_log_prob.detach(), linear_probs, angular_probs

    def evaluate(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor, linear_vel: torch.Tensor, angular_vel: torch.Tensor):
        # Get output from convolution and fully-connected
        conv_out = self.convolution(laser_obs)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs))
        fully_out = self.fully_connected(fully_in)
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.linear_vel_network(fully_out)
        angular_probs: torch.Tensor = self.angular_vel_network(fully_out)
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        
        action_log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        return action_log_prob.detach(), linear_probs, angular_probs
class CriticDiscrete(neuron_network.Module):
    def __init__(self, num_input_channels = 3):
        super(CriticDiscrete, self).__init__()
        # Actor network architecture
        self.convolution = neuron_network.Sequential(
            neuron_network.Conv1d(in_channels= num_input_channels, out_channels= 32, kernel_size= 5, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Conv1d(in_channels= 32, out_channels= 32, kernel_size= 3, stride= 2),
            neuron_network.ReLU(),
            neuron_network.Flatten(0, -1),
            neuron_network.Linear(in_features= 4288, out_features= 256),
            neuron_network.ReLU()
        )
        self.fully_connected = neuron_network.Sequential(
            neuron_network.Linear(in_features= 260, out_features= 128),
            neuron_network.ReLU(),
            neuron_network.Linear(in_features= 128, out_features= 1),
        )
    
    def get_value(self, laser_obs: torch.Tensor, goal_obs: torch.Tensor, vel_obs: torch.Tensor):
        # Get output from convolution and fully-connected
        conv_out = self.convolution(laser_obs)
        fully_in = torch.cat((conv_out, goal_obs, vel_obs))
        value = self.fully_connected(fully_in)
        
        return value