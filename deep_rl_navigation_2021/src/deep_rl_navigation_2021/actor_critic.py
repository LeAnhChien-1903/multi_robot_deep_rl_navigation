#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.distributions import Categorical
class Actor(nn.Module):
    def __init__(self, num_of_laser_channels = 4, num_of_actions = 21):
        super(Actor, self).__init__()
        # Actor network architecture
        self.laser_net = nn.Sequential(
            nn.Conv1d(in_channels= num_of_laser_channels, out_channels= 16, kernel_size= 7, stride= 3),
            nn.ReLU(),
            nn.Conv1d(in_channels= 16, out_channels= 32, kernel_size= 5, stride= 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features= 2816, out_features= 256),
            nn.ReLU()
        )
        self.orient_net = nn.Sequential(
            nn.Linear(in_features= 2, out_features= 32),
            nn.ReLU(),
        )
        self.dist_net = nn.Sequential(
            nn.Linear(in_features= 1, out_features= 16),
            nn.ReLU()
        )
        self.vel_net = nn.Sequential(
            nn.Linear(in_features= 2, out_features= 32), 
            nn.ReLU()
        )
        self.final_net = nn.Sequential(
            nn.Linear(in_features= 256 + 32 + 16 + 32, out_features= 384),
            nn.ReLU(),
        )
        self.out_linear_net = nn.Sequential(
            nn.Linear(in_features= 384, out_features= num_of_actions),
            nn.Softmax(-1)
        )
        self.out_angular_net = nn.Sequential(
            nn.Linear(in_features= 384, out_features= num_of_actions),
            nn.Softmax(-1)
        )
    
    def get_action(self, laser_obs: torch.Tensor, orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        final_out = self.final_net(final_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.out_linear_net(final_out)
        angular_probs: torch.Tensor = self.out_angular_net(final_out)
        
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        
        # Get action
        linear_vel = linear_distribution.sample()
        angular_vel = angular_distribution.sample()
        
        # Get action log prob
        batch_size = linear_probs.shape[0]
        num_of_actions = linear_probs.shape[1]
        
        vel_probs = torch.matmul(linear_probs.reshape(batch_size, num_of_actions, 1),
                                angular_probs.reshape(batch_size, 1, num_of_actions)).reshape(batch_size, -1)
        index = (linear_vel.type(torch.IntTensor) * num_of_actions + angular_vel.type(torch.IntTensor)).reshape(-1)
        
        action_log_prob = torch.log(vel_probs[:, index].diag())
        
        return linear_vel, angular_vel, action_log_prob, linear_probs.reshape(num_of_actions), angular_probs.reshape(num_of_actions)
    
    def exploit_policy(self, laser_obs: torch.Tensor, orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        final_out = self.final_net(final_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.out_linear_net(final_out)
        angular_probs: torch.Tensor = self.out_angular_net(final_out)
        
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        
        # Get action
        linear_vel = linear_probs.argmax(-1)
        angular_vel = angular_probs.argmax(-1)
        log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        return linear_vel, angular_vel, log_prob

    def evaluate(self, laser_obs: torch.Tensor,  orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor, linear_vel: torch.Tensor, angular_vel: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        final_out = self.final_net(final_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.out_linear_net(final_out)
        angular_probs: torch.Tensor = self.out_angular_net(final_out)
        
        # Get action log prob
        batch_size = linear_probs.shape[0]
        num_of_actions = linear_probs.shape[1]
        
        vel_probs = torch.matmul(linear_probs.reshape(batch_size, num_of_actions, 1),
                                angular_probs.reshape(batch_size, 1, num_of_actions)).reshape(batch_size, -1)
        index = (linear_vel.type(torch.IntTensor) * num_of_actions + angular_vel.type(torch.IntTensor)).reshape(-1)
        
        action_log_prob = torch.log(vel_probs[:, index].diag())
        
        return action_log_prob, linear_probs.reshape(batch_size, num_of_actions), angular_probs.reshape(batch_size, num_of_actions)
            
class Critic(nn.Module):
    def __init__(self, num_of_laser_channels = 4):
        super(Critic, self).__init__()
        # Actor network architecture
        self.laser_net = nn.Sequential(
            nn.Conv1d(in_channels= num_of_laser_channels, out_channels= 16, kernel_size= 7, stride= 3),
            nn.ReLU(),
            nn.Conv1d(in_channels= 16, out_channels= 32, kernel_size= 5, stride= 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features= 2816, out_features= 256),
            nn.ReLU()
        )
        self.orient_net = nn.Sequential(
            nn.Linear(in_features= 2, out_features= 32),
            nn.ReLU(),
        )
        self.dist_net = nn.Sequential(
            nn.Linear(in_features= 1, out_features= 16),
            nn.ReLU()
        )
        self.vel_net = nn.Sequential(
            nn.Linear(in_features= 2, out_features= 32), 
            nn.ReLU()
        )
        self.final_net = nn.Sequential(
            nn.Linear(in_features= 256 + 32 + 16 + 32, out_features= 384),
            nn.ReLU(),
            nn.Linear(in_features= 384, out_features= 1),
        )
    
    def get_value(self, laser_obs: torch.Tensor, orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        value = self.final_net(final_in)
        
        return value

class ActorCritic(nn.Module):
    def __init__(self, num_of_laser_channels = 4, num_of_actions = 21):
        super(ActorCritic, self).__init__()
        # Actor network architecture
        self.laser_net = nn.Sequential(
            nn.Conv1d(in_channels= num_of_laser_channels, out_channels= 16, kernel_size= 7, stride= 3),
            nn.ReLU(),
            nn.Conv1d(in_channels= 16, out_channels= 32, kernel_size= 5, stride= 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features= 2816, out_features= 256),
            nn.ReLU()
        )
        self.orient_net = nn.Sequential(
            nn.Linear(in_features= 2, out_features= 32),
            nn.ReLU(),
        )
        self.dist_net = nn.Sequential(
            nn.Linear(in_features= 1, out_features= 16),
            nn.ReLU()
        )
        self.vel_net = nn.Sequential(
            nn.Linear(in_features= 2, out_features= 32), 
            nn.ReLU()
        )
        self.final_net = nn.Sequential(
            nn.Linear(in_features= 256 + 32 + 16 + 32, out_features= 384),
            nn.ReLU(),
        )
        self.out_linear_net = nn.Sequential(
            nn.Linear(in_features= 384, out_features= num_of_actions),
            nn.Softmax(-1)
        )
        self.out_angular_net = nn.Sequential(
            nn.Linear(in_features= 384, out_features= num_of_actions),
            nn.Softmax(-1)
        )
        self.out_critic_net = nn.Sequential(
            nn.Linear(in_features= 384, out_features= 1)
        )
    
    def get_action(self, laser_obs: torch.Tensor, orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        final_out = self.final_net(final_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.out_linear_net(final_out)
        angular_probs: torch.Tensor = self.out_angular_net(final_out)
        
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        
        # Get action
        linear_vel = linear_distribution.sample()
        angular_vel = angular_distribution.sample()
        # Get action log prob
        batch_size = linear_probs.shape[0]
        num_of_actions = linear_probs.shape[1]
        
        vel_probs = torch.matmul(linear_probs.reshape(batch_size, num_of_actions, 1),
                                angular_probs.reshape(batch_size, 1, num_of_actions)).reshape(batch_size, -1)
        index = (linear_vel.type(torch.IntTensor) * num_of_actions + angular_vel.type(torch.IntTensor)).reshape(-1)
        
        action_log_prob = torch.log(vel_probs[:, index].diag())
        # Get value function
        value: torch.Tensor = self.out_critic_net(final_out)
        
        return linear_vel, angular_vel, action_log_prob, linear_probs.reshape(num_of_actions), angular_probs.reshape(num_of_actions), value
    
    def exploit_policy(self, laser_obs: torch.Tensor, orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        final_out = self.final_net(final_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.out_linear_net(final_out)
        angular_probs: torch.Tensor = self.out_angular_net(final_out)
        
        # Create linear velocity and angular velocity distributions
        linear_distribution = Categorical(linear_probs)
        angular_distribution = Categorical(angular_probs)
        
        # Get action
        linear_vel = linear_probs.argmax(-1)
        angular_vel = angular_probs.argmax(-1)
        log_prob = linear_distribution.log_prob(linear_vel) + angular_distribution.log_prob(angular_vel)
        
        return linear_vel, angular_vel, log_prob

    def evaluate(self, laser_obs: torch.Tensor,  orient_obs: torch.Tensor, dist_obs: torch.Tensor, vel_obs: torch.Tensor, linear_vel: torch.Tensor, angular_vel: torch.Tensor):
        laser_out = self.laser_net(laser_obs)
        orient_out = self.orient_net(orient_obs)
        dist_out = self.dist_net(dist_obs)
        vel_out = self.vel_net(vel_obs)
        
        final_in = torch.cat((laser_out, orient_out, dist_out, vel_out), dim = 1)
        final_out = self.final_net(final_in)
        
        # Get linear velocity and angular velocity probabilities list
        linear_probs:torch.Tensor = self.out_linear_net(final_out)
        angular_probs: torch.Tensor = self.out_angular_net(final_out)
        
        batch_size = linear_probs.shape[0]
        num_of_actions = linear_probs.shape[1]
        
        vel_probs = torch.matmul(linear_probs.reshape(batch_size, num_of_actions, 1),
                                angular_probs.reshape(batch_size, 1, num_of_actions)).reshape(batch_size, -1)
        index = (linear_vel.type(torch.IntTensor) * num_of_actions + angular_vel.type(torch.IntTensor)).reshape(-1)
        
        action_log_prob = torch.log(vel_probs[:, index].diag())
        # Get value function
        value: torch.Tensor = self.out_critic_net(final_out)
        
        return action_log_prob, linear_probs.reshape(batch_size, num_of_actions), angular_probs.reshape(batch_size, num_of_actions), value