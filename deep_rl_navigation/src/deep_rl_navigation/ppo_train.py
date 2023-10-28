import torch
import torch.nn as neuron_network
import torch.optim as optimization
from torch.distributions import MultivariateNormal
import random 
import numpy as np
import matplotlib.pyplot as plt
import os
from deep_rl_navigation.actor_critic import *
from deep_rl_navigation.ultis import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/data"
actor_parameters_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/policy_parameters/actor_parameters.pt"
critic_parameters_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/policy_parameters/critic_parameters.pt"
number_of_robot = 5
robot_paths = []


policy = Actor(actor_parameters_path)
old_policy = Actor(actor_parameters_path)

critic = Critic(critic_parameters_path)
old_critic = Critic(critic_parameters_path)

ppo_param = PPO()
beta_file = open("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/data/reward/beta.text", 'r')
ppo_param.beta = float(beta_file.read())
beta_file.close()
print("Current beta:", ppo_param.beta)
class RobotData:
    def __init__(self, data_path, robot_name):
        path = os.path.join(data_path, robot_name)
        self.laser_obs = torch.load(path + "/laser_obs.pt")
        self.goal_obs = torch.load(path + "/goal_obs.pt")
        self.vel_obs = torch.load(path + "/vel_obs.pt")
        self.action = torch.load(path + "/actions.pt")
        self.done = torch.load(path + "/done.pt")
        self.log_prob = torch.load(path + "/log_prob.pt")
        self.reward = torch.load(path + "/reward.pt")
        self.value = torch.load(path + "/value.pt")
        self.entropy = torch.load(path + "/entropy.pt")
        self.mean = torch.load(path + "/mean.pt")
        self.std = torch.load(path + "/std.pt")
        self.advantage = torch.zeros_like(self.reward)
        self.returns = torch.zeros_like(self.reward)

# Optimizer
policy_optimizer = optimization.Adam(policy.parameters(), lr= ppo_param.lr_theta_1st)
critic_optimizer = optimization.Adam(critic.parameters(), lr= ppo_param.lr_phi)

robot_data = []
for i in range(number_of_robot):
    robot_data.append(RobotData(data_path, "robot_" + str(i)))

# Create accumulative rewards
reward_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/data/reward/reward.text"
previous_rewards = np.loadtxt(reward_path)
previous_rewards.reshape((-1, number_of_robot))
current_rewards = np.zeros(number_of_robot)
for i in range(len(robot_data)):
    sum_reward = torch.sum(robot_data[i].reward)
    current_rewards[i] = sum_reward.numpy()

total_reward = np.concatenate((previous_rewards, current_rewards.reshape(1, number_of_robot)), axis=0)
np.savetxt(reward_path, total_reward, fmt='%.2f')

num_of_step = robot_data[0].value.shape[0] - 1
# Calculate the advantage function
for robot in robot_data:
    delta = torch.zeros(num_of_step)
    
    for t in range(num_of_step):
        next_non_terminal = 1.0 - robot.done[t+1]
        delta[t] = robot.reward[t+1] + ppo_param.gamma * robot.value[t+1] * next_non_terminal - robot.value[t]
    
    # Calculate the advantage function
    for t in range(num_of_step):
        for l in range(num_of_step - t):
            robot.advantage[t] = robot.advantage[t] + (ppo_param.gamma * ppo_param.lambda_)**l*delta[t+l]

# Data for training
laser_obs = robot_data[0].laser_obs[0:num_of_step]
goal_obs = robot_data[0].goal_obs[0:num_of_step]
vel_obs = robot_data[0].vel_obs[0: num_of_step]
action = robot_data[0].action[0:num_of_step]
mean = robot_data[0].mean[0:num_of_step]
std = robot_data[0].std[0:num_of_step]
log_prob = robot_data[0].log_prob[0: num_of_step]
value = robot_data[0].value[0: num_of_step]
advantage = robot_data[0].advantage[0: num_of_step]

for i in range(1, len(robot_data)):
    laser_obs = torch.concatenate((laser_obs, robot_data[i].laser_obs[0:num_of_step]))
    goal_obs = torch.concatenate((goal_obs, robot_data[i].goal_obs[0:num_of_step]))
    vel_obs = torch.concatenate((vel_obs, robot_data[i].vel_obs[0:num_of_step]))
    action = torch.concatenate((action, robot_data[i].action[0:num_of_step]))
    mean = torch.concatenate((mean, robot_data[i].mean[0:num_of_step]))
    std = torch.concatenate((std, robot_data[i].std[0:num_of_step]))
    log_prob = torch.concatenate((log_prob, robot_data[i].log_prob[0:num_of_step]))
    value = torch.concatenate((value, robot_data[i].value[0:num_of_step]))
    advantage = torch.concatenate((advantage, robot_data[i].advantage[0:num_of_step]))

# Start training
new_log_prob = torch.zeros_like(log_prob)
# new_value = torch.zeros_like(value)
kl_divergence = torch.zeros_like(log_prob)
kl_divergence_new_old = torch.zeros(1)
count = 1
for j in range(ppo_param.E_phi):
    print("Policy optimizer:" , count)
    # Calculate loss
    for t in range(ppo_param.T_max):
        _, log_prob_new, entropy_new, mean_new, std_new = policy.get_action(laser_obs[t], goal_obs[t], vel_obs[t], action[t])
        prev_cov_mat = torch.diag(std[t])
        curr_cov_mat = torch.diag(std_new)
        prev_dist = MultivariateNormal(mean[t], prev_cov_mat)
        curr_dist = MultivariateNormal(mean_new, curr_cov_mat)
        kl_divergence[t] = torch.distributions.kl_divergence(prev_dist, curr_dist)
        # value_new = critic.get_value(laser_obs[j], goal_obs[j], vel_obs[j])
        new_log_prob[t] = log_prob_new
        # new_value[j] = value_new
    
    log_ratio = new_log_prob - log_prob
    ratio = log_ratio.exp()
    kl_divergence_new_old = kl_divergence.mean()
    print("KL divergence",kl_divergence_new_old)
    if (kl_divergence_new_old > 4 * ppo_param.KL_target): continue
    
    loss = advantage * ratio - ppo_param.beta * kl_divergence_new_old + ppo_param.xi * torch.pow(torch.max(torch.zeros(1), kl_divergence_new_old - 2 * ppo_param.KL_target), 2)
    policy_loss = loss.mean()
    print("Policy loss:", policy_loss)
    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph= True)
    policy_optimizer.step()
    count += 1

policy.save_parameters(actor_parameters_path)

for k in range(ppo_param.E_v):
    value_loss = torch.zeros(1)
    print("Critic optimizer:", k)
    for robot in robot_data:
        robot_loss = torch.zeros(1)
        # Calculate the value list
        value_list = torch.zeros_like(robot.value)
        for t in range(num_of_step):
            value_list[t] = critic.get_value(robot.laser_obs[t], robot.goal_obs[t], robot.vel_obs[t])
        
        for t in range(num_of_step):
            if t == num_of_step - 1:
                continue
            temp_loss = torch.zeros(1)
            for t_ in range(t + 1, num_of_step):
                temp_loss += ppo_param.gamma**(t_ - t) * robot.reward[t_+1] - value_list[t]
            robot_loss = robot_loss + temp_loss**2
        value_loss = value_loss + robot_loss
        
    value_loss = - value_loss
    critic_optimizer.zero_grad()
    value_loss.backward(retain_graph= True)
    critic_optimizer.step()
    
critic.save_parameters(critic_parameters_path)

if kl_divergence_new_old > ppo_param.beta_high * ppo_param.KL_target:
    ppo_param.beta = ppo_param.alpha * ppo_param.beta
elif kl_divergence_new_old < ppo_param.beta_low * ppo_param.KL_target:
    ppo_param.beta = ppo_param.beta/ ppo_param.alpha

print("New beta:", ppo_param.beta)
beta_file = open("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/data/reward/beta.text", 'w')
beta_file.write(str(ppo_param.beta))
beta_file.close()

print("Done!")