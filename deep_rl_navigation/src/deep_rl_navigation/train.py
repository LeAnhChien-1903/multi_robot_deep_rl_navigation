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
parameter_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/parameters"
# Setup parameters
number_of_robot = 5
number_of_action = 21
num_observations = 3
num_laser_ray = 541
ppo_param = PPO()

# Load model
actor = ActorDiscrete(num_input_channels= 3, num_of_actions= number_of_action)
if not os.path.exists(os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(num_observations, num_laser_ray, number_of_action))):
    torch.save(actor.state_dict(), os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(num_observations, num_laser_ray, number_of_action)))
    print("Save initialized actor model!")
else:
    actor.load_state_dict(torch.load(os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(num_observations, num_laser_ray, number_of_action)), map_location='cpu'))
    print("Load actor model!")
critic = CriticDiscrete(num_input_channels= num_observations)
if not os.path.exists(os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(num_observations, num_laser_ray))):
    torch.save(critic.state_dict(), os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(num_observations, num_laser_ray)))
    print("Save initialized critic model!")
else:
    actor.load_state_dict(torch.load(os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(num_observations, num_laser_ray)), map_location= 'cpu'))
    print("Load critic model!")

# Optimizer
policy_optimizer = optimization.Adam(actor.parameters(), lr= ppo_param.lr_theta_1st)
critic_optimizer = optimization.Adam(critic.parameters(), lr= ppo_param.lr_phi)

# Load beta
beta_file = open(os.path.join(data_path, "reward", "beta.txt"), 'r')
ppo_param.beta = float(beta_file.read())
beta_file.close()
print("Current beta:", ppo_param.beta)

class RobotData:
    def __init__(self, data_path, robot_name):
        path = os.path.join(data_path, robot_name)
        self.laser_obs_batch = torch.load(path + "/laser_obs.pt")
        self.goal_obs_batch = torch.load(path + "/goal_obs.pt")
        self.vel_obs_batch = torch.load(path + "/vel_obs.pt")
        self.linear_vel_batch = torch.load(path + "/linear_vel.pt")
        self.angular_vel_batch = torch.load(path + "/angular_vel.pt")
        self.linear_probs_batch = torch.load(path + "/linear_probs.pt")
        self.angular_probs_batch = torch.load(path + "/angular_probs.pt")
        self.done_batch = torch.load(path + "/done.pt")
        self.log_prob_batch = torch.load(path + "/log_prob.pt")
        self.reward_batch = torch.load(path + "/reward.pt")
        self.value_batch = torch.load(path + "/value.pt")
        self.advantage_batch = torch.zeros_like(self.reward_batch)
        self.returns_batch = torch.zeros_like(self.reward_batch)

# Load data from database
robot_data = []
for i in range(number_of_robot):
    robot_data.append(RobotData(data_path, "robot_" + str(i)))

# Create accumulative rewards
previous_rewards = np.loadtxt(os.path.join(data_path, "reward", "reward.txt"))
previous_rewards.reshape((-1, number_of_robot))
current_rewards = np.zeros(number_of_robot)

for i in range(len(robot_data)):
    sum_reward = torch.sum(robot_data[i].reward_batch)
    current_rewards[i] = sum_reward.numpy()

print("Total reward:", current_rewards.sum())
total_reward = np.concatenate((previous_rewards, current_rewards.reshape(1, number_of_robot)), axis=0)
np.savetxt(os.path.join(data_path, "reward", "reward.txt"), total_reward, fmt='%.2f')


# Calculate the advantage function
num_of_step = robot_data[0].value_batch.shape[0] - 1
for robot in robot_data:
    last_advantage = torch.zeros(1)
    last_value = robot.value_batch[-1]
    
    for t in reversed(range(num_of_step)):
        mask = 1.0 - robot.done_batch[t+1]
        last_value = last_value * mask
        last_advantage = last_advantage * mask
        
        delta = robot.reward_batch[t] + ppo_param.gamma * last_value - robot.value_batch[t]
        
        last_advantage = delta + ppo_param.gamma * ppo_param.lambda_ * last_advantage
        
        robot.advantage_batch[t] = last_advantage
        
        last_value = robot.value_batch[t]

# # Data for training
laser_obs_batch = robot_data[0].laser_obs_batch[0:num_of_step]
goal_obs_batch = robot_data[0].goal_obs_batch[0:num_of_step]
vel_obs_batch = robot_data[0].vel_obs_batch[0: num_of_step]
linear_vel_batch = robot_data[0].linear_vel_batch[0:num_of_step]
linear_probs_batch = robot_data[0].linear_probs_batch[0:num_of_step]
angular_vel_batch = robot_data[0].angular_vel_batch[0:num_of_step]
angular_probs_batch = robot_data[0].angular_probs_batch[0:num_of_step]
log_prob_batch = robot_data[0].log_prob_batch[0: num_of_step]
value_batch = robot_data[0].value_batch[0: num_of_step]
advantage_batch = robot_data[0].advantage_batch[0: num_of_step]

for i in range(1, len(robot_data)):
    laser_obs_batch = torch.concatenate((laser_obs_batch, robot_data[i].laser_obs_batch[0:num_of_step]))
    goal_obs_batch = torch.concatenate((goal_obs_batch, robot_data[i].goal_obs_batch[0:num_of_step]))
    vel_obs_batch = torch.concatenate((vel_obs_batch, robot_data[i].vel_obs_batch[0:num_of_step]))
    linear_vel_batch = torch.concatenate((linear_vel_batch, robot_data[i].linear_vel_batch[0:num_of_step]))
    linear_probs_batch = torch.concatenate((linear_probs_batch, robot_data[i].linear_probs_batch[0:num_of_step]))
    angular_vel_batch = torch.concatenate((angular_vel_batch, robot_data[i].angular_vel_batch[0:num_of_step]))
    angular_probs_batch = torch.concatenate((angular_probs_batch, robot_data[i].angular_probs_batch[0:num_of_step]))
    log_prob_batch = torch.concatenate((log_prob_batch, robot_data[i].log_prob_batch[0:num_of_step]))
    value_batch = torch.concatenate((value_batch, robot_data[i].value_batch[0:num_of_step]))
    advantage_batch = torch.concatenate((advantage_batch, robot_data[i].advantage_batch[0:num_of_step]))

# Start training
new_log_prob_batch= torch.zeros_like(log_prob_batch)
new_value_batch = torch.zeros_like(value_batch)
kl_divergence_batch = torch.zeros_like(log_prob_batch)
kl_divergence_new_old = torch.zeros(1)
count = 1
for j in range(ppo_param.E_phi):
    print("Policy optimizer:" , count)
    # Calculate loss
    for t in range(ppo_param.T_max):
        new_log_prob, new_linear_probs, new_angular_probs = actor.evaluate(laser_obs_batch[t], goal_obs_batch[t], vel_obs_batch[t], linear_vel_batch[t], angular_vel_batch[t])
        kl_divergence_batch[t] = calculateMultiKLDivergence(linear_probs_batch[t], angular_probs_batch[t], new_linear_probs, new_angular_probs)
        new_log_prob_batch[t] = new_log_prob 
    
    log_ratio = new_log_prob_batch - log_prob_batch
    ratio = log_ratio.exp()
    
    kl_divergence_new_old = kl_divergence_batch.mean()
    
    if (kl_divergence_new_old > 4 * ppo_param.KL_target): continue
    
    loss = -advantage_batch * ratio + ppo_param.beta * kl_divergence_new_old - ppo_param.xi * torch.pow(torch.max(torch.zeros(1), kl_divergence_new_old - 2 * ppo_param.KL_target), 2)
    policy_loss = loss.sum()

    policy_optimizer.zero_grad()
    policy_loss.backward(retain_graph= True)
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
    policy_optimizer.step()
    count += 1

torch.save(actor.state_dict(), os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(num_observations, num_laser_ray, number_of_action)))

for k in range(ppo_param.E_v):
    value_loss = torch.zeros(1)
    print("Critic optimizer:", k)
    for robot in robot_data:
        robot_loss = torch.zeros(1)
        # Calculate the value list
        value_batch = torch.zeros_like(robot.value_batch)
        for t in range(num_of_step):
            value_batch[t] = critic.get_value(robot.laser_obs_batch[t], robot.goal_obs_batch[t], robot.vel_obs_batch[t])
        
        for t in range(num_of_step):
            if t == num_of_step - 1:
                continue
            temp_loss = torch.zeros(1)
            for t_ in range(t + 1, num_of_step):
                temp_loss += ppo_param.gamma**(t_ - t) * robot.reward[t_+1] - value_batch[t]
            robot_loss = robot_loss + temp_loss**2
        value_loss = value_loss + robot_loss
        
    value_loss = - value_loss
    critic_optimizer.zero_grad()
    value_loss.backward(retain_graph= True)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    critic_optimizer.step()
    
torch.save(critic.state_dict(), os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(num_observations, num_laser_ray)))

if kl_divergence_new_old > ppo_param.beta_high * ppo_param.KL_target:
    ppo_param.beta = ppo_param.alpha * ppo_param.beta
elif kl_divergence_new_old < ppo_param.beta_low * ppo_param.KL_target:
    ppo_param.beta = ppo_param.beta/ ppo_param.alpha

print("New beta:", ppo_param.beta)
beta_file = open("/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/data/reward/beta.text", 'w')
beta_file.write(str(ppo_param.beta))
beta_file.close()

print("Done!")