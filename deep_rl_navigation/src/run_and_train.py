#!/usr/bin/env python

import numpy as np
import math
import os

import torch

from geometry_msgs.msg import Twist

from deep_rl_navigation.ultis import *
from deep_rl_navigation.actor_critic import *
from deep_rl_navigation.agent import *

parameter_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation/parameters"

# Initialize robot to collect data
rospy.init_node("deep_rl_node", anonymous=False)

# Initialize the hyper parameters
hyper_params = HyperParameters()

# Create linear and angular vector
linear_vel_vector = np.linspace(hyper_params.setup.min_linear_velocity, hyper_params.setup.max_linear_velocity, hyper_params.number_of_action)
angular_vel_vector = np.linspace(hyper_params.setup.min_angular_velocity, hyper_params.setup.max_angular_velocity, hyper_params.number_of_action)

# Create agents and single buffer list
agents = []
for i in range(hyper_params.number_of_robot):
    agents.append(Agent('/robot_{}'.format(i), hyper_param= hyper_params))



# Load actor and critic parameters
actor = ActorDiscrete(num_input_channels= hyper_params.setup.num_observations, num_of_actions= hyper_params.number_of_action)
if not os.path.exists(os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray, hyper_params.number_of_action))):
    torch.save(actor.state_dict(), os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray, hyper_params.number_of_action)))
    print("Save initialized actor model!")
else:
    actor.load_state_dict(torch.load(os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray, hyper_params.number_of_action)), map_location= 'cpu'))
    print("Load actor model!")


critic = CriticDiscrete(num_input_channels= hyper_params.setup.num_observations)
if not os.path.exists(os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray))):
    torch.save(critic.state_dict(), os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray)))
    print("Save initialized critic model!")
else:
    critic.load_state_dict(torch.load(os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray)), map_location= 'cpu'))
    print("Load critic model!")


# Create the optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr= hyper_params.ppo.lr_theta_1st)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr= hyper_params.ppo.lr_phi)
rate = rospy.Rate(float(1/hyper_params.setup.sample_time))

# for iteration in range(1000):
# Load beta parameters
beta_file = open(os.path.join(parameter_path, "beta.txt"), 'r')
hyper_params.ppo.beta = float(beta_file.read())
beta_file.close()
print("Current beta:", hyper_params.ppo.beta)

single_buffer_list = []
for i in range(hyper_params.number_of_robot):
    agents[i].randomPose()    
    single_buffer_list.append(SingleBuffer(hyper_params.mini_batch_size, hyper_params.setup, hyper_params.number_of_action))
# Check data 
for i in range(10):
    rate.sleep()

# Collect the data
for t in range(hyper_params.mini_batch_size + 1):
    print("Collecting data at t = {}".format(round(t*hyper_params.setup.sample_time, 1)))
    cmd_vel_list = []
    for i in range(hyper_params.number_of_robot):
        # Collect the sample data
        sample: dict = agents[i].step(actor, critic)
        # Update sampled data
        single_buffer_list[i].update(t, sample)
        # Publish the velocity to robot
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel_vector[sample['linear'].item()]
        cmd_vel.angular.z = angular_vel_vector[sample['angular'].item()]
        cmd_vel_list.append(cmd_vel)
        
    for i in range(hyper_params.number_of_robot):
        agents[i].cmd_vel_pub.publish(cmd_vel_list[i])
    rate.sleep()

zero_vel = Twist()
for i in range(hyper_params.number_of_robot):
    agents[i].cmd_vel_pub.publish(zero_vel)

# Create accumulative rewards
previous_rewards = np.loadtxt(os.path.join(parameter_path, "cumulative_reward.txt"))
previous_rewards.reshape((-1, hyper_params.number_of_robot))
current_rewards = np.zeros(hyper_params.number_of_robot)

for i in range(hyper_params.number_of_robot):
    sum_reward = torch.sum(single_buffer_list[i].reward_mini_batch)
    current_rewards[i] = sum_reward.numpy()

print("Total reward:", current_rewards.sum())

total_reward = np.concatenate((previous_rewards, current_rewards.reshape(1, hyper_params.number_of_robot)), axis=0)
np.savetxt(os.path.join(parameter_path, "cumulative_reward.txt"), total_reward, fmt='%.2f')

# Estimate the advantage function by using GAE
for i in range(hyper_params.number_of_robot):
    single_buffer_list[i].advantageEstimator(hyper_params.ppo.gamma, hyper_params.ppo.lambda_)

# Create buffer from single buffer list
buffer = Buffer(single_buffer_list)

# Start training
kl_divergence_final = 0.0
count = 1
for j in range(hyper_params.ppo.E_phi):
    print("Policy optimizer:" , count)
    new_log_prob_batch = torch.zeros(hyper_params.batch_size)
    new_value_batch = torch.zeros(hyper_params.batch_size)
    kl_divergence_batch = torch.zeros(hyper_params.batch_size)
    # Calculate loss
    for t in range(hyper_params.batch_size):
        new_log_prob, new_linear_probs, new_angular_probs = actor.evaluate( buffer.laser_obs_batch[t], buffer.goal_obs_batch[t], 
                                                                            buffer.vel_obs_batch[t], buffer.linear_vel_batch[t], 
                                                                            buffer.angular_vel_batch[t])
        kl_divergence_batch[t] = calculateMultiKLDivergence(buffer.linear_probs_batch[t], 
                                                            buffer.angular_probs_batch[t], 
                                                            new_linear_probs, new_angular_probs)
        new_log_prob_batch[t] = new_log_prob 
    
    log_ratio = new_log_prob_batch - buffer.log_prob_batch
    ratio = log_ratio.exp()
    
    kl_divergence_new_old = kl_divergence_batch.mean()
    kl_divergence_final = kl_divergence_new_old.item()
    count += 1
    if (kl_divergence_new_old > 4 * hyper_params.ppo.KL_target): 
        break
    
    loss1 = -buffer.advantage_batch * ratio
    loss2 = hyper_params.ppo.beta * kl_divergence_new_old
    loss3 = -hyper_params.ppo.xi * torch.pow(torch.max(torch.zeros(1), kl_divergence_new_old - 2 * hyper_params.ppo.KL_target), 2)
    loss = loss1 + loss2 + loss3
    actor_loss = loss.sum()
    
    # print("Policy loss:", actor_loss.item())
    actor_optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
    actor_optimizer.step()
    
    
    del new_log_prob_batch
    del new_value_batch 
    del kl_divergence_batch

torch.save(actor.state_dict(), os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(  hyper_params.setup.num_observations, 
                                                                                                    hyper_params.setup.num_laser_ray, 
                                                                                                    hyper_params.number_of_action)))

for k in range(hyper_params.ppo.E_v):
    value_loss = torch.zeros(1)
    print("Critic optimizer:", k+1)
    for single_buffer in single_buffer_list:
        robot_loss = torch.zeros(1)
        # Calculate the value list
        value_batch = torch.zeros_like(single_buffer.value_mini_batch)
        for t in range(hyper_params.mini_batch_size):
            value_batch[t] = critic.get_value(single_buffer.laser_obs_mini_batch[t], single_buffer.goal_obs_mini_batch[t], single_buffer.vel_obs_mini_batch[t])
        
        for t in range(hyper_params.mini_batch_size):
            temp_loss = torch.zeros(1)
            if t < hyper_params.mini_batch_size:
                for t_ in range(t + 1, hyper_params.mini_batch_size):
                    temp_loss += hyper_params.ppo.gamma**(t_ - t) * single_buffer.reward_mini_batch[t_]
            robot_loss = robot_loss + (temp_loss - value_batch[t])**2
        value_loss = value_loss + robot_loss
    
    # value_loss = - value_loss

    # print("Value loss:", value_loss.item())
    critic_optimizer.zero_grad()
    value_loss.backward(retain_graph= True)
    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
    critic_optimizer.step()
    
torch.save(critic.state_dict(), os.path.join(parameter_path, "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                                                hyper_params.setup.num_laser_ray)))

if kl_divergence_final > hyper_params.ppo.beta_high * hyper_params.ppo.KL_target:
    hyper_params.ppo.beta = hyper_params.ppo.alpha * hyper_params.ppo.beta
elif kl_divergence_final < hyper_params.ppo.beta_low * hyper_params.ppo.KL_target:
    hyper_params.ppo.beta = hyper_params.ppo.beta/ hyper_params.ppo.alpha

print("New beta:", hyper_params.ppo.beta)
beta_file = open(os.path.join(parameter_path, "beta.txt"), 'w')
beta_file.write(str(hyper_params.ppo.beta))
beta_file.close()

print("KL divergence:", kl_divergence_final)
print("Done one update!")
del buffer
del single_buffer_list
del previous_rewards
del current_rewards

# Test policy
# num_of_test = 5000
# reward_list = np.zeros((hyper_params.number_of_robot, num_of_test))
# for t in range(num_of_test):
#     print("Run policy at t = {}".format(round(t*hyper_params.setup.sample_time, 1)))
#     cmd_vel_list = []
#     for i in range(hyper_params.number_of_robot):
#         # Collect the sample data
#         linear_vel, angular_vel, log_prob, reward, done = agents[i].run_policy(actor)
#         # print(linear_vel_vector[linear_vel.item()])
#         # print(angular_vel_vector[angular_vel.item()])
#         # print("Action prob of robot {}:{}".format(i, log_prob.exp().item()*100))
#         reward_list[i][t-1] = reward
#         # Publish the velocity to robot
#         cmd_vel = Twist()
#         cmd_vel.linear.x = linear_vel_vector[linear_vel.item()]
#         cmd_vel.angular.z = angular_vel_vector[angular_vel.item()]
#         cmd_vel_list.append(cmd_vel)
        
#     for i in range(hyper_params.number_of_robot):
#         agents[i].cmd_vel_pub.publish(cmd_vel_list[i])
#     rate.sleep()
    
# total_reward = reward_list.sum(axis=1)
# for i in range(hyper_params.number_of_robot):
#     print("Total reward of robot {}: {}".format(i, total_reward[i]))