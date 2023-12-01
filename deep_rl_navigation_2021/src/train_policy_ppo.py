#!/usr/bin/env python

import numpy as np
import math
import os

import torch

from geometry_msgs.msg import Twist

from deep_rl_navigation_2021.ultis import *
from deep_rl_navigation_2021.actor_critic import *
from deep_rl_navigation_2021.agent_ppo import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parameter_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation_2021/ppo_train_parameters"

# Initialize robot to collect data
rospy.init_node("deep_rl_node", anonymous=False)

# Initialize the hyper parameters
hyper_params = HyperParameters()

# Create linear and angular vector
linear_vel_vector = np.linspace(hyper_params.setup.min_linear_velocity, 
                                hyper_params.setup.max_linear_velocity, 
                                hyper_params.number_of_action)
angular_vel_vector = np.linspace(hyper_params.setup.min_angular_velocity, 
                                hyper_params.setup.max_angular_velocity, 
                                hyper_params.number_of_action)

# Create agents and single buffer list
agents = []
for i in range(hyper_params.number_of_robot):
    agents.append(Agent('/robot_{}'.format(i), hyper_param= hyper_params))

if not os.path.exists(os.path.join(parameter_path, "cumulative_reward.txt")):
    np.savetxt(os.path.join(parameter_path, "cumulative_reward.txt"), np.zeros((2, hyper_params.number_of_robot)), fmt='%.2f')

# Load actor and critic parameters
policy = ActorCritic(num_of_laser_channels= hyper_params.setup.num_observations, num_of_actions= hyper_params.number_of_action).to(device)
if not os.path.exists(os.path.join(parameter_path, 
                                    "policy_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                            hyper_params.setup.num_laser_ray, 
                                                                            hyper_params.number_of_action))):
    torch.save(policy.state_dict(), 
                os.path.join(parameter_path, 
                "policy_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                    hyper_params.setup.num_laser_ray, 
                                                    hyper_params.number_of_action)))
    print("Save initialized model!")
else:
    policy.load_state_dict(torch.load(os.path.join(parameter_path, 
                                    "policy_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                        hyper_params.setup.num_laser_ray, 
                                                                        hyper_params.number_of_action)), map_location= device))
    print("Load model!")


# Create the optimizers
policy_optimizer = torch.optim.Adam(policy.parameters(), lr= 0.0003)
rate = rospy.Rate(float(1/hyper_params.setup.sample_time))

# for i in range(300):
single_buffer_list = []
for i in range(hyper_params.number_of_robot):
    # agents[i].randomPose()    
    single_buffer_list.append(SingleBuffer(hyper_params.mini_batch_size, hyper_params.setup, hyper_params.number_of_action, device))
# Check data 
for i in range(10):
    rate.sleep()

# Collect the data
for t in range(hyper_params.mini_batch_size + 1):
    print("Collecting data at t = {}".format(round(t*hyper_params.setup.sample_time, 1)))
    cmd_vel_list = []
    for i in range(hyper_params.number_of_robot):
        # Collect the sample data
        sample: dict = agents[i].step(policy, device)

        # Update sampled data
        single_buffer_list[i].update(t, sample)

        # Publish the velocity to robot
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel_vector[sample['linear'][0].item()]
        cmd_vel.angular.z = angular_vel_vector[sample['angular'][0].item()]
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
    current_rewards[i] = sum_reward.to('cpu').numpy()

print("Total reward:", current_rewards.sum())

total_reward = np.concatenate((previous_rewards, current_rewards.reshape(1, hyper_params.number_of_robot)), axis=0)
np.savetxt(os.path.join(parameter_path, "cumulative_reward.txt"), total_reward, fmt='%.2f')

# # Estimate the advantage function by using GAE
for i in range(hyper_params.number_of_robot):
    single_buffer_list[i].advantageEstimator(hyper_params.ppo.gamma, hyper_params.ppo.lambda_)

# Create buffer from single buffer list
buffer = Buffer(single_buffer_list)
# Start training
count = 1
old_probs = torch.matmul(buffer.linear_probs_batch.reshape(hyper_params.batch_size, 
                                                            hyper_params.number_of_action, 1),
                        buffer.angular_probs_batch.reshape(hyper_params.batch_size, 
                                                            1, hyper_params.number_of_action)).reshape(hyper_params.batch_size,
                                                                                                hyper_params.number_of_action**2)
for j in range(hyper_params.ppo.E_phi):
    print("Policy optimizer:" , count)    
    new_log_prob_batch, new_linear_probs, new_angular_probs, new_value_batch = policy.evaluate(buffer.laser_obs_batch, buffer.orient_obs_batch,
                                                                        buffer.dist_obs_batch, buffer.vel_obs_batch,
                                                                        buffer.linear_vel_batch, buffer.angular_vel_batch)
    
    new_probs = torch.matmul(new_linear_probs.reshape(hyper_params.batch_size, 
                                                    hyper_params.number_of_action, 1),
                            new_angular_probs.reshape(hyper_params.batch_size, 
                                                    1, hyper_params.number_of_action)).reshape(hyper_params.batch_size,
                                                                                                hyper_params.number_of_action**2)
    entropy_batch = -torch.sum(new_probs * torch.log(new_probs), dim = 1)
    log_ratio = new_log_prob_batch.to(device) - buffer.log_prob_batch
    
    ratio = log_ratio.exp()
    
    count += 1
    
    loss1 = ratio * buffer.advantage_batch
    loss2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * buffer.advantage_batch

    actor_loss  = - torch.min(loss1, loss2).mean()
    critic_loss = (buffer.advantage_batch + buffer.value_batch - new_value_batch).pow(2).mean()
    
    loss = (0.5 * critic_loss + actor_loss - 0.001 * entropy_batch)
    print("Policy loss:", actor_loss.item())
    policy_optimizer.zero_grad()
    actor_loss.backward()
    policy_optimizer.step()
    

torch.save(policy.state_dict(), 
            os.path.join(parameter_path, 
            "policy_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                    hyper_params.setup.num_laser_ray, 
                                                    hyper_params.number_of_action)))

print("Done one update!")
