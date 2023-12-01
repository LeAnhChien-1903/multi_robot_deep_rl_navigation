#!/usr/bin/env python

import numpy as np
import math
import os

import torch
from torch.distributions.kl import kl_divergence
from geometry_msgs.msg import Twist

from deep_rl_navigation_2021.ultis import *
from deep_rl_navigation_2021.actor_critic import *
from deep_rl_navigation_2021.agent import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parameter_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation_2021/weights_bias"

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

if not os.path.exists(os.path.join(parameter_path, "beta.txt")):
    np.savetxt(os.path.join(parameter_path, "beta.txt"), np.ones(1), fmt='%.2f')

if not os.path.exists(os.path.join(parameter_path, "cumulative_reward.txt")):
    np.savetxt(os.path.join(parameter_path, "cumulative_reward.txt"), np.zeros((2, hyper_params.number_of_robot)), fmt='%.2f')

# Load actor and critic parameters
actor = Actor(num_of_laser_channels= hyper_params.setup.num_observations, num_of_actions= hyper_params.number_of_action).to(device)
if not os.path.exists(os.path.join(parameter_path, 
                                    "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                                        hyper_params.setup.num_laser_ray, 
                                                                                        hyper_params.number_of_action))):
    torch.save(actor.state_dict(), os.path.join(parameter_path, 
                                                "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                                    hyper_params.setup.num_laser_ray, 
                                                                                    hyper_params.number_of_action)))
    print("Save initialized actor model!")
else:
    actor.load_state_dict(torch.load(os.path.join(parameter_path, 
                                                "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                                    hyper_params.setup.num_laser_ray, 
                                                                                    hyper_params.number_of_action)), map_location= 'cpu'))
    print("Load actor model!")

critic = Critic(num_of_laser_channels= hyper_params.setup.num_observations).to(device)
if not os.path.exists(os.path.join(parameter_path, 
                                    "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                        hyper_params.setup.num_laser_ray))):
    torch.save(critic.state_dict(), os.path.join(parameter_path, 
                                                "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                                    hyper_params.setup.num_laser_ray)))
    print("Save initialized critic model!")
else:
    critic.load_state_dict(torch.load(os.path.join(parameter_path, 
                                                "critic_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, 
                                                                                    hyper_params.setup.num_laser_ray)), map_location= 'cpu'))
    print("Load critic model!")


# Create the optimizers
actor_optimizer = torch.optim.Adam(actor.parameters(), lr= hyper_params.ppo.lr_theta_1st)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr= hyper_params.ppo.lr_phi)

rate = rospy.Rate(float(1/hyper_params.setup.sample_time))

for i in range(300):
# Load beta parameters
    beta_file = open(os.path.join(parameter_path, "beta.txt"), 'r')
    hyper_params.ppo.beta = float(beta_file.read())
    beta_file.close()

    print("Current beta:", hyper_params.ppo.beta)

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
            sample: dict = agents[i].step(actor, critic, device)

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
    kl_divergence_final = 0.0
    count = 1
    old_probs = torch.matmul(buffer.linear_probs_batch.reshape(hyper_params.batch_size, 
                                                        hyper_params.number_of_action, 1),
                                buffer.angular_probs_batch.reshape(hyper_params.batch_size, 
                                                        1, hyper_params.number_of_action)).reshape(hyper_params.batch_size,
                                                                                                    hyper_params.number_of_action**2)
    for j in range(hyper_params.ppo.E_phi):
        print("Policy optimizer:" , count)    
        new_log_prob_batch, new_linear_probs, new_angular_probs = actor.evaluate(buffer.laser_obs_batch, buffer.orient_obs_batch,
                                                                            buffer.dist_obs_batch, buffer.vel_obs_batch,
                                                                            buffer.linear_vel_batch, buffer.angular_vel_batch)
        
        new_probs = torch.matmul(new_linear_probs.reshape(hyper_params.batch_size, 
                                                        hyper_params.number_of_action, 1),
                                new_angular_probs.reshape(hyper_params.batch_size, 
                                                        1, hyper_params.number_of_action)).reshape(hyper_params.batch_size,
                                                                                                    hyper_params.number_of_action**2)
        log_ratio = new_log_prob_batch - buffer.log_prob_batch
        ratio = log_ratio.exp()
        
        kl_div = ((old_probs * (old_probs / new_probs).log()).sum(dim = 1)).mean()
        kl_divergence_final = kl_div.item()
        count += 1
        if (kl_div > 4 * hyper_params.ppo.KL_target): 
            break
        
        loss1 = -buffer.advantage_batch * ratio
        loss2 = hyper_params.ppo.beta * kl_div
        loss3 = -hyper_params.ppo.xi * torch.pow(torch.max(torch.zeros(1).to(device), kl_div - 2 * hyper_params.ppo.KL_target), 2)
        loss = loss1 + loss2 + loss3
        actor_loss = loss.sum()
        
        print("Policy loss:", actor_loss.item())
        print("kl_divergence:", kl_divergence_final)
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        

    torch.save(actor.state_dict(), os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(  hyper_params.setup.num_observations, 
                                                                                                        hyper_params.setup.num_laser_ray, 
                                                                                                        hyper_params.number_of_action)))

    for k in range(hyper_params.ppo.E_v):
        value_loss = torch.zeros(1).to(device)
        print("Critic optimizer:", k+1)
        for single_buffer in single_buffer_list:
            robot_loss = torch.zeros(1).to(device)
            # Calculate the value list
            value_batch = critic.get_value(single_buffer.laser_obs_mini_batch, single_buffer.orient_obs_mini_batch, single_buffer.dist_obs_mini_batch, single_buffer.vel_obs_mini_batch)
                    
            for t in range(hyper_params.mini_batch_size):
                temp_loss = torch.zeros(1).to(device)
                if t < hyper_params.mini_batch_size:
                    for t_ in range(t + 1, hyper_params.mini_batch_size):
                        temp_loss += hyper_params.ppo.gamma**(t_ - t) * single_buffer.reward_mini_batch[t_]
                robot_loss = robot_loss + (temp_loss - value_batch[t][0])**2
            value_loss = value_loss + robot_loss
        
        # value_loss = - value_loss

        print("Value loss:", value_loss.item())
        critic_optimizer.zero_grad()
        value_loss.backward()
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
