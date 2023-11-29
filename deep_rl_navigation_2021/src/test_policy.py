#!/usr/bin/env python

import numpy as np
import math
import os

import torch

from geometry_msgs.msg import Twist

from deep_rl_navigation_2021.ultis import *
from deep_rl_navigation_2021.actor_critic import *
from deep_rl_navigation_2021.agent import *

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

parameter_path = "/home/leanhchien/deep_rl_ws/src/multi_robot_deep_rl_navigation/deep_rl_navigation_2021/parameters"

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
actor = Actor(num_of_laser_channels= hyper_params.setup.num_observations, num_of_actions= hyper_params.number_of_action).to(device)
if not os.path.exists(os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray, hyper_params.number_of_action))):
    torch.save(actor.state_dict(), os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray, hyper_params.number_of_action)))
    print("Save initialized actor model!")
else:
    actor.load_state_dict(torch.load(os.path.join(parameter_path, "actor_{}_{}_{}_parameters.pt".format(hyper_params.setup.num_observations, hyper_params.setup.num_laser_ray, hyper_params.number_of_action)), map_location= 'cpu'))
    print("Load actor model!")


critic = Critic(num_of_laser_channels= hyper_params.setup.num_observations).to(device)
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

for i in range(10):
    rate.sleep()

# Test policy
num_of_test = 5000
reward_list = np.zeros((hyper_params.number_of_robot, num_of_test))
for t in range(num_of_test):
    print("Run policy at t = {}".format(round(t*hyper_params.setup.sample_time, 1)))
    cmd_vel_list = []
    for i in range(hyper_params.number_of_robot):
        # Collect the sample data
        linear_vel, angular_vel, log_prob, reward, done = agents[i].run_policy(actor, device)
        # print(linear_vel_vector[linear_vel.item()])
        # print(angular_vel_vector[angular_vel.item()])
        # print("Action prob of robot {}:{}".format(i, log_prob.exp().item()*100))
        reward_list[i][t-1] = reward
        # Publish the velocity to robot
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_vel_vector[linear_vel.item()]
        cmd_vel.angular.z = angular_vel_vector[angular_vel.item()]
        cmd_vel_list.append(cmd_vel)
        
    for i in range(hyper_params.number_of_robot):
        agents[i].cmd_vel_pub.publish(cmd_vel_list[i])
    rate.sleep()
    
total_reward = reward_list.sum(axis=1)
for i in range(hyper_params.number_of_robot):
    print("Total reward of robot {}: {}".format(i, total_reward[i]))