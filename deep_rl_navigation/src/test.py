#!/usr/bin/env python

import torch
import torch.nn as nn
from deep_rl_navigation.actor_critic import *
from deep_rl_navigation.ultis import *


actor = Actor(3, 21)
critic = Critic(3)

laser = torch.randn((3, 3, 541))
goal = torch.randn((3, 2))
vel = torch.randn((3, 2))

linear_vel, angular_vel, log_prob, vel_probs = actor.get_action(laser, goal, vel)

print(log_prob)
new_log_prob, new_vel_probs = actor.evaluate(laser, goal, vel, linear_vel, angular_vel) 
value = critic.get_value(laser, goal, vel)

print(value)