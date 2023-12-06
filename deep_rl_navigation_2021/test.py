import torch
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

x = np.array([1, 2, 3])
print(x+ 1)


# Calculate value with for (do not delete)
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