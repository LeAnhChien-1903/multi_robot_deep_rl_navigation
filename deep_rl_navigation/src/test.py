from deep_rl_navigation.actor_critic import *
from deep_rl_navigation.ultis import *

actor = ActorDiscrete(20)
critic = CriticDiscrete()


laser_obs = torch.randn((1, 3, 541))
goal_obs = torch.randn(2)
vel_obs = torch.randn(2)

_, _, _, p1, p2 = actor.get_action(laser_obs, goal_obs, vel_obs)
laser_obs = torch.randn((1, 3, 541))
goal_obs = torch.randn(2)
vel_obs = torch.randn(2)
_, _, _, q1, q2 = actor.get_action(laser_obs, goal_obs, vel_obs)

print(p2.shape[0] * p1.shape[0])
print(p1, p2, q1, q2, sep = '\n', )
print("KL divergence:", calculateMultiKLDivergence(p1, p2, q1, q2))