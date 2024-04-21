import torch
from env import get_map_array
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from ppo import Agent , make_env
import gymnasium as gym

# Load checkpoint
checkpoint = torch.load("./ppo.cleanrl_model")
args = checkpoint["args"]
args = type("Args", (), args)

map_array = get_map_array()
# Init env and reset
env = make_env(args.env_id, map_array , "human")()
obs, _ = env.reset()
env.render()

action_space_shape = (
        (env.action_space.n,)
        if isinstance(env.action_space, gym.spaces.Discrete)
        else tuple(env.action_space.nvec)
    )
agent = Agent(env.action_space).to('cuda')
agent.load_state_dict(checkpoint["model_weights"])

total_reward = 0
done = False
truncated = False
while not done or not truncated:
    # 获取动作
    with torch.no_grad():
        obs_tensor = torch.tensor(obs).unsqueeze(0).to('cuda')  # 将观察值转换为张量，并添加批次维度
        action, _, _, _ = agent.get_action_and_value(obs_tensor)
    action = action.item()  # 将动作张量转换为标量
    print(action)

    # 执行动作
    next_obs, reward, done, truncated , _= env.step(action)
    env.render()
    # 更新观察值和总奖励
    obs = next_obs
    total_reward += reward

print("Total reward:", total_reward)

# 关闭环境
env.close()
