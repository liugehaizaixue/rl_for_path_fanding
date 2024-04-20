import torch
from env import make_env
import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 4 * 4, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
    
# 加载保存的模型
loaded_agent = torch.load('save.pt')

# 创建环境
env = make_env()  # 请确保 make_env 函数正确创建了你的环境，或者你可以使用其他方式创建环境

# 运行一个 episode 进行推理
obs , info = env.reset(if_render=True)
done = False
total_reward = 0

while not done:
    # 获取动作
    with torch.no_grad():
        obs_tensor = torch.tensor(obs).unsqueeze(0).to('cuda')  # 将观察值转换为张量，并添加批次维度
        action, _, _, _ = loaded_agent.get_action_and_value(obs_tensor)
    action = action.item()  # 将动作张量转换为标量

    # 执行动作
    next_obs, reward, done, _ , _= env.step(action)

    # 更新观察值和总奖励
    obs = next_obs
    total_reward += reward

print("Total reward:", total_reward)

# 关闭环境
env.close()
