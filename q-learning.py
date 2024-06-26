import numpy as np
from env import get_map_array
import gymnasium as gym

map_array = get_map_array()
env = gym.make("PathFinding-v0", map_array = map_array)
obs, _ = env.reset()

# 定义参数
alpha = 0.1  # 学习率
gamma = 0.99  # 折扣因子
epsilon = 0.3  # ε-greedy策略中的ε
num_episodes = 1e4

class QLearningAgent:
    def __init__(self, map_size, action_size, alpha, gamma, epsilon):
        self.q_table = np.zeros((map_size[0], map_size[1], action_size))
        self.actions = [0,1,2,3]
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def choose_action(self, pos):
        # print(pos)
        if np.random.uniform() < self.epsilon:  # ε-greedy策略
            action_index = np.random.choice(len(self.actions))
        else:
            action_index = np.argmax(self.q_table[pos[0]][pos[1]])
        
        # print(action_index)
        return action_index
    
    def update_q_table(self, pos, action, reward, next_pos):
        next_max = np.max(self.q_table[next_pos[0]][next_pos[1]])
        self.q_table[pos[0]][pos[1]][action] += self.alpha * (reward + self.gamma * next_max - self.q_table[pos[0]][pos[1]][action])

    def get_pos(self,obs):
        obs = np.array(obs)
        indices = np.where(obs[0] == 2)
        pos = list(zip(indices[0], indices[1]))
        return pos[0]

policy = QLearningAgent(env.map_size, env.action_size, alpha, gamma, epsilon)

def train():
    for episode in range(int(num_episodes)):
        print(f"current episode: {episode} / {num_episodes}")
        obs , info  = env.reset()
        pos = policy.get_pos(obs)
        while True:
            action = policy.choose_action(pos)
            next_obs , reward , done , truncated , info = env.step(action)
            next_pos = policy.get_pos(next_obs)
            policy.update_q_table(pos, action, reward, next_pos)
            pos = next_pos
            if done:  
                break

def eval():
    obs , info = env.reset()
    env.render()
    pos = policy.get_pos(obs)
    while True:
        action = np.argmax(policy.q_table[pos[0]][pos[1]])
        next_obs , reward , done , truncated, info= env.step(action)
        env.render()
        next_pos = policy.get_pos(next_obs)
        pos = next_pos
        if done or truncated:  
            break

train()
eval()