from util import utils
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from load_map import MAPS_REGISTRY
import random
from copy import deepcopy

COLORS = ['#FFFFFF', '#000000', '#6600CC', '#FF0000']

gym.register(
    id="PathFinding-v0",
    entry_point="env:PathFindingEnv",
    max_episode_steps=512,
)

class PathFindingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}
    def __init__(self, map_array, render_mode="human"):
        self.pos = None
        self.target = None
        self.original_map = deepcopy(map_array)
        self.map_array = map_array
        self.map_size = self.map_array.shape
        self.render_mode = render_mode
        self.start_render = False
        self.action_size = 4
        self.obs_size = 65
        self.max_episode_steps = 512
        self.current_step = 0
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 3.0, shape=(1, 64, 64), dtype=np.float32)
        """ 
        ---------------> y
        |
        |
        |
        ↓
        x 
        """

    def get_action(self, action):
        if action == 0: # 上
            return [-1,0]
        elif action == 1: #下
            return [1,0]
        elif action == 2: #左
            return [0,-1]
        elif action == 3: #右
            return [0,1]

    def step(self, action):
        self.current_step = self.current_step+1
        _action = self.get_action(action)
        
        if self.pos[0]+_action[0] >= 0 and self.pos[0]+_action[0] < self.map_size[0] and \
            self.pos[1]+_action[1] >= 0 and self.pos[1]+_action[1] < self.map_size[1]:
            if self.map_array[self.pos[0]+_action[0]][self.pos[1]+_action[1]] != 1:
                self.map_array[self.pos[0]][self.pos[1]] = 0
                self.pos = (self.pos[0]+_action[0] , self.pos[1]+_action[1])
                self.map_array[self.pos[0]][self.pos[1]] = 2
                reward = -0.001
            else:
                # 撞障碍
                reward = -0.002
        else:
            # 撞边界
            reward = -0.002
        
        if self.pos == self.target:
            reward = 1
            done = True
        else:
            done = False
        
        if self.current_step > self.max_episode_steps:
            truncated = True
        else:
            truncated = False

        info = {}
        obs = np.expand_dims(self.map_array, axis=0).astype(np.float32)
        return obs , reward , done , truncated, info


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.start_render = False
        self.current_step = 0
        start , target = utils.generate_start_target(self.map_array, seed)
        self.map_array = deepcopy(self.original_map)
        self.map_array[start[0]][start[1]] = 2
        self.map_array[target[0]][target[1]] = 3
        self.pos = start
        self.target = target

        obs = np.expand_dims(self.map_array, axis=0).astype(np.float32)
        info = {}
        return obs , info
    
    def render(self):
        if not self.start_render:
            plt.ion()
            cmap = ListedColormap(COLORS)
            self.rendering = plt.imshow(self.map_array, cmap=cmap, interpolation='none')
            plt.axis('off')
            plt.show()
            self.start_render = True
        else:
            self.rendering.set_array(self.map_array)
            # 刷新图形
            plt.pause(1)
    
    def close(self):
        if self.map_array is not None:
            self.map_array = None
        plt.close()
        


def make_env():
    random_key = random.choice(list(MAPS_REGISTRY.keys()))
    map_str = MAPS_REGISTRY[random_key]
    map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])
    return PathFindingEnv(map_array=map_array)


def get_map_array():
    map_str = MAPS_REGISTRY['sc1-AcrosstheCape']
    map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])
    return map_array

if __name__ == "__main__":
    random_key = random.choice(list(MAPS_REGISTRY.keys()))
    map_str = MAPS_REGISTRY[random_key]
    map_array = np.array([[1 if c == '#' else 0 for c in line] for line in map_str.strip().split('\n')])
    env = gym.make("PathFinding-v0", map_array = map_array)
    o, _ = env.reset()
    env.render()
    done = False
    for i in range(20):
        action = env.action_space.sample()
        o, r, done, _, _ = env.step(action)
        env.render()
    env.close()